# projects/sol_fn_miniloop/callbacks/forge_callback.py
import os
import re
import shutil
import subprocess
import json
from typing import Dict
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


def _normalize_sig_text(s: str) -> str:
    if not s:
        return ''
    s = s.replace('\n', ' ')
    s = ' '.join(s.split())
    return s.strip()


def _extract_signature_head_text(full_signature: str) -> str:
    if not full_signature:
        return ''
    import re

    s = full_signature.strip()
    m = re.search(r"\bfunction\b\s+[A-Za-z0-9_]+\s*\(.*?\)", s, flags=re.DOTALL)
    return (m.group(0).strip() if m else '').strip()


def _canonicalize_function_signature(full_signature: str) -> tuple:
    if not full_signature:
        return None, None
    import re

    head = _extract_signature_head_text(full_signature) or full_signature
    mm = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", head)
    func_name = mm.group(1) if mm else None
    if not func_name:
        return None, None
    m2 = re.search(r"function\s+%s\s*\((.*?)\)" % re.escape(func_name), head, flags=re.DOTALL)
    raw_params = (m2.group(1) if m2 else '').strip()
    if not raw_params:
        return func_name, f"function {func_name}()"
    param_types = []
    for seg in raw_params.split(','):
        seg = seg.strip()
        if not seg:
            continue
        tokens = [t for t in seg.split() if t]
        if len(tokens) <= 1:
            ptype = tokens[0] if tokens else ''
        else:
            ptype = ' '.join(tokens[:-1])
        ptype = re.sub(r"\s+", " ", ptype).strip().replace(' ', '')
        if ptype:
            param_types.append(ptype)
    return func_name, f"function {func_name}({','.join(param_types)})"


def _canonicalize_source_signature_head(lines: List[str], start_idx: int) -> str:
    if start_idx < 0 or start_idx >= len(lines):
        return ''
    hdr = ''
    j = start_idx
    while j < len(lines) and '{' not in hdr and ';' not in hdr:
        ln = lines[j]
        ln = ln.split('//', 1)[0]
        hdr += ln.strip() + ' '
        if '{' in ln or ';' in ln:
            break
        j += 1
    hdr = hdr.strip()
    if not hdr:
        return ''
    return _normalize_sig_text(hdr)


def _locate_function_in_lines(lines: List[str], full_signature: str, expected_start: Optional[int]) -> tuple:
    func_name, canonical_head = _canonicalize_function_signature(full_signature)
    if not func_name:
        return None, None

    full_sig_norm = _normalize_sig_text(full_signature)
    head_norm = _normalize_sig_text(_extract_signature_head_text(full_signature))

    full_sig_candidates = []
    head_candidates = []
    canonical_candidates = []

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if 'function' in line_stripped:
            hdr = ''
            j = i - 1
            while j < len(lines) and '{' not in hdr and ';' not in hdr:
                ln = lines[j]
                ln = ln.split('//', 1)[0]
                hdr += ln.strip() + ' '
                if '{' in ln or ';' in ln:
                    break
                j += 1

            hdr_norm = _normalize_sig_text(hdr)
            if full_sig_norm and full_sig_norm in hdr_norm:
                full_sig_candidates.append(i)
            if head_norm and head_norm in hdr_norm:
                head_candidates.append(i)

        if canonical_head and f"function {func_name}" in line_stripped:
            src_head = _canonicalize_source_signature_head(lines, i - 1)
            if src_head and src_head.replace(' ', '') == canonical_head.replace(' ', ''):
                canonical_candidates.append(i)

    def choose_nearest(cands: List[int]) -> Optional[int]:
        if not cands:
            return None
        if expected_start is None:
            return cands[0]
        return min(cands, key=lambda v: abs(v - expected_start))

    start_line = choose_nearest(full_sig_candidates) or choose_nearest(head_candidates) or choose_nearest(canonical_candidates)
    if not start_line:
        return None, None

    brace = 0
    in_fn = False
    end_line = None
    for idx in range(start_line - 1, len(lines)):
        ln = lines[idx]
        if not in_fn and '{' in ln:
            in_fn = True
        if in_fn:
            brace += ln.count('{') - ln.count('}')
            if brace <= 0:
                end_line = idx + 1
                break
    return start_line, end_line


class ForgeCallback(Callback):
    """
    Run `forge build` and `forge test --gas-report`, then append results as a user message.
    - Runs inside the configured output_dir.
    - If forge is missing, instructs installation.
    - Saves test reports to forge_test_report.txt and forge_test_report.json.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.test_report_txt = 'forge_test_report.txt'
        self.test_report_json = 'forge_test_report.json'
        # Track latest build/test status for stop-condition coordination
        self.last_build_ok: Optional[bool] = None
        self.last_test_ok: Optional[bool] = None
        self._pending_forge_run: bool = False
        self._pending_work_subfolder: Optional[str] = None
        self._task_work_subfolder: Optional[str] = None
        self._last_failed_tests_fp: Optional[str] = None
        self._same_failed_tests_rounds: int = 0
        self._max_forge_rounds: int = 4
        self._forge_rounds: int = 0

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        # Reset stall detection state per task/run to avoid cross-task pollution.
        self._last_failed_tests_fp = None
        self._same_failed_tests_rounds = 0
        self._forge_rounds = 0
        self._task_work_subfolder = None
        await self.file_system.connect()

    @contextmanager
    def chdir_context(self, folder: Optional[str] = None):
        path = os.getcwd()
        work_dir = getattr(self.config, 'output_dir', 'output')
        if folder:
            work_dir = os.path.join(work_dir, folder)
        try:
            if not os.getcwd().endswith(work_dir):
                os.makedirs(work_dir, exist_ok=True)
                os.chdir(work_dir)
            yield
        finally:
            if os.getcwd() != path:
                os.chdir(path)

    def _run_cmd(self, cmd: List[str], timeout: Optional[int] = None) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=True
            )
            return True, (result.stdout or '') + (result.stderr or '')
        except subprocess.CalledProcessError as e:
            out = (e.stdout or '') + (e.stderr or '')
            if not out:
                out = str(e)
            return False, out
        except FileNotFoundError as e:
            return False, f'FileNotFoundError: {e}'

    def _forge_available(self) -> bool:
        return shutil.which('forge') is not None

    def _parse_workdir_from_messages(self, messages: List[Message]) -> Optional[str]:
        # Prefer explicit WORKDIR tag from TaskPrep user message
        for m in reversed(messages):
            if m.role == 'user' and m.content:
                for line in m.content.splitlines():
                    if line.strip().startswith('WORKDIR:'):
                        _, _, rest = line.partition(':')
                        wd = rest.strip()
                        if wd:
                            return wd
                # Fallback: raw JSON payload with task_id
                try:
                    obj = json.loads(m.content)
                    if isinstance(obj, dict) and 'task_id' in obj:
                        return f"task_{obj['task_id']}"
                except Exception:
                    pass
        return None

    def _pick_single_task_dir(self) -> Optional[str]:
        # If only one task_* folder exists under output_dir, use it.
        base = getattr(self.config, 'output_dir', 'output')
        try:
            entries = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d.startswith('task_')]
            if len(entries) == 1:
                return entries[0]
        except Exception:
            return None
        return None

    def _truncate(self, s: str, max_len: int) -> str:
        if not s:
            return ''
        if len(s) <= max_len:
            return s
        return s[:max_len] + '\n... (truncated)'

    def _filter_compile_errors(self, compiler_output: str) -> str:
        if not compiler_output:
            return ''
        lines = compiler_output.split('\n')
        filtered_lines = []
        skip_until_next_section = False
        start = False

        for i, line in enumerate(lines):
            if 'Compiler run failed' in line:
                filtered_lines.append(line)
                start = True
                continue
            if not start:
                continue

            if line.startswith('Warning ('):
                skip_until_next_section = True
                continue

            if line.startswith('Error ('):
                skip_until_next_section = False
                filtered_lines.append(line)
                continue

            if skip_until_next_section:
                if i + 1 < len(lines) and (
                    lines[i + 1].startswith('Warning (')
                    or lines[i + 1].startswith('Error (')
                ):
                    skip_until_next_section = False
                continue

            if not skip_until_next_section:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines).strip()

    def _parse_forge_test_stdout(self, captured_stdout: str) -> dict:
        if not captured_stdout:
            return {}
        if 'Compiler run failed' in captured_stdout:
            return {'compile_error': self._filter_compile_errors(captured_stdout) or captured_stdout}

        summary = {}
        m = re.search(r":\s*(\d+)\s*tests\s*passed,\s*(\d+)\s*failed.*?\((\d+)\s*total\s*tests\)", captured_stdout)
        if m:
            summary['passed'] = int(m.group(1))
            summary['failed'] = int(m.group(2))
            summary['total'] = int(m.group(3))

        fails = {}
        parts = re.split(r"Failing tests:\s*", captured_stdout, maxsplit=1)
        if len(parts) >= 2:
            section = parts[1]
            pattern = re.compile(
                r"^\[FAIL:[^\n]*?\]\s+([a-zA-Z0-9_]+\(.*?\))\s*(?:\(runs:.*)?(?:\s*\(gas:\s*\d+\))?$",
                re.MULTILINE,
            )
            for mm in pattern.finditer(section):
                full_line = mm.group(0)
                test_name = mm.group(1)
                cleaned = re.sub(r"\(runs:.*", "", full_line).strip()
                cleaned = re.sub(r"\(gas:\s*\d+\).*", "", cleaned).strip()
                fails[test_name] = cleaned

        if fails:
            summary['fails'] = fails
        return summary

    def _normalize_fail_reason_for_fp(self, s: str) -> str:
        if not s:
            return ''
        s = s.strip()
        s = re.sub(r";\s*counterexample:.*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*counterexample:.*$", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\bcalldata=0x[0-9a-fA-F]+", "calldata=0x...", s)
        s = re.sub(r"\bargs=\[.*\]", "args=[...]", s)

        # For stall detection, we want stability across fuzz/counterexamples. Replace long hex literals
        # (addresses/bytes32/etc) with a placeholder so the fingerprint reflects the *reason pattern*.
        s = re.sub(r"0x[0-9a-fA-F]{16,}", "0x...", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > 260:
            s = s[:260] + '...'
        return s

    def _fingerprint_failed_tests(self, parsed_test: dict) -> Optional[str]:
        fails = (parsed_test or {}).get('fails')
        if not isinstance(fails, dict) or not fails:
            return None
        keys = [str(k) for k in sorted(fails.keys(), key=lambda x: str(x))]
        rows: List[str] = []
        for k in keys:
            v = fails.get(k)
            v2 = self._normalize_fail_reason_for_fp(str(v) if v is not None else '')
            rows.append(f"{k} | {v2}" if v2 else k)
        return f"count={len(keys)}\n" + "\n".join(rows)

    def _build_test_hints(self, parsed_test: dict, raw_test_out: str) -> List[str]:
        hints: List[str] = []
        fails = (parsed_test or {}).get('fails')
        if not isinstance(fails, dict) or not fails:
            return hints

        joined = '\n'.join(str(v) for v in fails.values())
        if 'panic: arithmetic underflow or overflow (0x11)' in joined:
            hints.append('- Detected Solidity panic 0x11 (arithmetic overflow/underflow). If the function is intended to be modulo-2^256 arithmetic, use unchecked { } around the arithmetic instead of reverting.')
            hints.append('- If the function is intended to revert on overflow, then the tests likely expect wrap-around; align implementation semantics with the tests.')
        elif 'panic:' in joined:
            hints.append('- Detected Solidity panic (built-in revert). Identify whether the test expects revert or expects a return value under the counterexample inputs.')

        if 'revert:' in joined:
            hints.append('- Detected explicit revert. If fuzz tests include large counterexamples, adding require/revert may cause perpetual failures; consider matching expected wrap-around behavior if applicable.')

        raw = raw_test_out or ''
        # Avoid regex that stops at the first ']' because forge counterexamples may contain nested brackets.
        idx = raw.lower().find('counterexample:')
        if idx >= 0:
            sub = raw[idx:]
            aidx = sub.lower().find('args=')
            if aidx >= 0:
                sub2 = sub[aidx + len('args='):]
                # Take until end-of-line to preserve full args even if it includes nested brackets.
                eol = sub2.find('\n')
                seg = (sub2[:eol] if eol >= 0 else sub2).strip()
                seg = re.sub(r"\s+", " ", seg)
                if seg:
                    if len(seg) > 260:
                        seg = seg[:260] + '...'
                    hints.append(f"- Counterexample args: {seg}")

        if not hints:
            hints.append('- Review the failing test name and failure line. Prefer minimal changes that make the function behave correctly under the provided counterexample inputs.')
        return hints[:6]

    def _extract_build_errors(self, build_out: str) -> str:
        """Extract key compilation errors from forge build output"""
        if not build_out:
            return ''
        
        errors = []
        lines = build_out.split('\n')
        
        for i, line in enumerate(lines):
            # Look for compilation error patterns
            if 'Error (' in line and ':' in line:
                errors.append(line.strip())
                # Add context lines (next 1-2 lines)
                for j in range(1, 3):
                    if i + j < len(lines):
                        context_line = lines[i + j].strip()
                        if context_line and ('-->' in context_line or '|' in context_line or '^' in context_line):
                            errors.append(context_line)
                        elif context_line and not context_line.startswith('Error:'):
                            break
            elif line.strip().startswith('Compiler run failed'):
                errors.append(line.strip())
        
        return '\n'.join(errors) if errors else ''

    def _extract_test_failures(self, test_out: str) -> str:
        """Extract key test failure information from forge test output"""
        if not test_out:
            return ''
        
        failures = []
        lines = test_out.split('\n')
        in_failing_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Look for failing tests section
            if 'Failing tests:' in line_stripped or 'Encountered' in line_stripped and 'failing test' in line_stripped:
                in_failing_section = True
                failures.append(line_stripped)
                continue
            
            # Collect failure details
            if in_failing_section:
                if line_stripped.startswith('[FAIL:') or line_stripped.startswith('Encountered'):
                    failures.append(line_stripped)
                elif line_stripped.startswith('Suite result:') or not line_stripped:
                    in_failing_section = False
        
        return '\n'.join(failures) if failures else ''

    def _parse_target_file_and_range(self, messages: List[Message]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        target_file = None
        start = end = None
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            if 'RANGE:' not in m.content and 'TARGET_FILE:' not in m.content:
                continue
            for line in m.content.splitlines():
                s = line.strip()
                if s.startswith('TARGET_FILE:') and target_file is None:
                    target_file = s.split(':', 1)[1].strip()
                elif s.startswith('RANGE:') and (start is None or end is None):
                    seg = s.split(':', 1)[1].strip()
                    if seg.startswith('[') and seg.endswith(']') and ':' in seg:
                        try:
                            a, b = seg[1:-1].split(':', 1)
                            start, end = int(a), int(b)
                        except Exception:
                            pass
                    elif '-' in seg:
                        try:
                            a, b = seg.split('-', 1)
                            start, end = int(a), int(b)
                        except Exception:
                            pass
            if target_file and start is not None and end is not None:
                break
        return target_file, start, end

    def _normalize_target_rel(self, target_file: str) -> str:
        target_rel = (target_file or '').strip().replace('\\', '/')
        if target_rel.startswith('task_'):
            parts = target_rel.split('/', 1)
            if len(parts) > 1:
                target_rel = parts[1]
        return target_rel

    def _parse_full_signature_from_messages(self, messages: List[Message]) -> Optional[str]:
        if not messages:
            return None
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            for line in m.content.splitlines():
                s = line.strip()
                if s.startswith('FUNCTION_SIGNATURE:'):
                    sig = s.split(':', 1)[1].strip()
                    if sig:
                        return sig
            content = m.content.strip()
            if content.startswith('{') and '"full_signature"' in content:
                try:
                    obj = json.loads(content)
                    if isinstance(obj, dict) and obj.get('full_signature'):
                        return str(obj.get('full_signature'))
                except Exception:
                    pass
        return None

    def _inject_current_function_snapshot(self, messages: List[Message], work_subfolder: Optional[str]) -> None:
        if not messages or not work_subfolder:
            return
        target_file, start, end = self._parse_target_file_and_range(messages)
        if not target_file or start is None or end is None:
            return

        target_rel = self._normalize_target_rel(target_file)
        dest_root = os.path.join(getattr(self.config, 'output_dir', 'output'), work_subfolder)
        target_path = os.path.join(dest_root, target_rel)
        try:
            if not os.path.exists(target_path):
                return
            with open(target_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            full_signature = self._parse_full_signature_from_messages(messages) or ''
            loc_start, loc_end = (None, None)
            if full_signature:
                loc_start, loc_end = _locate_function_in_lines(lines, full_signature, expected_start=start)
            if loc_start and loc_end and loc_start <= loc_end:
                start = loc_start
                end = loc_end
                start_idx = max(0, start - 1)
                end_idx = min(len(lines), end)
                snippet = ''.join(lines[start_idx:end_idx]).rstrip('\n')
            else:
                start_idx = max(0, start - 1)
                end_idx = min(len(lines), end)
                snippet = ''.join(lines[start_idx:end_idx]).rstrip('\n')
        except Exception:
            return

        msg = (
            '[CURRENT_FUNCTION_SNAPSHOT]\n'
            f'WORKDIR: {work_subfolder}\n'
            f'TARGET_FILE: {work_subfolder}/{target_rel}\n'
            f'RANGE: [{start}:{end}]\n\n'
            f'```solidity\n{snippet}\n```\n\n'
            'Please base your next fix on THIS current code snapshot (on disk), not the previous assistant draft.'
        )
        messages.append(Message(role='user', content=msg))

    def _prune_messages(self, messages: List[Message]):
        if not messages:
            return
        
        # 收集关键消息
        system_msg = None
        task_json_msg = None
        public_tests_msg = None
        refine_meta_msg = None
        latest_forge_feedback = None
        latest_iter_policy_msg = None
        latest_function_snapshot_msg = None
        latest_assistant = None
        
        # 倒序查找，确保找到最新的
        for m in reversed(messages):
            if m.role == 'system' and system_msg is None:
                system_msg = m
            elif m.role == 'user' and m.content:
                # Task JSON
                if task_json_msg is None and m.content.strip().startswith('{') and '"task_id"' in m.content:
                    try:
                        json.loads(m.content)
                        task_json_msg = m
                    except:
                        pass
                # FN_SECURE_REFINE meta (token-optimized TaskPrep message)
                # 注意：倒序遍历时第一次命中才是“最新”的 meta，因此需要只赋值一次。
                elif refine_meta_msg is None and 'FN_SECURE_REFINE STAGE' in m.content and 'RANGE:' in m.content and 'TARGET_FILE:' in m.content:
                    refine_meta_msg = m
                # PUBLIC_TESTS
                elif public_tests_msg is None and '[PUBLIC_TESTS]' in m.content:
                    public_tests_msg = m
                # FORGE_FEEDBACK
                elif latest_forge_feedback is None and '[FORGE_FEEDBACK]' in m.content:
                    latest_forge_feedback = m
                # ITERATIVE FIX POLICY (SecurityCallback injected)
                elif latest_iter_policy_msg is None and 'ITERATIVE FIX POLICY' in m.content:
                    latest_iter_policy_msg = m
                # CURRENT_FUNCTION_SNAPSHOT (helpful grounding)
                elif latest_function_snapshot_msg is None and '[CURRENT_FUNCTION_SNAPSHOT]' in m.content:
                    latest_function_snapshot_msg = m
            elif m.role == 'assistant' and latest_assistant is None:
                latest_assistant = m

        # 构建新的消息列表
        new_msgs = []
        if system_msg:
            new_msgs.append(system_msg)
        if task_json_msg:
            new_msgs.append(task_json_msg)
        if public_tests_msg:
            new_msgs.append(public_tests_msg)
        if refine_meta_msg:
            new_msgs.append(refine_meta_msg)
        if latest_iter_policy_msg:
            new_msgs.append(latest_iter_policy_msg)
        if latest_function_snapshot_msg:
            new_msgs.append(latest_function_snapshot_msg)
        if latest_assistant:
            new_msgs.append(latest_assistant)
        if latest_forge_feedback:
            new_msgs.append(latest_forge_feedback)

        # 替换原消息列表
        messages.clear()
        messages.extend(new_msgs)

    def _save_test_reports(self, build_ok: bool, build_out: str, test_ok: bool, test_out: str):
        """Save test reports to text and JSON files."""
        timestamp = datetime.now().isoformat()
        
        # Save text report (human-readable)
        try:
            with open(self.test_report_txt, 'w', encoding='utf-8') as f:
                f.write(f'Forge Test Report\n')
                f.write(f'Generated at: {timestamp}\n')
                f.write('=' * 60 + '\n\n')
                f.write(f'BUILD_STATUS: {"OK" if build_ok else "FAIL"}\n')
                f.write(f'TEST_STATUS: {"OK" if test_ok else "FAIL"}\n')
                f.write('\n' + '=' * 60 + '\n')
                f.write('FORGE BUILD OUTPUT\n')
                f.write('=' * 60 + '\n')
                f.write(build_out)
                f.write('\n' + '=' * 60 + '\n')
                f.write('FORGE TEST OUTPUT (with gas report)\n')
                f.write('=' * 60 + '\n')
                f.write(test_out)
            logger.info(f'Test report saved to {self.test_report_txt}')
        except Exception as e:
            logger.warning(f'Failed to save text test report: {e}')
        
        # Save JSON report (machine-readable)
        try:
            test_report_json = {
                'timestamp': timestamp,
                'build': {
                    'status': 'OK' if build_ok else 'FAIL',
                    'output': build_out
                },
                'test': {
                    'status': 'OK' if test_ok else 'FAIL',
                    'output': test_out
                }
            }
            with open(self.test_report_json, 'w', encoding='utf-8') as f:
                json.dump(test_report_json, f, indent=2, ensure_ascii=False)
            logger.info(f'Test report JSON saved to {self.test_report_json}')
        except Exception as e:
            logger.warning(f'Failed to save JSON test report: {e}')

    def _ensure_complete_file_with_function(self, messages: List[Message], work_subfolder: str):
        """确保目标 .sol 文件存在；若缺失则拷贝原始文件。方案1下不在这里做函数替换写回。"""
        task_json = None
        target_rel = None
        
        for m in messages:
            if m.role == 'user' and m.content and m.content.strip().startswith('{'):
                try:
                    task_obj = json.loads(m.content)
                    if 'task_id' in task_obj:
                        task_json = task_obj
                        break
                except Exception:
                    pass
        
        for m in messages:
            if m.role == 'user' and m.content and '[PUBLIC_TESTS]' in m.content:
                for line in m.content.splitlines():
                    if line.startswith('TARGET_FILE:'):
                        target_rel = line.split(':', 1)[1].strip()
                        break
        
        if not task_json or not target_rel:
            return False
        
        # Normalize: PUBLIC_TESTS may include task_ prefix
        if target_rel.startswith('task_'):
            parts = target_rel.split('/', 1)
            if len(parts) > 1:
                target_rel = parts[1]
        
        dest_root = os.path.join(getattr(self.config, 'output_dir', 'output'), work_subfolder)
        target_path = os.path.join(dest_root, target_rel)
        if os.path.exists(target_path):
            return True
        
        source_id = task_json.get('source_id', '')
        if not source_id:
            return False
        
        try:
            workspace_root = os.getcwd()
            local_dir = getattr(self.config, 'local_dir', '') or 'projects/sol_fn_miniloop'
            dataset_root = os.path.join(workspace_root, local_dir, 'root')
            
            norm = source_id.replace('\\', '/').lstrip('/')
            candidates = [
                os.path.join(dataset_root, norm),
                os.path.join(dataset_root, norm[len('root/'):]) if norm.startswith('root/') else None,
                os.path.join(workspace_root, norm),
                os.path.join(workspace_root, 'root', norm),
            ]
            
            original_source = None
            for c in candidates:
                if c and os.path.exists(c):
                    original_source = c
                    break
            
            if not original_source:
                return False
            
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(original_source, target_path)
            logger.info(f'ForgeCallback: copied missing source from {original_source} to {target_path}')
            return True
        except Exception as e:
            logger.warning(f'ForgeCallback: failed to ensure complete file exists: {e}')
            return False

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        # Skip if we're in a tool-calling turn or responding to a tool
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            return

        logger.info(f'ForgeCallback: before prune messages: {messages}')
        # Prune chat history to avoid overly long contexts
        self._prune_messages(messages)
        logger.info(f'ForgeCallback: after prune messages: {messages}')
        
        # Determine work folder and defer forge run until after tool writes (artifact callback)
        work_subfolder = self._parse_workdir_from_messages(messages) or self._pick_single_task_dir()
        logger.info(f'work_subfolder: {work_subfolder}')

        # Reset per-task counters when switching to a new task/workdir. Callback instances may be reused
        # across tasks in a batch run; without this, forge round counters can carry over and trigger
        # max-rounds stop immediately.
        if work_subfolder and work_subfolder != self._task_work_subfolder:
            self._last_failed_tests_fp = None
            self._same_failed_tests_rounds = 0
            self._forge_rounds = 0

        self._inject_current_function_snapshot(messages, work_subfolder)
        
        self._pending_work_subfolder = work_subfolder
        self._pending_forge_run = bool(work_subfolder)
        if work_subfolder:
            self._task_work_subfolder = work_subfolder
        logger.info(
            'ForgeCallback.on_generate_response: pending_forge_run=%s pending_work_subfolder=%s',
            self._pending_forge_run,
            self._pending_work_subfolder,
        )

    def _run_forge_and_append_feedback(self, messages: List[Message], work_subfolder: str):
        with self.chdir_context(folder=work_subfolder):
            if not self._forge_available():
                feedback = (
                    '[FORGE_FEEDBACK]\n'
                    'forge not found. Please install Foundry (https://book.getfoundry.sh/) '
                    'or run inside WSL/Docker with forge available.\n'
                )
                messages.append(Message(role='user', content=feedback))
                logger.info('ForgeCallback: forge not found; appended FORGE_FEEDBACK (len=%d)', len(feedback))
                return

            build_ok, build_out = self._run_cmd(['forge', 'build'])
            logger.info('ForgeCallback: forge build ok=%s output_len=%d', build_ok, len(build_out or ''))

            # Priority 1: use function-oriented logical split produced by TestSelectCallback
            selection_file = 'public_tests.json'
            ran_selected = False
            if os.path.exists(selection_file):
                try:
                    with open(selection_file, 'r', encoding='utf-8') as f:
                        sel = json.load(f)
                    test_file = sel.get('test_file') or ''
                    tests = sel.get('tests') or []
                    cmd = ['forge', 'test', '--gas-report']
                    if test_file:
                        cmd += ['--match-path', test_file]
                    if tests:
                        patt = '|'.join(re.escape(t) for t in tests)
                        if patt:
                            cmd += ['--match-test', patt]
                    ok, out = self._run_cmd(cmd)
                    test_ok, test_out = ok, out
                    ran_selected = True
                except Exception:
                    ran_selected = False

            # Priority 2: run public overlay tests under test/public/**
            if not ran_selected:
                public_dir = os.path.join('test', 'public')
                has_public_dir = False
                has_public_overlay = False
                try:
                    if os.path.isdir(public_dir):
                        for root, _, files in os.walk(public_dir):
                            if any(f.endswith('.sol') for f in files):
                                has_public_dir = True
                                break
                except Exception:
                    has_public_dir = False
                try:
                    test_root = 'test'
                    if os.path.isdir(test_root):
                        for root, _, files in os.walk(test_root):
                            if any(f.endswith('.public.t.sol') for f in files):
                                has_public_overlay = True
                                break
                except Exception:
                    has_public_overlay = False

                if has_public_dir:
                    test_ok, test_out = self._run_cmd(['forge', 'test', '--match-path', 'test/public/**', '--gas-report'])
                elif has_public_overlay:
                    test_ok, test_out = self._run_cmd(['forge', 'test', '--match-path', '**/*.public.t.sol', '--gas-report'])
                else:
                    test_file = 'test/Target.t.sol'
                    if os.path.exists(test_file):
                        test_ok, test_out = self._run_cmd(['forge', 'test', '-r', test_file, '--gas-report'])
                    else:
                        test_ok, test_out = self._run_cmd(['forge', 'test', '--gas-report'])

            # Record status for other callbacks (e.g., security) and stop control
            self.last_build_ok = build_ok
            self.last_test_ok = test_ok
            logger.info('ForgeCallback: forge test ok=%s output_len=%d', test_ok, len(test_out or ''))

            # Save test reports to files
            self._save_test_reports(build_ok, build_out, test_ok, test_out)

            # Extract key error information for better LLM understanding
            build_errors = ''
            if not build_ok:
                build_errors = self._filter_compile_errors(build_out) or self._extract_build_errors(build_out)
            parsed_test = self._parse_forge_test_stdout(test_out) if not test_ok else {}
            test_failures = self._extract_test_failures(test_out) if not test_ok else ''
            test_hints = self._build_test_hints(parsed_test, test_out) if not test_ok else []

            failed_tests_fp = None
            if not test_ok:
                failed_tests_fp = self._fingerprint_failed_tests(parsed_test)
                if failed_tests_fp and failed_tests_fp == self._last_failed_tests_fp:
                    self._same_failed_tests_rounds += 1
                else:
                    self._same_failed_tests_rounds = 0
                self._last_failed_tests_fp = failed_tests_fp
            else:
                self._last_failed_tests_fp = None
                self._same_failed_tests_rounds = 0
            
            # Compose final feedback with focused error information
            feedback_parts = [
                '[FORGE_FEEDBACK]',
                f'BUILD_STATUS: {"OK" if build_ok else "FAIL"}',
                f'TEST_STATUS: {"OK" if test_ok else "FAIL"}'
            ]
            
            if build_errors:
                feedback_parts.extend([
                    '',
                    'COMPILATION ERRORS:',
                    build_errors
                ])
            
            if test_failures:
                feedback_parts.extend([
                    '',
                    'TEST FAILURES:',
                    test_failures
                ])

            if parsed_test.get('fails'):
                feedback_parts.extend([
                    '',
                    'FAILED_TESTS:',
                ])
                for k, v in list(parsed_test['fails'].items())[:10]:
                    feedback_parts.append(f'- {k}: {v}')

            if test_hints:
                feedback_parts.extend(['', 'HINTS:'])
                feedback_parts.extend(test_hints)
            
            # Add detailed output only if needed (for complex issues)
            if (not build_ok and not build_errors) or (not test_ok and not test_failures):
                build_out_snip = self._truncate(build_out, 4000)
                test_out_snip = self._truncate(test_out, 4000)
                feedback_parts.extend([
                    '',
                    '--- forge build (full output) ---' if not build_ok else '',
                    build_out_snip if not build_ok else '',
                    '--- forge test --gas-report (full output) ---' if not test_ok else '',
                    test_out_snip if not test_ok else ''
                ])

            feedback_parts.extend([
                '',
                f'Full reports saved to: {self.test_report_txt}, {self.test_report_json}',
                'NEXT_ACTIONS:',
                f'- Read the failing test details from {self.test_report_txt} and focus on fixing them first.',
                '- Make minimal changes within the allowed range only.',
                '- Re-output ONLY the target function code block using the exact WORKDIR/TARGET_FILE path (you may include // [start:x] [end:y] markers).',
            ])

            if self._same_failed_tests_rounds >= 1 and failed_tests_fp:
                feedback_parts.extend([
                    '',
                    '[STALL_DETECTED]',
                    'FAILED_TESTS did not change across consecutive rounds.',
                    'You MUST propose a DIFFERENT root-cause hypothesis and apply a DIFFERENT code change than last round.',
                    'If you cannot find a different fix, explain precisely why and stop.',
                    '',
                    'REQUIRED_OUTPUT:',
                    '- List at least 2 concrete code edits you will do that differ from the previous round.',
                    '- Then output ONLY the updated target function code block.',
                ])

            if not build_ok:
                feedback_parts.append('Please fix compilation errors.')
            if not test_ok:
                feedback_parts.append('Please fix the remaining failing tests.')
            
            feedback = '\n'.join(filter(None, feedback_parts))
            messages.append(Message(role='user', content=feedback))
            logger.info(
                'ForgeCallback: appended FORGE_FEEDBACK (build_ok=%s test_ok=%s len=%d)',
                build_ok,
                test_ok,
                len(feedback),
            )

    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        """
        Keep the loop alive if build or test failed, so the agent can fix issues.
        Let the default loop stopping rule apply only when both build and test passed.
        """
        logger.info(
            'ForgeCallback.after_tool_call: enter pending_forge_run=%s pending_work_subfolder=%s last_build_ok=%s last_test_ok=%s should_stop(before)=%s',
            self._pending_forge_run,
            self._pending_work_subfolder,
            self.last_build_ok,
            self.last_test_ok,
            getattr(runtime, 'should_stop', None),
        )
        if not self._pending_forge_run:
            logger.info('ForgeCallback.after_tool_call: pending_forge_run is False; skip running forge')
            return
        if not self._pending_work_subfolder:
            self._pending_forge_run = False
            logger.info('ForgeCallback.after_tool_call: missing work_subfolder; pending_forge_run reset to False')
            return

        # Ensure file existence conservatively; do not overwrite artifact integration.
        self._ensure_complete_file_with_function(messages, self._pending_work_subfolder)

        self._pending_forge_run = False
        self._run_forge_and_append_feedback(messages, self._pending_work_subfolder)

        self._forge_rounds += 1

        if getattr(self, '_same_failed_tests_rounds', 0) >= 1 and getattr(self, '_last_failed_tests_fp', None):
            try:
                setattr(runtime, 'force_stop', True)
            except Exception:
                pass
            runtime.should_stop = True
            logger.info('ForgeCallback.after_tool_call: stall detected (same fingerprint twice) -> force_stop=True should_stop=True')
            return

        if getattr(self, '_forge_rounds', 0) >= getattr(self, '_max_forge_rounds', 4):
            try:
                setattr(runtime, 'force_stop', True)
            except Exception:
                pass
            runtime.should_stop = True
            logger.info('ForgeCallback.after_tool_call: max rounds reached -> force_stop=True should_stop=True')
            return

        try:
            setattr(runtime, 'forge_last_build_ok', self.last_build_ok)
            setattr(runtime, 'forge_last_test_ok', self.last_test_ok)
        except Exception:
            pass

        # Decide whether to continue iterating based on the forge result we just obtained.
        if self.last_build_ok is False or self.last_test_ok is False:
            runtime.should_stop = False
            logger.info('ForgeCallback.after_tool_call: build/test failed (post-forge) -> should_stop set to False')
        elif self.last_build_ok is True and self.last_test_ok is True:
            runtime.should_stop = False
            logger.info('ForgeCallback.after_tool_call: build/test passed (post-forge) -> should_stop set to False (waiting for security gate)')
        logger.info(
            'ForgeCallback.after_tool_call: forge run complete last_build_ok=%s last_test_ok=%s should_stop(after)=%s',
            self.last_build_ok,
            self.last_test_ok,
            getattr(runtime, 'should_stop', None),
        )
