# projects/sol_fn_miniloop/callbacks/security_callback.py
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from typing import List, Optional, Tuple
from datetime import datetime

import re

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
    s = re.sub(r"//.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_signature_head_text(full_signature: str) -> str:
    if not full_signature:
        return ''
    m = re.search(r"(function\s+[A-Za-z0-9_]+\s*\(.*?\))", full_signature, flags=re.DOTALL)
    return (m.group(1) if m else '').strip()


def _canonicalize_function_signature(full_signature: str) -> Tuple[Optional[str], Optional[str]]:
    if not full_signature:
        return None, None
    mm = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", full_signature)
    func_name = mm.group(1) if mm else None
    if not func_name:
        return None, None
    m2 = re.search(r"function\s+%s\s*\((.*?)\)" % re.escape(func_name), full_signature, flags=re.DOTALL)
    raw_params = (m2.group(1) if m2 else '').strip()
    if not raw_params:
        return func_name, f"function {func_name}()"

    param_types: List[str] = []
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


def _canonicalize_source_signature_head(lines: List[str], start_idx_0: int) -> str:
    buf = ''
    i = start_idx_0
    while i < len(lines) and ')' not in buf:
        ln = re.sub(r"//.*$", "", lines[i])
        buf += ln.strip() + ' '
        i += 1
    buf = re.sub(r"\s+", " ", buf).strip()
    m = re.search(r"(function\s+[A-Za-z0-9_]+\s*\(.*?\))", buf)
    if not m:
        return ''
    head = m.group(1)
    mm = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", head)
    func_name = mm.group(1) if mm else None
    if not func_name:
        return ''
    m2 = re.search(r"function\s+%s\s*\((.*?)\)" % re.escape(func_name), head, flags=re.DOTALL)
    raw_params = (m2.group(1) if m2 else '').strip()
    if not raw_params:
        return f"function {func_name}()"
    param_types: List[str] = []
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
    return f"function {func_name}({','.join(param_types)})"


def _locate_function_in_lines(lines: List[str], full_signature: str, expected_start: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    func_name, canonical_head = _canonicalize_function_signature(full_signature)
    if not func_name:
        return None, None

    full_sig_norm = _normalize_sig_text(full_signature).replace(' ', '')
    head_norm = _normalize_sig_text(_extract_signature_head_text(full_signature)).replace(' ', '')

    full_sig_candidates: List[int] = []
    head_candidates: List[int] = []
    canonical_candidates: List[int] = []

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if 'function' in line_stripped:
            hdr = ''
            j = i - 1
            while j < len(lines) and '{' not in hdr and ';' not in hdr:
                ln = re.sub(r"//.*$", "", lines[j])
                hdr += ln.strip() + ' '
                if '{' in ln or ';' in ln:
                    break
                j += 1
            hdr_norm = _normalize_sig_text(hdr).replace(' ', '')
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


class SecurityCallback(Callback):
    """
    Run Slither after tests pass, then append findings summary as a user message.
    - Gated: only runs if `forge test` currently passes (to avoid multi-feedback per round).
    - Writes slither.json (raw) and layered reports (txt/json) in the project root.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.slither_json = 'slither_after.json'
        self.slither_baseline_json = 'slither_baseline.json'
        self.layered_report_txt = 'slither_layered_report.txt'
        self.layered_report_json = 'slither_layered_report.json'
        self._iter_state = {
            'last_edit_family': None,
            'last_hypothesis': None,
        }

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    def _truncate(self, s: str, max_chars: int = 8000) -> str:
        if s is None:
            return ''
        s = s.strip()
        if len(s) > max_chars:
            return s[:max_chars] + f'\n...[truncated {len(s) - max_chars} chars]'
        return s

    def _load_slither_report(self, json_path: str) -> Optional[dict]:
        try:
            if not os.path.exists(json_path):
                return None
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _collect_high_medium_findings(self, report: dict) -> List[dict]:
        out: List[dict] = []
        detectors = (((report or {}).get('results') or {}).get('detectors') or [])
        for d in detectors:
            impact = (d.get('impact') or '').strip()
            if impact not in ('High', 'Medium'):
                continue
            check = (d.get('check') or '').strip()
            desc = (d.get('description') or '').strip()
            for el in (d.get('elements') or []):
                sm = (el.get('source_mapping') or {})
                filename_relative = (sm.get('filename_relative') or '').replace('\\', '/')
                lines = sm.get('lines') or []
                out.append({
                    'impact': impact,
                    'check': check,
                    'filename_relative': filename_relative,
                    'lines': lines,
                    'description': desc,
                })
        return out

    def _finding_key(self, finding: dict) -> str:
        impact = finding.get('impact') or ''
        check = finding.get('check') or ''
        fn = finding.get('filename_relative') or ''
        lines = finding.get('lines') or []
        return f"{impact}|{check}|{fn}|{','.join([str(x) for x in lines])}"

    def _format_new_findings_mini(self, new_findings: List[dict], ok: bool, out: str) -> str:
        status_line = f'SLITHER_STATUS: {"OK" if ok else "ERROR"}'
        if not new_findings:
            return (
                '[SLITHER_FEEDBACK_MINI]\n'
                'No new High/Medium findings compared to baseline.\n'
                f'{status_line}\n'
            )

        lines: List[str] = [
            '[SLITHER_FEEDBACK_MINI]',
            f'New High/Medium findings compared to baseline: {len(new_findings)}',
        ]
        for fnd in new_findings[:30]:
            loc = f"{fnd.get('filename_relative', '')}:{fnd.get('lines', [])}"
            lines.append(f"- {fnd.get('impact')} {fnd.get('check')}: {loc}")
        if len(new_findings) > 30:
            lines.append(f"(truncated; total {len(new_findings)})")

        lines.extend([
            status_line,
            f'Raw JSON: {self.slither_json}',
        ])
        if (not ok) and out:
            lines.append('Error (truncated):')
            lines.append(self._truncate(out, 1500))
        lines.append('Please fix ONLY the newly introduced High/Medium findings with minimal code changes.')
        return '\n'.join(lines) + '\n'

    def _compress_forge_feedback(self, forge_feedback: Message) -> Message:
        """Compress FORGE_FEEDBACK into a short, hint-forward message.

        The goal is to ensure the LLM sees the failing test signal + actionable HINTS without
        being diluted by long boilerplate.
        """
        if not forge_feedback or not forge_feedback.content:
            return forge_feedback

        content = forge_feedback.content
        lines = content.splitlines()

        # If compilation failed, prefer a tiny, compilation-focused mini message.
        if 'COMPILATION ERRORS:' in content or 'Compiler run failed:' in content:
            err_snip = None
            for ln in lines:
                s = ln.strip()
                if s.startswith('Error (') or s.startswith('Error:'):
                    err_snip = s
                    break
            if not err_snip:
                for ln in lines:
                    s = ln.strip()
                    if 'Explicit type conversion not allowed' in s or 'ParserError' in s or 'TypeError' in s:
                        err_snip = s
                        break
            mini = ['[FORGE_FEEDBACK_MINI]']
            if err_snip:
                mini.extend(['COMPILATION_ERROR_SNIP:', f'- {err_snip}'])
            return Message(role='user', content='\n'.join(mini) + '\n')

        failed_tests: List[str] = []
        in_failed_tests = False
        for ln in lines:
            s = ln.strip()
            if s == 'FAILED_TESTS:':
                in_failed_tests = True
                continue
            if in_failed_tests:
                if (not s) or s in ('HINTS:', 'NEXT_ACTIONS:') or s.startswith('Full reports saved to:') or s.startswith('[STALL_DETECTED]'):
                    break
                if s.startswith('-'):
                    item = s.lstrip('-').strip()
                    if ':' in item:
                        item = item.split(':', 1)[0].strip()
                    if item:
                        failed_tests.append(item)

        if not failed_tests:
            for ln in lines:
                s = ln.strip()
                if not s.startswith('[FAIL:'):
                    continue
                m = re.search(r"\]\s*(test[^\s]*\(.*?\))", s)
                if m:
                    failed_tests.append(m.group(1).strip())

        seen = set()
        uniq_failed_tests: List[str] = []
        for t in failed_tests:
            if t not in seen:
                seen.add(t)
                uniq_failed_tests.append(t)

        fail_reason_snip = None
        candidates: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s.startswith('[FAIL:'):
                continue
            s2 = re.sub(r";\s*counterexample:.*$", "", s, flags=re.IGNORECASE).strip()
            s2 = re.sub(r"\s*counterexample:.*$", "", s2, flags=re.IGNORECASE).strip()
            if s2:
                candidates.append(s2)
        if candidates:
            fail_reason_snip = min(candidates, key=len)

        suggestion_snip = None
        root_cause_snip = None
        minimal_fix_ideas: List[str] = []
        if fail_reason_snip:
            fr = fail_reason_snip.lower()
            if 'assertion failed' in fr:
                suggestion_snip = '- assertion failed: check encoding/layout, boundary conditions, and return value semantics vs test expectation'
                root_cause_snip = '- MOST_LIKELY_ROOT_CAUSE: encoding/layout/ABI-alignment mismatch'
                minimal_fix_ideas = [
                    '- Ensure narrow types are right-aligned in 32-byte words (abi.encode semantics); avoid left-shifts unless explicitly required',
                    '- Mask dirty upper bits before hashing/packing (e.g., address=>160-bit, bool=>1-bit, bytes4=>32-bit)',
                ]
            elif 'panic: 0x11' in fr:
                suggestion_snip = '- panic 0x11: likely arithmetic overflow/underflow; check unchecked blocks and overflow semantics'
                root_cause_snip = '- MOST_LIKELY_ROOT_CAUSE: arithmetic overflow/underflow (panic 0x11)'
                minimal_fix_ideas = [
                    '- Remove/limit unchecked blocks; add precondition checks for bounds before arithmetic',
                    '- Re-check type widths/casts that may truncate or overflow intermediate values',
                ]
            elif ('revert:' in fr) or ('evmerror' in fr):
                suggestion_snip = '- revert/EvmError: check require/revert conditions and error semantics match the test expectation'
                root_cause_snip = '- MOST_LIKELY_ROOT_CAUSE: require/revert condition or error semantics mismatch'
                minimal_fix_ideas = [
                    '- Align require/revert conditions with test preconditions; verify branches match expected revert/non-revert cases',
                    '- If reverting, ensure revert reason/custom error selector matches what tests expect (or remove overly-strict checks)',
                ]

        # Extract counterexample args (keep args only; do NOT keep calldata).
        counterexample_args = None
        for ln in lines:
            s = ln.strip()
            if 'args=[' not in s:
                continue
            m = re.search(r"args=\[(.*?)\]", s)
            if m:
                counterexample_args = f"args=[{m.group(1)}]"
                if len(counterexample_args) > 260:
                    counterexample_args = counterexample_args[:260] + '...'
                break

        mini_parts: List[str] = ['[FORGE_FEEDBACK_MINI]']
        if uniq_failed_tests:
            mini_parts.append('FAILED_TESTS:')
            for t in uniq_failed_tests[:3]:
                mini_parts.append(f'- {t}')
        if fail_reason_snip:
            mini_parts.extend(['ASSERT_SNIP:', f'- {fail_reason_snip}'])
        if root_cause_snip:
            mini_parts.extend(['ROOT_CAUSE:', root_cause_snip.lstrip('- ').strip()])
        if (fail_reason_snip or '').lower().find('assertion failed') != -1:
            mini_parts.extend(['HINT:', 'narrow types are right-aligned within 32 bytes'])
        if counterexample_args:
            mini_parts.extend(['COUNTEREXAMPLE_ARGS:', f'- {counterexample_args}'])

        return Message(role='user', content='\n'.join(mini_parts) + '\n')

    def _prune_messages(self, messages: List[Message], build_ok: bool = None, test_ok: bool = None):
        if not messages:
            return
        
        # 收集关键消息
        system_msg = None
        task_json_msg = None
        public_tests_msg = None
        refine_meta_msg = None
        latest_forge_feedback = None
        latest_slither_feedback = None
        latest_snapshot = None
        latest_iter_policy_msg = None
        latest_assistant = None
        last_strict_system = None
        
        # 倒序查找，确保找到最新的
        for m in reversed(messages):
            if m.role == 'system' and m.content:
                if m.content.strip().startswith('STRICT_OUTPUT_MODE'):
                    if last_strict_system is None:
                        last_strict_system = m
                    continue
                if system_msg is None:
                    system_msg = m
            elif m.role == 'user' and m.content:
                # Task JSON
                if task_json_msg is None and m.content.strip().startswith('{') and '"task_id"' in m.content:
                    try:
                        json.loads(m.content)
                        task_json_msg = m
                    except:
                        pass
                # FN_SECURE_REFINE meta
                elif refine_meta_msg is None and 'FN_SECURE_REFINE STAGE' in m.content and 'RANGE:' in m.content and 'TARGET_FILE:' in m.content:
                    refine_meta_msg = m
                # PUBLIC_TESTS
                elif public_tests_msg is None and '[PUBLIC_TESTS]' in m.content:
                    public_tests_msg = m
                # FORGE_FEEDBACK - 关键逻辑
                elif latest_forge_feedback is None and '[FORGE_FEEDBACK]' in m.content:
                    # Forge失败时：保留FORGE_FEEDBACK（用于引导LLM修复）
                    # Forge成功时：不保留FORGE_FEEDBACK（已经处理过）
                    if build_ok is False or test_ok is False:
                        latest_forge_feedback = m
                # CURRENT_FUNCTION_SNAPSHOT
                elif latest_snapshot is None and '[CURRENT_FUNCTION_SNAPSHOT]' in m.content:
                    latest_snapshot = m
                # ITERATIVE FIX POLICY (MUST FOLLOW)
                elif latest_iter_policy_msg is None and 'ITERATIVE FIX POLICY' in m.content:
                    latest_iter_policy_msg = m
                # SLITHER_FEEDBACK
                elif latest_slither_feedback is None and '[SLITHER_FEEDBACK]' in m.content:
                    latest_slither_feedback = m
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

        # Preserve the iterative policy prompt (injected by SecurityCallback in prior rounds)
        # so it survives pruning and remains visible in after-prune context.
        if latest_iter_policy_msg:
            new_msgs.append(latest_iter_policy_msg)

        # Avoid keeping previous assistant outputs when forge failed; they can anchor the model in a wrong direction.
        if (build_ok is True and test_ok is True):
            if latest_assistant and latest_assistant.content and len(latest_assistant.content) <= 2000:
                new_msgs.append(latest_assistant)
        # 消息优先级：SLITHER_FEEDBACK > FORGE_FEEDBACK
        if latest_slither_feedback:
            new_msgs.append(latest_slither_feedback)
        elif latest_forge_feedback:
            new_msgs.append(self._compress_forge_feedback(latest_forge_feedback))

        if latest_snapshot:
            new_msgs.append(latest_snapshot)

        if last_strict_system is not None:
            new_msgs.append(last_strict_system)
        else:
            new_msgs.append(
                Message(
                    role='system',
                    content=(
                        'STRICT_OUTPUT_MODE\n'
                        '- Output EXACTLY ONE function code block and NOTHING ELSE.\n'
                        '- The function code block MUST be in the exact form:```solidity: <WORKDIR>/<TARGET_FILE> ...```\n'
                        '- Do NOT output explanations, headings, lists, or a second code block.\n'
                        '- If you need to change code, apply it directly inside that single file output.\n'
                    ),
                )
            )

        # 替换原消息列表
        messages.clear()
        messages.extend(new_msgs)

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

    def _run_cmd(self, cmd: List[str], timeout: Optional[int] = None, check: bool = False) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=check
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

    def _slither_available(self) -> bool:
        return shutil.which('slither') is not None

    def _tests_pass(self, runtime: Runtime) -> bool:
        ok, _ = self._run_cmd(['forge', 'test', '-q'])
        build_ok, test_ok = self._get_forge_status_from_runtime(runtime)
        return ok and test_ok

    def _get_forge_status_from_runtime(self, runtime: Runtime) -> Tuple[Optional[bool], Optional[bool]]:
        """Try to read forge build/test status from sibling callbacks (ForgeCallback)."""
        build_ok = None
        test_ok = None
        try:
            for cb in getattr(runtime, 'callbacks', []):
                if hasattr(cb, 'last_build_ok') and hasattr(cb, 'last_test_ok'):
                    build_ok, test_ok = cb.last_build_ok, cb.last_test_ok
                    break
        except Exception:
            pass

        try:
            cb_attr = getattr(runtime, 'callbacks', None)
            logger.info(
                'SecurityCallback.debug: has_callbacks_attr=%s callbacks_type=%s callbacks_len=%s',
                hasattr(runtime, 'callbacks'),
                type(cb_attr).__name__ if cb_attr is not None else None,
                len(cb_attr) if isinstance(cb_attr, list) else None,
            )
            if isinstance(cb_attr, list):
                logger.info(
                    'SecurityCallback.debug: callbacks_classes=%s',
                    [type(c).__name__ for c in cb_attr],
                )
            keys = []
            for k in dir(runtime):
                lk = str(k).lower()
                if any(t in lk for t in ('callback', 'stop', 'round', 'chat', 'state', 'config')):
                    keys.append(k)
            logger.info('SecurityCallback.debug: runtime_keys_filtered=%s', keys)
            d = getattr(runtime, '__dict__', None)
            if isinstance(d, dict):
                logger.info('SecurityCallback.debug: runtime___dict___keys=%s', list(d.keys()))
        except Exception as e:
            logger.warning('SecurityCallback.debug: runtime introspection failed: %s', e)
        return build_ok, test_ok

    def _parse_workdir_from_messages(self, messages: List[Message]) -> Optional[str]:
        for m in reversed(messages):
            if m.role == 'user' and m.content:
                for line in m.content.splitlines():
                    if line.strip().startswith('WORKDIR:'):
                        _, _, rest = line.partition(':')
                        wd = rest.strip()
                        if wd:
                            return wd
                try:
                    obj = json.loads(m.content)
                    if isinstance(obj, dict) and 'task_id' in obj:
                        return f"task_{obj['task_id']}"
                except Exception:
                    pass
        return None

    def _pick_single_task_dir(self) -> Optional[str]:
        base = getattr(self.config, 'output_dir', 'output')
        try:
            entries = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d.startswith('task_')]
            if len(entries) == 1:
                return entries[0]
        except Exception:
            return None
        return None

    def _parse_task_range(self, messages: List[Message]) -> Optional[tuple]:
        """Extract target file and line range from TaskPrep message."""
        # Scan in reverse to ensure we pick the LATEST range meta (range can be updated after integration).
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            if ('[TASK_PREP]' not in m.content
                and 'FN_SECURE_REFINE STAGE' not in m.content
                and 'UPDATED RANGE META' not in m.content):
                continue
                
            target_file = None
            start = end = None
            
            for line in m.content.splitlines():
                if line.startswith('TARGET_FILE:'):
                    target_file = line.split(':', 1)[1].strip()
                elif line.startswith('📍 TARGET FUNCTION:'):
                    # Extract file from "📍 TARGET FUNCTION: filename.sol [lines start:end]"
                    import re
                    match = re.search(r'([^s]+)\.sol\s*\[lines\s*(\d+):(\d+)\]', line)
                    if match:
                        target_file = match.group(1) + '.sol'
                        start = int(match.group(2))
                        end = int(match.group(3))
                elif line.startswith('RANGE:'):
                    seg = line.split(':', 1)[1].strip()
                    # 支持新格式 [start:end] 和旧格式 start-end
                    if seg.startswith('[') and seg.endswith(']'):
                        # 新格式: [617:658]
                        try:
                            s, e = seg[1:-1].split(':', 1)
                            start, end = int(s), int(e)
                        except Exception:
                            pass
                    else:
                        # 旧格式: 617-658
                        try:
                            s, e = seg.split('-', 1)
                            start, end = int(s), int(e)
                        except Exception:
                            pass
            
            if target_file and start and end:
                return (target_file, start, end)
        return None

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

    def _normalize_target_rel(self, target_file: str) -> str:
        target_rel = (target_file or '').strip().replace('\\', '/')
        if target_rel.startswith('task_'):
            parts = target_rel.split('/', 1)
            if len(parts) > 1:
                target_rel = parts[1]
        return target_rel

    def _get_effective_task_range(self, messages: List[Message]) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[Tuple[str, int, int]], bool]:
        """Return (target_file, start, end, meta_range, used_real).

        - meta_range is whatever _parse_task_range found (may be stale).
        - If possible, compute REAL range from disk via full_signature + brace counting and use it.
        """
        meta_range = self._parse_task_range(messages)
        if not meta_range:
            return None, None, None, None, False
        meta_target_file, meta_start, meta_end = meta_range

        workdir = self._parse_workdir_from_messages(messages) or self._pick_single_task_dir()
        full_signature = self._parse_full_signature_from_messages(messages) or ''
        if not workdir or not full_signature:
            logger.info(
                'SecurityCallback._get_effective_task_range: fallback to META_RANGE (missing workdir or full_signature) '
                'meta_target_file=%r meta_range=[%s:%s] workdir=%r sig_present=%s',
                meta_target_file,
                meta_start,
                meta_end,
                workdir,
                bool(full_signature),
            )
            return meta_target_file, meta_start, meta_end, meta_range, False

        target_rel = self._normalize_target_rel(meta_target_file)
        target_path = os.path.join(getattr(self.config, 'output_dir', 'output'), workdir, target_rel)
        alt_target_path = os.path.abspath(target_rel)
        try:
            if not os.path.exists(target_path):
                if os.path.exists(alt_target_path):
                    logger.info(
                        'SecurityCallback._get_effective_task_range: primary target_path not found; using alt_target_path (cwd-relative) '
                        'meta_target_file=%r workdir=%r alt_target_path=%r',
                        meta_target_file,
                        workdir,
                        alt_target_path,
                    )
                    target_path = alt_target_path
                else:
                    logger.info(
                        'SecurityCallback._get_effective_task_range: fallback to META_RANGE (file not found) '
                        'meta_target_file=%r meta_range=[%s:%s] workdir=%r target_path=%r alt_target_path=%r sig=%r',
                        meta_target_file,
                        meta_start,
                        meta_end,
                        workdir,
                        target_path,
                        alt_target_path,
                        (full_signature[:160] + '...') if len(full_signature) > 160 else full_signature,
                    )
                    return meta_target_file, meta_start, meta_end, meta_range, False
            with open(target_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            real_start, real_end = _locate_function_in_lines(lines, full_signature, expected_start=meta_start)
            if real_start and real_end and real_start <= real_end:
                logger.info(
                    'SecurityCallback._get_effective_task_range: using REAL_RANGE '
                    'meta_target_file=%r meta_range=[%s:%s] real_range=[%s:%s] workdir=%r target_path=%r sig=%r',
                    meta_target_file,
                    meta_start,
                    meta_end,
                    real_start,
                    real_end,
                    workdir,
                    target_path,
                    (full_signature[:160] + '...') if len(full_signature) > 160 else full_signature,
                )
                return meta_target_file, real_start, real_end, meta_range, True
        except Exception:
            logger.exception(
                'SecurityCallback._get_effective_task_range: exception while computing REAL_RANGE -> fallback to META_RANGE '
                'meta_target_file=%r meta_range=[%s:%s] workdir=%r target_path=%r sig=%r',
                meta_target_file,
                meta_start,
                meta_end,
                workdir,
                target_path,
                (full_signature[:160] + '...') if len(full_signature) > 160 else full_signature,
            )
            return meta_target_file, meta_start, meta_end, meta_range, False

        logger.info(
            'SecurityCallback._get_effective_task_range: fallback to META_RANGE (no REAL_RANGE match) '
            'meta_target_file=%r meta_range=[%s:%s] workdir=%r target_path=%r sig=%r',
            meta_target_file,
            meta_start,
            meta_end,
            workdir,
            target_path,
            (full_signature[:160] + '...') if len(full_signature) > 160 else full_signature,
        )
        return meta_target_file, meta_start, meta_end, meta_range, False

    def _parse_slither_report_minimal(self, path: str, messages: List[Message]) -> Tuple[str, Optional[int]]:
        """Parse slither JSON and return a minimal in-range High/Medium-only summary to save tokens."""
        if not os.path.exists(path):
            return 'Slither JSON report not found.', None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f'Failed to parse Slither JSON: {e}', None

        results = data.get('results', {})
        detectors = results.get('detectors', []) or results.get('infos', [])
        target_file, start, end, meta_range, used_real = self._get_effective_task_range(messages)
        task_range = (target_file, start, end) if (target_file and start and end) else None

        in_range_high_med = 0
        top_items = []

        if task_range:
            target_file, start, end = task_range

        for d in detectors:
            impact = d.get('impact') or d.get('severity') or 'Low'
            if impact not in ('High', 'Medium'):
                continue
            elements = d.get('elements', [])
            first_elem = elements[0] if elements else {}

            if task_range:
                if not self._is_in_target_range(first_elem, target_file, start, end):
                    continue

            in_range_high_med += 1
            check = d.get('check') or d.get('check_type') or 'unknown'
            description = d.get('description', '') or d.get('markdown', '') or 'N/A'
            # Try to show a representative line number
            src_map = first_elem.get('source_mapping', {}) if isinstance(first_elem, dict) else {}
            lines = src_map.get('lines', []) if isinstance(src_map, dict) else []
            line_hint = ''
            try:
                if isinstance(lines, list) and lines:
                    line_hint = f":{int(lines[0])}"
            except Exception:
                line_hint = ''
            top_items.append(f"- [{impact}] {check}{line_hint} | {str(description).strip()[:200]}")

        # Keep message short
        top_items = top_items[:5]
        header = '[SLITHER_FEEDBACK]'
        if task_range:
            header += f"\nTARGET: {target_file} [lines {start}:{end}]"
            if meta_range and used_real:
                _, ms, me = meta_range
                header += f"\nMETA_RANGE: [{ms}:{me}]"
                header += f"\nREAL_RANGE: [{start}:{end}]"
        header += f"\nIN_RANGE_HIGH_MED: {in_range_high_med}"

        if in_range_high_med == 0:
            return header + "\n✅ PASS - No High/Medium issues in target range.", 0

        body = "\n".join(top_items) if top_items else '(no details)'
        return header + "\n❌ FAIL - High/Medium issues found in target range:\n" + body, in_range_high_med

    def _is_in_target_range(self, element: dict, target_file: str, start: int, end: int) -> bool:
        """Check if a Slither detection element is within the target function range."""
        source_mapping = element.get('source_mapping', {})
        filename = source_mapping.get('filename_short', '') or source_mapping.get('filename_absolute', '')
        
        # Normalize paths for comparison
        if not filename:
            return False
        filename_norm = filename.replace('\\', '/')
        target_norm = target_file.replace('\\', '/')
        
        if not filename_norm.endswith(target_norm):
            return False
        
        # Check line range
        lines = source_mapping.get('lines', [])
        if not lines:
            return False
        
        # If any line of the detection is within [start:end], consider it in-range
        for line_num in lines:
            if start <= line_num <= end:
                return True
        return False

    def _parse_slither_report(self, path: str, messages: List[Message] = None) -> str:
        if not os.path.exists(path):
            return 'Slither JSON report not found.'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f'Failed to parse Slither JSON: {e}'

        results = data.get('results', {})
        detectors = results.get('detectors', []) or results.get('infos', [])
        
        # Parse target range if available
        if messages:
            tf, s, e, _, _ = self._get_effective_task_range(messages)
            task_range = (tf, s, e) if (tf and s and e) else None
        else:
            task_range = None
        
        # Categorize findings
        in_range_findings = []
        out_range_findings = []
        
        counts_in = {'High': 0, 'Medium': 0, 'Low': 0, 'Optimization': 0, 'Informational': 0}
        counts_out = {'High': 0, 'Medium': 0, 'Low': 0, 'Optimization': 0, 'Informational': 0}
        counts_total = {'High': 0, 'Medium': 0, 'Low': 0, 'Optimization': 0, 'Informational': 0}
        
        for d in detectors:
            impact = d.get('impact') or d.get('severity') or 'Low'
            check = d.get('check') or d.get('check_type') or 'unknown'
            description = d.get('description', '') or d.get('markdown', '') or 'N/A'
            
            # Get first element for location
            elements = d.get('elements', [])
            first_elem = elements[0] if elements else {}
            
            counts_total[impact] = counts_total.get(impact, 0) + 1
            
            finding_info = {
                'impact': impact,
                'check': check,
                'description': description[:200],  # Truncate long descriptions
                'element': first_elem
            }
            
            # Categorize by range
            if task_range:
                target_file, start, end = task_range
                if self._is_in_target_range(first_elem, target_file, start, end):
                    in_range_findings.append(finding_info)
                    counts_in[impact] = counts_in.get(impact, 0) + 1
                else:
                    out_range_findings.append(finding_info)
                    counts_out[impact] = counts_out.get(impact, 0) + 1
            else:
                # No range info, treat all as in-range
                in_range_findings.append(finding_info)
                counts_in[impact] = counts_in.get(impact, 0) + 1
        
        # Build report
        report_lines = []
        
        if task_range:
            target_file, start, end = task_range
            report_lines.append(f'🎯 TARGET FUNCTION: {target_file} [lines {start}:{end}]')
            report_lines.append('🔍 MODE: FN_SECURE_REFINE - Only testing newly generated function')
            report_lines.append('')
            
            # In-range findings (critical for evaluation)
            report_lines.append('=' * 60)
            report_lines.append('📋 SECURITY FINDINGS IN TARGET FUNCTION (lines {}:{})'.format(start, end))
            report_lines.append('=' * 60)
            report_lines.append('Summary: ' + ', '.join(f'{k}: {v}' for k, v in counts_in.items() if v > 0) or 'None')
            
            if in_range_findings:
                report_lines.append('')
                report_lines.append('🚨 SECURITY ISSUES REQUIRING FIXES:')
                for i, f in enumerate(in_range_findings[:10], 1):  # Show first 10
                    report_lines.append(f"{i}. [{f['impact']}] {f['check']}")
                    report_lines.append(f"   {f['description']}")
                if len(in_range_findings) > 10:
                    report_lines.append(f'   ... and {len(in_range_findings) - 10} more')
            else:
                report_lines.append('✅ No security issues detected in target function!')
            
            report_lines.append('')
            report_lines.append('=' * 60)
            report_lines.append('📎 FINDINGS OUTSIDE TARGET RANGE (context only)')
            report_lines.append('=' * 60)
            report_lines.append('Summary: ' + ', '.join(f'{k}: {v}' for k, v in counts_out.items() if v > 0) or 'None')
            
            if out_range_findings:
                report_lines.append('')
                report_lines.append('ℹ️  These issues exist in the original contract but are NOT in your')
                report_lines.append('   modification scope. They are listed for context only.')
                report_lines.append('')
                for i, f in enumerate(out_range_findings[:5], 1):  # Show first 5
                    report_lines.append(f"{i}. [{f['impact']}] {f['check']}")
                if len(out_range_findings) > 5:
                    report_lines.append(f'   ... and {len(out_range_findings) - 5} more (see slither.json for details)')
            
            report_lines.append('')
            report_lines.append('🎯 FN_SECURE_REFINE EVALUATION CRITERIA:')
            in_high_med = counts_in.get('High', 0) + counts_in.get('Medium', 0)
            if in_high_med == 0:
                report_lines.append('✅ PASS - No High/Medium issues in target function')
                report_lines.append('✅ Function is secure and ready!')
            else:
                report_lines.append(f'❌ FAIL - {in_high_med} High/Medium issue(s) in target function require fixing')
                report_lines.append('🔧 Apply minimal security fixes and regenerate.')
            # Record for stop-condition enforcement
            try:
                self.last_inrange_highmed = in_high_med
            except Exception:
                pass
        else:
            # Fallback: no range info
            report_lines.append('Slither Findings Summary (all):')
            report_lines.append(', '.join(f'{k}: {v}' for k, v in counts_total.items()))
            report_lines.append('')
            for i, f in enumerate(in_range_findings[:20], 1):
                report_lines.append(f"{i}. [{f['impact']}] {f['check']}")
        
        return '\n'.join(report_lines)

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        # Skip if we're in a tool-calling turn or responding to a tool
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            return
        
        # 读取forge状态
        build_ok = getattr(runtime, 'forge_last_build_ok', None)
        test_ok = getattr(runtime, 'forge_last_test_ok', None)

        # Update iteration state BEFORE pruning (prune may drop the last assistant code block).
        try:
            self._update_iter_state_from_messages(messages)
        except Exception as e:
            logger.warning('SecurityCallback.on_generate_response: failed to update iter state pre-prune: %s', e)

        # Detect stall markers BEFORE pruning as well.
        try:
            stall_detected = self._detect_stall(messages)
        except Exception:
            stall_detected = False
        
        # Prune chat history to avoid overly long contexts
        logger.info(f'SecurityCallback: before prune messages: {messages}')
        self._prune_messages(messages, build_ok, test_ok)
        logger.info(f'SecurityCallback: after prune messages: {messages}')

        try:
            required_family = self._pick_required_edit_family(stall_detected)
            policy = self._build_iter_policy_prompt(stall_detected, required_family)
            if policy:
                messages.append(Message(role='user', content=policy))
        except Exception as e:
            logger.warning('SecurityCallback.on_generate_response: failed to inject iter policy: %s', e)
        return


    def _detect_stall(self, messages: List[Message]) -> bool:
        for m in reversed(messages[-12:]):
            if not getattr(m, 'content', None):
                continue
            if '[STALL_DETECTED]' in m.content:
                return True
        return False

    def _extract_last_assistant_code(self, messages: List[Message]) -> Optional[str]:
        for m in reversed(messages):
            if getattr(m, 'role', None) != 'assistant':
                continue
            c = getattr(m, 'content', None) or ''
            if '```' not in c:
                continue
            parts = c.split('```')
            for i in range(len(parts) - 1, 0, -1):
                block = parts[i]
                if 'solidity' in block[:64]:
                    return block
                if 'function ' in block:
                    return block
            return c
        return None

    def _infer_edit_family(self, code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        s = code.lower()
        if 'mstore' in s or 'mload' in s:
            return 'FAMILY_C_MSTORE_PACK'
        if 'bytes.concat' in s or 'concat(' in s:
            return 'FAMILY_D_BYTES_CONCAT_STYLE'
        if 'assembly' not in s:
            return 'FAMILY_A_CAST_ONLY'
        if 'shl(' in s or 'shr(' in s or 'and(' in s or 'or(' in s:
            return 'FAMILY_B_SHIFT_OR_MASK'
        return None

    def _pick_required_edit_family(self, stall_detected: bool) -> str:
        """Pick a required edit family for this round to force rotation.
        When stalling, force a larger jump away from shift/mask.
        """
        last_family = self._iter_state.get('last_edit_family')
        rotation = [
            'FAMILY_A_CAST_ONLY',
            'FAMILY_C_MSTORE_PACK',
            'FAMILY_B_SHIFT_OR_MASK',
            'FAMILY_D_BYTES_CONCAT_STYLE',
            'FAMILY_E_REFERENCE_IMPL',
        ]

        if last_family not in rotation:
            return 'FAMILY_A_CAST_ONLY'

        idx = rotation.index(last_family)
        step = 2 if stall_detected else 1
        return rotation[(idx + step) % len(rotation)]

    def _update_iter_state_from_messages(self, messages: List[Message]) -> None:
        code = self._extract_last_assistant_code(messages)
        fam = self._infer_edit_family(code)
        if fam:
            self._iter_state['last_edit_family'] = fam

    def _build_iter_policy_prompt(self, stall_detected: bool, required_family: str) -> str:
        last_family = self._iter_state.get('last_edit_family') or 'UNKNOWN'

        family_constraints = {
            'FAMILY_A_CAST_ONLY': (
                '- MUST NOT use `assembly`\n'
                '- MUST use Solidity casting/bitwise ops only\n'
            ),
            'FAMILY_B_SHIFT_OR_MASK': (
                '- MUST use `assembly`\n'
                '- MUST use bitwise ops (`shl`/`shr`/`and`/`or`)\n'
            ),
            'FAMILY_C_MSTORE_PACK': (
                '- MUST use `assembly`\n'
                '- MUST use memory packing with `mstore`/`mload`\n'
            ),
            'FAMILY_D_BYTES_CONCAT_STYLE': (
                '- MUST use `bytes.concat` (or an equivalent concat)\n'
                '- MUST NOT use `assembly`\n'
            ),
            'FAMILY_E_REFERENCE_IMPL': (
                '- Use the simplest known-correct reference implementation\n'
                '- Prefer readability and correctness over micro-optimizations\n'
            ),
        }

        extra = ''
        if stall_detected:
            extra = (
                'STALL_DETECTED is true: you MUST materially change the implementation model (not just constants/renames).\n'
            )

        return (
            'ITERATIVE FIX POLICY (MUST FOLLOW)\n\n'
            'Do NOT output hypotheses/explanations; enforce diversity through implementation families.\n\n'
            'State from last round:\n'
            f'- LAST_EDIT_FAMILY: {last_family}\n'
            f'- STALL_DETECTED: {str(stall_detected).lower()}\n\n'
            f'THIS ROUND REQUIRED_EDIT_FAMILY: {required_family}\n'
            'Family constraints:\n'
            f"{family_constraints.get(required_family, '')}"
            '\n'
            'Hard requirements:\n'
            f'- MUST satisfy REQUIRED_EDIT_FAMILY and MUST differ from LAST_EDIT_FAMILY ({last_family}).\n'
            f'{extra}'
        )


    def _save_layered_reports(self, summary_text: str, ok: bool, raw_stdout: str):
        """Save layered slither summary into text and JSON files."""
        timestamp = datetime.now().isoformat()
        # Text report
        try:
            with open(self.layered_report_txt, 'w', encoding='utf-8') as f:
                f.write('Slither Layered Report\n')
                f.write(f'Generated at: {timestamp}\n')
                f.write('=' * 60 + '\n')
                f.write(f'STATUS: {"OK" if ok else "ERROR"}\n')
                f.write('=' * 60 + '\n\n')
                f.write(summary_text or '')
                f.write('\n')
            logger.info(f'Layered slither text report saved to {self.layered_report_txt}')
        except Exception as e:
            logger.warning(f'Failed to save layered slither text report: {e}')
        # JSON report (store summary text and minimal metadata)
        try:
            payload = {
                'timestamp': timestamp,
                'status': 'OK' if ok else 'ERROR',
                'summary_text': summary_text,
                'slither_stdout_stderr': raw_stdout
            }
            with open(self.layered_report_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f'Layered slither JSON report saved to {self.layered_report_json}')
        except Exception as e:
            logger.warning(f'Failed to save layered slither JSON report: {e}')
    
    async def after_tool_call(self, runtime: Runtime, messages: List[Message]):
        """
        Stop-condition control:
        - Continue iterating unless BOTH:
          (a) forge build/test passed AND
          (b) in-range High/Medium == 0.
        """
        # 打印进入时的 messages 概览（最小日志，不刷屏）
        logger.info(f'SecurityCallback messages: {messages}')
        logger.info(runtime)
        logger.info(
            'SecurityCallback.after_tool_call: messages_len=%s last_role=%s last_content_preview=%s',
            len(messages),
            messages[-1].role if messages else None,
            (messages[-1].content[:120] + '...' if len(messages[-1].content) > 120 else messages[-1].content) if messages and messages[-1].content else None,
        )

        build_ok = getattr(runtime, 'forge_last_build_ok', None)
        test_ok = getattr(runtime, 'forge_last_test_ok', None)

        logger.info(
            'SecurityCallback.debug: has_callbacks_attr=%s callbacks_type=%s callbacks_len=%s forge_last_build_ok=%s forge_last_test_ok=%s',
            hasattr(runtime, 'callbacks'),
            type(getattr(runtime, 'callbacks', None)).__name__ if getattr(runtime, 'callbacks', None) is not None else None,
            len(getattr(runtime, 'callbacks', [])) if isinstance(getattr(runtime, 'callbacks', []), list) else None,
            getattr(runtime, 'forge_last_build_ok', None),
            getattr(runtime, 'forge_last_test_ok', None),
        )

        logger.info(
            'SecurityCallback.after_tool_call: enter build_ok=%s test_ok=%s last_inrange_highmed=%s should_stop(before)=%s',
            build_ok,
            test_ok,
            getattr(self, 'last_inrange_highmed', None),
            getattr(runtime, 'should_stop', None),
        )

        # Respect any external force-stop flag (e.g., repeated identical FAILED_TESTS stall detection).
        if getattr(runtime, 'force_stop', False):
            runtime.should_stop = True
            logger.info('SecurityCallback.after_tool_call: force_stop=True -> should_stop set to True')
            return

        # If forge failed, do not run slither; keep iterating.
        if build_ok is False or test_ok is False:
            runtime.should_stop = False
            logger.info('SecurityCallback.after_tool_call: forge failed -> should_stop set to False')
            return

        # If forge hasn't run yet (unknown), do not override loop control.
        if build_ok is None or test_ok is None:
            return

        # Forge passed: MUST run slither as final completion gate.
        work_subfolder = self._parse_workdir_from_messages(messages) or self._pick_single_task_dir()
        with self.chdir_context(folder=work_subfolder):
            if not self._forge_available():
                runtime.should_stop = False
                return

            if not self._slither_available():
                all_local_files = await self.file_system.list_files()
                messages.append(Message(
                    role='user',
                    content=(
                        '[SLITHER_FEEDBACK]\n'
                        'slither not found. Please install Slither (preferably via WSL/Docker) '
                        'or ensure it is on PATH.\n'
                        f'Local files: {all_local_files}\n'
                    )
                ))
                runtime.should_stop = False
                return

            # Run slither and write JSON report
            if os.path.exists(self.slither_json):
                try:
                    os.remove(self.slither_json)
                except Exception:
                    pass
            ok, out = self._run_cmd(
                ['slither', '.', '--compile-force-framework', 'foundry', '--json', self.slither_json],
            )
            if (not ok) and os.path.exists(self.slither_json):
                ok = True
            minimal_summary, _in_range_high_med = self._parse_slither_report_minimal(self.slither_json, messages)

            baseline_report = self._load_slither_report(self.slither_baseline_json)
            after_report = self._load_slither_report(self.slither_json)
            baseline_findings = self._collect_high_medium_findings(baseline_report or {}) if baseline_report else []
            after_findings = self._collect_high_medium_findings(after_report or {}) if after_report else []

            baseline_keys = {self._finding_key(f) for f in baseline_findings}
            new_findings = [f for f in after_findings if self._finding_key(f) not in baseline_keys]

            # Record for stop-condition enforcement: require NO newly introduced High/Medium.
            self.last_inrange_highmed = len(new_findings)

            # Still save layered reports to files (use full parser for richer local debugging)
            full_summary = self._parse_slither_report(self.slither_json, messages)
            self._save_layered_reports(full_summary, ok, out)

            # Prefer diff-based mini feedback; fall back to minimal summary if baseline missing.
            if baseline_report is None:
                status_line = f'SLITHER_STATUS: {"OK" if ok else "ERROR"}'
                summary_snip = self._truncate(minimal_summary, 1500)
                feedback = (
                    '[SLITHER_FEEDBACK]\n'
                    'Baseline slither_baseline.json not found; cannot diff.\n'
                    f'{summary_snip}\n'
                    f'{status_line}\n'
                    f'Layered reports saved to: {self.layered_report_txt}, {self.layered_report_json}\n'
                    f'Raw JSON: {self.slither_json}\n'
                )
                messages.append(Message(role='user', content=feedback))
                logger.info('SecurityCallback.after_tool_call: appended SLITHER_FEEDBACK (no baseline).')
            else:
                feedback_mini = self._format_new_findings_mini(new_findings, ok, out)
                messages.append(Message(role='user', content=feedback_mini))
                logger.info('SecurityCallback.after_tool_call: appended SLITHER_FEEDBACK_MINI new_findings=%s', len(new_findings))

        # Final stop condition: forge passed AND slither has no in-range High/Medium.
        if self.last_inrange_highmed == 0:
            runtime.should_stop = True
            logger.info('SecurityCallback.after_tool_call: slither clean -> should_stop set to True')
        else:
            runtime.should_stop = False
            logger.info('SecurityCallback.after_tool_call: slither has High/Medium -> should_stop set to False')

        logger.info(
            'SecurityCallback.after_tool_call: exit last_inrange_highmed=%s should_stop(after)=%s',
            getattr(self, 'last_inrange_highmed', None),
            getattr(runtime, 'should_stop', None),
        )
