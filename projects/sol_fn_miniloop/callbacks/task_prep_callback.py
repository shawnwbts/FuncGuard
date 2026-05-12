# projects/sol_fn_miniloop/callbacks/task_prep_callback.py
import json
import os
import shutil
import subprocess
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
    s = full_signature.strip()
    m = None
    try:
        import re

        m = re.search(r"\bfunction\b\s+[A-Za-z0-9_]+\s*\(.*?\)", s, flags=re.DOTALL)
    except Exception:
        m = None
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


def _extract_function_block_from_file_text(file_text: str, full_signature: str) -> Optional[List[str]]:
    if not file_text:
        return None
    lines = file_text.splitlines(True)
    full_sig_norm = _normalize_sig_text(full_signature)
    head_norm = _normalize_sig_text(_extract_signature_head_text(full_signature))
    start_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if 'function' not in s:
            continue
        hdr = ''
        j = i
        while j < len(lines) and '{' not in hdr and ';' not in hdr:
            ln = lines[j]
            hdr += ln.strip() + ' '
            if '{' in ln or ';' in ln:
                break
            j += 1
        hdr_norm = _normalize_sig_text(hdr)
        if full_sig_norm and full_sig_norm in hdr_norm:
            start_idx = i
            break
        if head_norm and head_norm in hdr_norm:
            start_idx = i
            break
        if s.startswith('function') and head_norm and head_norm in _normalize_sig_text(s):
            start_idx = i
            break

    if start_idx is None:
        for i, line in enumerate(lines):
            if line.strip().startswith('function'):
                start_idx = i
                break
    if start_idx is None:
        return None

    brace = 0
    in_fn = False
    end_idx = None
    for k in range(start_idx, len(lines)):
        ln = lines[k]
        if not in_fn and '{' in ln:
            in_fn = True
        if in_fn:
            brace += ln.count('{') - ln.count('}')
            if brace <= 0:
                end_idx = k
                break
    if end_idx is None:
        end_idx = min(len(lines) - 1, start_idx)
    return lines[start_idx : end_idx + 1]


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


class TaskPrepCallback(Callback):
    """
    Prepare a per-task Foundry workspace using local sources under workspace/root/.
    - Parse the first user JSON for fields: task_id, source_id, start, end, full_signature, prompt.
    - Find the nearest Foundry project root (folder containing foundry.toml) for the source file.
    - Copy that project into output/task_<task_id>/ (as the project root).
    - Compute TARGET_FILE as the path relative to the project root.
    - Append a user message with WORKDIR, TARGET_FILE, RANGE, and include foundry.toml and the full original file content.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.output_dir = getattr(config, 'output_dir', 'output')

    def _tool_available(self, name: str) -> bool:
        try:
            return shutil.which(name) is not None
        except Exception:
            return False

    def _run_cmd(self, cmd: List[str], cwd: str) -> Tuple[bool, str]:
        try:
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            out = (p.stdout or '') + (p.stderr or '')
            return p.returncode == 0, out
        except Exception as e:
            return False, str(e)

    def _maybe_generate_slither_baseline(self, runtime: Runtime, messages: List[Message], dest_root: str):
        runtime_tag = getattr(runtime, 'tag', '')
        is_fn_coding = 'fn_coding' in str(runtime_tag)
        if not is_fn_coding:
            return

        baseline_abs_path = os.path.join(dest_root, 'slither_baseline.json')
        baseline_rel_path = 'slither_baseline.json'
        if os.path.exists(baseline_abs_path):
            return

        if not self._tool_available('slither'):
            messages.append(Message(role='user', content=(
                '[SLITHER_BASELINE]\n'
                'slither not found; baseline skipped.\n'
            )))
            return

        if self._tool_available('forge'):
            self._run_cmd(['forge', 'clean'], cwd=dest_root)
            self._run_cmd(['forge', 'build'], cwd=dest_root)

        ok, out = self._run_cmd(
            ['slither', '.', '--compile-force-framework', 'foundry', '--json', baseline_rel_path],
            cwd=dest_root,
        )

        # Some Slither runs can return non-zero but still write the JSON file.
        if (not ok) and os.path.exists(baseline_abs_path):
            ok = True

        if not ok:
            messages.append(Message(role='user', content=(
                '[SLITHER_BASELINE]\n'
                'failed to generate slither_baseline.json.\n'
                f'Error:\n{out[:2000]}\n'
            )))
        else:
            messages.append(Message(role='user', content=(
                '[SLITHER_BASELINE]\n'
                'saved: slither_baseline.json\n'
            )))

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()
        try:
            task = self._extract_task(messages)
            if not task:
                return

            task_id = str(task.get('task_id', 'unknown'))
            source_id = str(task.get('source_id', '')).strip()
            start = task.get('start', '')
            end = task.get('end', '')
            full_signature = task.get('full_signature', '') or ''
            
            # Convert start/end to integers if they're not already
            try:
                start = int(start) if start else None
                end = int(end) if end else None
            except (ValueError, TypeError):
                start = end = None

            # Resolve local absolute path to the source file based on NEW dataset layout
            # Preferred base: projects/sol_fn_miniloop/root (under current workspace)
            workspace_root = os.getcwd()
            local_dir = getattr(self.config, 'local_dir', '') or 'projects/sol_fn_miniloop'
            dataset_root = os.path.join(workspace_root, local_dir, 'root')
            if not os.path.isdir(dataset_root):
                # Backward compatibility: try legacy workspace/root
                legacy_root = os.path.join(workspace_root, 'root')
                if os.path.isdir(legacy_root):
                    dataset_root = legacy_root

            norm = (source_id or '').replace('\\', '/').lstrip('/')
            candidates = []
            # 1) direct under dataset_root
            candidates.append(os.path.join(dataset_root, norm))
            # 2) if norm starts with 'root/', drop the prefix
            if norm.startswith('root/'):
                candidates.append(os.path.join(dataset_root, norm[len('root/'):]))
            # 3) legacy workspace joins
            candidates.append(os.path.join(workspace_root, norm))
            candidates.append(os.path.join(workspace_root, 'root', norm))

            local_source = None
            for c in candidates:
                if os.path.exists(c):
                    local_source = c
                    break

            if not local_source:
                tried = '\n'.join([f'- {os.path.abspath(c)}' for c in candidates])
                messages.append(Message(role='user', content=(
                    '[TASK_PREP]\n'
                    f'Cannot locate source file from source_id: {source_id}\n'
                    f'Tried paths (in order):\n{tried}\n'
                    'Please check the dataset or adjust mapping.'
                )))
                return

            project_root = self._find_nearest_foundry_root(local_source)
            if not project_root:
                project_root = os.path.dirname(local_source)
                logger.warning(f'TaskPrep: foundry.toml not found; using parent folder as project root: {project_root}')

            task_dir_name = f'task_{task_id}'
            dest_root = os.path.join(self.output_dir, task_dir_name)
            logger.info(f'Destination root: {dest_root}')
            
            # Check if this is fn_secure_refine stage to avoid deleting fn_coding results
            runtime_tag = getattr(runtime, 'tag', '')
            is_fn_secure_refine = 'fn_secure_refine' in str(runtime_tag)
            
            if is_fn_secure_refine and os.path.exists(dest_root):
                logger.info(f'[{runtime_tag}] Reusing existing task directory: {dest_root}')
            else:
                # Clean and create fresh directory
                if os.path.exists(dest_root):
                    try:
                        shutil.rmtree(dest_root)
                    except Exception as e:
                        logger.warning(f'[{runtime_tag}] Failed to clean old task dir: {e}')
                shutil.copytree(project_root, dest_root)

            self._maybe_generate_slither_baseline(runtime, messages, dest_root)

            # Compute target file path
            target_rel = os.path.relpath(local_source, start=project_root)
            target_rel_posix = target_rel.replace('\\', '/')

            # Read foundry.toml
            foundry_path = os.path.join(dest_root, 'foundry.toml')
            try:
                with open(foundry_path, 'r', encoding='utf-8') as f:
                    foundry_toml = f.read()
            except Exception:
                foundry_toml = ''

            # Check current stage and prepare context accordingly
            is_fn_secure_refine = 'fn_secure_refine' in str(runtime.tag) if hasattr(runtime, 'tag') else False
            is_fn_coding = 'fn_coding' in str(runtime.tag) if hasattr(runtime, 'tag') else False
            
            # 保存runtime tag供后续使用
            self.runtime_tag = runtime.tag if hasattr(runtime, 'tag') else ''
            
            # Initialize prep_msg list
            prep_msg = []
            
            if is_fn_secure_refine:
                # fn_secure_refine阶段：过滤冗余的fn_coding指令，然后使用专门的智能集成方法
                self._filter_fn_coding_messages(messages)
                self._prepare_fn_secure_refine_context(task, dest_root, target_rel_posix, start, end, full_signature, prep_msg)
            elif is_fn_coding:
                # fn_coding阶段：使用最小上下文模式
                self._prepare_fn_coding_context(task, dest_root, target_rel_posix, start, end, full_signature, prep_msg)
            else:
                # 其他阶段：使用传统的前文后文模式
                self._prepare_traditional_context(task, dest_root, target_rel, foundry_toml, start, end, full_signature, prep_msg)
            
            messages.append(Message(role='user', content='\n'.join(prep_msg)))

        except Exception as e:
            logger.exception('TaskPrep: unexpected error')
            messages.append(Message(role='user', content=f'[TASK_PREP]\nUnexpected error: {e}'))

    def _extract_task(self, messages: List[Message]) -> Optional[dict]:
        """Extract task JSON from messages"""
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            content = m.content.strip()
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and ('source_id' in obj or 'task_id' in obj):
                    # Filter out canonical_solution to prevent ground truth leakage
                    obj.pop('canonical_solution', None)
                    return obj
            except Exception:
                pass
            # Try fenced code block
            start = content.find('```json')
            if start != -1:
                end = content.find('```', start + 7)
                if end != -1:
                    snippet = content[start + 7:end].strip()
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict) and ('source_id' in obj or 'task_id' in obj):
                            obj.pop('canonical_solution', None)
                            return obj
                    except Exception:
                        pass
        return None

    def _find_nearest_foundry_root(self, file_path: str) -> Optional[str]:
        """Find the nearest foundry.toml directory"""
        cur = os.path.abspath(file_path)
        if os.path.isfile(cur):
            cur = os.path.dirname(cur)
        while True:
            cand = os.path.join(cur, 'foundry.toml')
            if os.path.exists(cand):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    def _prepare_traditional_context(self, task: dict, dest_root: str, target_rel: str, foundry_toml: str, start: Optional[int], end: Optional[int], full_signature: str, prep_msg: List[str]):
        """传统模式：读取原始文件，准备前文后文上下文"""
        # 提取task目录名（如output/task_384 -> task_384）
        task_dir_name = os.path.basename(dest_root)
        
        # 检查是否是fn_coding阶段 - 使用传入的参数判断
        # 因为这个方法只在非fn_secure_refine阶段调用，所以如果是fn_coding应该直接调用对应方法
        # 为了简化，我们让调用者在on_task_begin中直接路由到正确的方法
        
        # 其他阶段：使用传统的前文后文模式
        # Read original file and split into prefix/suffix (excluding target function range)
        copied_source = os.path.join(dest_root, target_rel)
        original_lines = []
        prefix_lines = []
        suffix_lines = []
        
        try:
            with open(copied_source, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Only process prefix/suffix if we have valid start/end integers
            if start is not None and end is not None:
                # Convert line numbers to 0-based indices
                start_idx = max(0, start - 1)
                end_idx = min(len(original_lines), end)
                
                # Extract prefix (lines before start) and suffix (lines after end)
                prefix_lines = original_lines[:start_idx]
                suffix_lines = original_lines[end_idx:]
            
        except Exception as e:
            logger.warning(f'Failed to read/process source file: {e}')

        # Append context message for the coding agent (with prefix/suffix but without target function implementation)
        prep_msg.extend([
            '[TASK_PREP]',
            f'WORKDIR: {task_dir_name}',
            f'TARGET_FILE: {target_rel}',
        ])
        
        # Calculate dynamic range for function-level replacement
        if start is not None and end is not None:
            # For dynamic function generation, we only constrain the function signature line
            # This allows AI to generate functions of any length
            dynamic_start = start
            dynamic_end = start  # Only constrain the signature line
            prep_msg.append(f'RANGE: {dynamic_start}-{dynamic_end}')
            prep_msg.append(f'DYNAMIC_MODE: true')
        else:
            prep_msg.append(f'RANGE: {start}-{end}')
            prep_msg.append(f'DYNAMIC_MODE: false')
        
        prep_msg.extend([
            f'FULL_SIGNATURE: {full_signature}',
            '',
            '📄 ORIGINAL FILE CONTEXT:',
            f'File: {target_rel}',
            f'Range to modify: [{start}:{end}]',
            '',
            '📄 PREFIX (lines 1-{}):'.format(start - 1 if start else 'start'),
            '```solidity',
        ])
        
        # Add prefix content
        prep_msg.extend([line.rstrip() for line in prefix_lines])
        prep_msg.extend([
            '```',
            '',
            '🎯 TARGET FUNCTION (lines {}-{}):'.format(start, end),
            '```solidity',
            '// Function implementation needed here',
            '```',
            '',
            '📄 SUFFIX (lines {}-end):'.format(end + 1),
            '```solidity',
        ])
        
        # Add suffix content
        prep_msg.extend([line.rstrip() for line in suffix_lines])
        prep_msg.extend([
            '```',
            '',
            '🔧 TASK:',
            '- Implement the target function based on the provided signature and requirements',
            '- Keep all changes within the specified range',
            '- Maintain exact prefix and suffix as shown above',
        ])

    def _prepare_fn_coding_context(self, task: dict, dest_root: str, target_rel_posix: str, start: Optional[int], end: Optional[int], full_signature: str, prep_msg: List[str]):
        """为fn_coding阶段准备上下文：只提供prompt和full_signature，不提供前文后文"""
        prompt = task.get('prompt', '')
        
        # 提取task目录名（如output/task_384 -> task_384）
        task_dir_name = os.path.basename(dest_root)
        logger.info(f'Task dir name: {task_dir_name}')
        logger.info(f'Target rel posix: {target_rel_posix}')
        
        prep_msg.extend([
            '[TASK_PREP]',
            f'WORKDIR: {task_dir_name}',
            f'TARGET_FILE: {target_rel_posix}',
            f'RANGE: {start}-{end}',
            'DYNAMIC_MODE: false',
            '',
            'FN_CODING STAGE - MINIMAL CONTEXT MODE',
            '',
            'TASK PROMPT:',
            prompt,
            '',
            f'FUNCTION SIGNATURE:',
            full_signature,
            '',
            'OUTPUT INSTRUCTIONS:',
            '- Generate ONLY the function implementation based on the prompt and signature',
            '- Do NOT include any prefix/suffix context - generate standalone function',
            '- Output as a complete Solidity file with just your function',
            '- Focus on correctness and security best practices',
            '- CRITICAL: You MUST use EXACTLY this path in the code block header:',
            f'  {task_dir_name}/{target_rel_posix}',
            '- CRITICAL: Do NOT use source_id paths like /root/... or any absolute path',
            '',
            'Example format:',
            f'```solidity: {task_dir_name}/{target_rel_posix}',
            full_signature,
            '{',
            '    // Your implementation here',
            '}',
            '```',
        ])

    def _prepare_fn_secure_refine_context(self, task: dict, dest_root: str, target_rel_posix: str, start: Optional[int], end: Optional[int], full_signature: str, prep_msg: List[str]):
        """fn_secure_refine阶段：只复制原始完整文件，不进行函数替换"""
        # 提取task目录名（如output/task_384 -> task_384）
        task_dir_name = os.path.basename(dest_root)
        
        # 从source_id复制原始完整文件（含上下文）
        source_id = task.get('source_id', '')
        if not source_id:
            logger.error('No source_id found in task')
            return
            
        # 尝试从数据集根目录找到原始文件
        workspace_root = os.getcwd()
        logger.info(f'Workspace root: {workspace_root}')
        local_dir = getattr(self.config, 'local_dir', '') or 'projects/sol_fn_miniloop'
        dataset_root = os.path.join(workspace_root, local_dir, 'root')
        logger.info(f'Dataset root: {dataset_root}')
        
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
            logger.error(f'Could not find original file for source_id: {source_id}')
            return
            
        original_file_path = os.path.join(dest_root, target_rel_posix)

        snippet_lines: Optional[List[str]] = None
        try:
            if os.path.exists(original_file_path):
                with open(original_file_path, 'r', encoding='utf-8') as fprev:
                    prev_text = fprev.read()
                snippet_lines = _extract_function_block_from_file_text(prev_text, full_signature)
        except Exception:
            snippet_lines = None

        shutil.copy2(original_source, original_file_path)
        logger.info(f'Copied original file from {original_source} to {original_file_path}')

        try:
            if snippet_lines:
                with open(original_file_path, 'r', encoding='utf-8') as fsrc:
                    original_lines = fsrc.readlines()
                old_start, old_end = _locate_function_in_lines(original_lines, full_signature, expected_start=start)
                if old_start and old_end:
                    old_start_idx = old_start - 1
                    old_end_idx = old_end
                    new_lines = original_lines[:old_start_idx] + snippet_lines + original_lines[old_end_idx:]
                    with open(original_file_path, 'w', encoding='utf-8') as fdst:
                        fdst.writelines(new_lines)
        except Exception:
            pass
        
        # 使用初始 range（来自 task JSON）
        if start is None or end is None:
            logger.error('Start or end position not specified')
            return
            
        # 准备精简的上下文信息（不包含完整文件内容）
        prep_msg.extend([
            '',
            'FN_SECURE_REFINE STAGE - TOKEN-OPTIMIZED MODE',
            '',
            f'WORKDIR: {task_dir_name}',
            f'TARGET_FILE: {target_rel_posix}',
            f'RANGE: [{start}:{end}]',
            f'FUNCTION_SIGNATURE: {full_signature}',
            '',
            'SECURITY OPTIMIZATION INSTRUCTIONS:',
            '- Focus ONLY on security issues within the specified function range',
            '- Use file reading tools to examine the target function as needed',
            '- Keep all changes within the allowed range [{start}:{end}]',
            '- Do NOT modify any code outside the target function',
            '',
            'Start security optimization now!',
        ])
        
    def _filter_fn_coding_messages(self, messages: List[Message]):
        """
        在fn_secure_refine阶段过滤掉冗余的fn_coding指令信息，
        只保留必要的上下文：Task JSON
        """
        if not messages:
            return
            
        # 找到fn_coding阶段的消息范围
        fn_coding_start_idx = -1
        fn_coding_end_idx = -1
        
        for i, msg in enumerate(messages):
            content = msg.content if msg.content else ''
            if 'FN_CODING STAGE - MINIMAL CONTEXT MODE' in content:
                fn_coding_start_idx = i
            elif fn_coding_start_idx != -1 and content.startswith('Save file <') and 'successfully.' in content:
                fn_coding_end_idx = i
                break
        
        # 如果找到了fn_coding阶段的消息，进行过滤
        if fn_coding_start_idx != -1 and fn_coding_end_idx != -1:
            filtered_messages = []
            original_count = len(messages)
            
            # 保留fn_coding之前的所有消息
            filtered_messages.extend(messages[:fn_coding_start_idx])
            
            # 在fn_coding阶段中，只保留Task JSON
            for i in range(fn_coding_start_idx, fn_coding_end_idx + 1):
                msg = messages[i]
                content = msg.content if msg.content else ''
                
                # 只保留Task JSON
                if msg.role == 'user' and content.strip().startswith('{') and 'task_id' in content:
                    filtered_messages.append(msg)
                # 过滤掉所有其他消息，包括生成的函数代码
            
            # 保留fn_coding之后的消息，但过滤掉保存确认和TASK_PREP消息
            for i in range(fn_coding_end_idx + 1, len(messages)):
                msg = messages[i]
                content = msg.content if msg.content else ''
                
                # 过滤掉保存确认和TASK_PREP消息
                if not (content.startswith('Save file <') and 'successfully.' in content) and \
                   not (content.startswith('[TASK_PREP]') or content.startswith('WORKDIR:') or 
                        content.startswith('TARGET_FILE:') or content.startswith('RANGE:') or
                        content.startswith('ORIGINAL_RANGE:') or content.startswith('DYNAMIC_MODE:') or
                        content.startswith('FULL_SIGNATURE:')):
                    filtered_messages.append(msg)
            
            # 替换原消息列表
            messages.clear()
            messages.extend(filtered_messages)
            
            removed_count = original_count - len(filtered_messages)
            logger.info(f'Filtered fn_coding stage: removed {removed_count} redundant messages, keeping only Task JSON')
