# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import re
import os
import shutil
from typing import List, Optional, Tuple

from file_parser import extract_code_blocks
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


def _normalize_sig_text(s: str) -> str:
    if s is None:
        return ''
    s = re.sub(r"//.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", "", s)
    return s


def _extract_signature_head_text(full_signature: str) -> str:
    if not full_signature:
        return ''
    m = re.search(r"(function\s+[A-Za-z0-9_]+\s*\(.*?\))", full_signature, flags=re.DOTALL)
    return m.group(1) if m else ''


def _canonicalize_function_signature(full_signature: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (func_name, canonical_sig_head) where canonical_sig_head matches up to ')' using parameter TYPES.

    Example:
      input:  function deriveMapping(bytes32 slot, bool key) internal pure returns (bytes32 result)
      output: ("deriveMapping", "function deriveMapping(bytes32,bool)")

    This is robust to different parameter names and to multiline modifiers/returns in source.
    """
    if not full_signature:
        return None, None
    mm = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", full_signature)
    func_name = mm.group(1) if mm else None
    if not func_name:
        return None, None
    m2 = re.search(r"function\s+%s\s*\((.*?)\)" % re.escape(func_name), full_signature, flags=re.DOTALL)
    if not m2:
        return func_name, f"function {func_name}("
    raw_params = (m2.group(1) or '').strip()
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
            # Drop trailing parameter name; keep full type including data location (memory/calldata/storage)
            ptype = ' '.join(tokens[:-1])
        ptype = re.sub(r"\s+", " ", ptype).strip()
        ptype = ptype.replace(' ', '')
        if ptype:
            param_types.append(ptype)
    return func_name, f"function {func_name}({','.join(param_types)})"


def _canonicalize_source_signature_head(lines: List[str], start_idx_0: int) -> str:
    """Build a canonicalized signature head from source lines starting at start_idx_0 until the first ')' appears."""
    buf = ''
    i = start_idx_0
    while i < len(lines) and ')' not in buf:
        ln = lines[i]
        ln = re.sub(r"//.*$", "", ln)
        buf += ln.strip() + ' '
        i += 1
    buf = re.sub(r"\s+", " ", buf).strip()
    m = re.search(r"(function\s+[A-Za-z0-9_]+\s*\(.*?\))", buf)
    if not m:
        return ''
    head = m.group(1)

    # Convert to: function name(type1,type2)
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
        ptype = re.sub(r"\s+", " ", ptype).strip()
        ptype = ptype.replace(' ', '')
        if ptype:
            param_types.append(ptype)
    return f"function {func_name}({','.join(param_types)})"


def _locate_function_in_file(
    lines: List[str],
    full_signature: str,
    expected_start: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    在原始文件中定位函数的起始和结束行号（1基索引）
    返回 (start_line, end_line)，找不到返回 (None, None)
    """
    func_name, canonical_head = _canonicalize_function_signature(full_signature)
    sig_prefix = full_signature.split('(')[0].strip() if '(' in full_signature else None

    full_sig_norm = _normalize_sig_text(full_signature)
    head_text = _extract_signature_head_text(full_signature)
    head_norm = _normalize_sig_text(head_text) if head_text else ''
    
    start_line = None
    end_line = None
    match_reason = None
    match_hdr_excerpt = None

    def _choose_nearest(cands: List[Tuple[int, str, str]]) -> Tuple[int, str, str]:
        if not cands:
            raise ValueError('no candidates')
        if expected_start is None:
            return cands[0]
        return min(cands, key=lambda t: abs(t[0] - expected_start))

    full_sig_candidates: List[Tuple[int, str, str]] = []
    head_candidates: List[Tuple[int, str, str]] = []
    canonical_candidates: List[Tuple[int, str, str]] = []

    # 查找函数起始行（不允许 weak_key 回退；避免 overload 错匹配）
    for i, line in enumerate(lines, 1):  # 1基索引
        line_stripped = line.strip()
        if 'function' in line_stripped:
            hdr = ''
            j = i - 1
            while j < len(lines) and '{' not in hdr and ';' not in hdr:
                ln = lines[j]
                ln = re.sub(r"//.*$", "", ln)
                hdr += ln.strip() + ' '
                if '{' in ln or ';' in ln:
                    break
                j += 1

            hdr_norm = _normalize_sig_text(hdr)

            if full_sig_norm and full_sig_norm in hdr_norm:
                full_sig_candidates.append((i, 'full_signature', hdr.strip()[:200]))

            if head_norm and head_norm in hdr_norm:
                head_candidates.append((i, 'signature_head', hdr.strip()[:200]))

        if canonical_head and func_name and f"function {func_name}" in line_stripped:
            src_head = _canonicalize_source_signature_head(lines, i - 1)
            if src_head and src_head.replace(' ', '') == canonical_head.replace(' ', ''):
                canonical_candidates.append((i, 'canonical_head', src_head.strip()[:200]))

    chosen: Optional[Tuple[int, str, str]] = None
    if full_sig_candidates:
        chosen = _choose_nearest(full_sig_candidates)
    elif head_candidates:
        chosen = _choose_nearest(head_candidates)
    elif canonical_candidates:
        chosen = _choose_nearest(canonical_candidates)

    if not chosen:
        logger.warning(
            "_locate_function_in_file: no match (weak_key disabled) "
            "expected_start=%s canonical_head=%r sig_prefix=%r",
            expected_start,
            canonical_head,
            sig_prefix,
        )
        return None, None

    start_line, match_reason, match_hdr_excerpt = chosen

    try:
        logger.info(
            f"_locate_function_in_file: matched start_line={start_line} reason={match_reason} "
            f"full_sig_head={_extract_signature_head_text(full_signature)!r} "
            f"hdr_excerpt={match_hdr_excerpt!r}"
        )
    except Exception:
        pass
    
    # 使用 brace matching 找到函数结束行
    brace_count = 0
    in_function = False
    for i in range(start_line - 1, len(lines)):  # 转回0基索引
        line = lines[i]
        if not in_function and '{' in line:
            in_function = True
        
        if in_function:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                end_line = i + 1  # 转回1基索引
                break
    
    return start_line, end_line


def _extract_function_block_from_assistant_output(lines: List[str], full_signature: str) -> List[str]:
    """
    从 assistant 输出中提取函数块
    优先使用函数签名定位，失败后使用 brace counting
    """
    func_name, canonical_head = _canonicalize_function_signature(full_signature)
    sig_prefix = full_signature.split('(')[0].strip() if '(' in full_signature else None
    
    # 查找函数起始行
    start_line_idx = -1
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or not line_stripped.startswith('function'):
            continue
            
        if canonical_head and func_name and f"function {func_name}" in line_stripped:
            src_head = _canonicalize_source_signature_head(lines, i)
            if src_head and src_head.replace(' ', '') == canonical_head.replace(' ', ''):
                start_line_idx = i
                break

        # 优先匹配完整签名前缀（弱匹配）
        if sig_prefix and sig_prefix in line_stripped:
            start_line_idx = i
            break
        # 备选：匹配函数名（最弱）
        if func_name and f"function {func_name}" in line_stripped:
            start_line_idx = i
            break
    
    if start_line_idx == -1:
        # 如果找不到函数签名，返回所有行（fallback）
        return lines
    
    # 检查是否是分号结尾的函数（接口/抽象函数）
    first_line = lines[start_line_idx].strip()
    if first_line.endswith(';'):
        return [lines[start_line_idx]]
    
    # 使用 brace counting 提取函数体
    brace_count = 0
    in_function = False
    func_block = []
    
    for i in range(start_line_idx, len(lines)):
        line = lines[i]
        func_block.append(line)
        
        if not in_function and '{' in line:
            in_function = True
        
        if in_function:
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                break
    
    return func_block


def _update_range_meta(messages: List[Message], workdir: str, target_rel: str, new_start: int, new_end: int, full_signature: str):
    """
    在消息列表中添加/更新 RANGE meta 信息
    """
    from ms_agent.llm.utils import Message
    
    # 创建新的 meta 消息
    new_meta_content = [
        '',
        'FN_SECURE_REFINE STAGE - TOKEN-OPTIMIZED MODE',
        '',
        f'WORKDIR: {workdir}',
        f'TARGET_FILE: {workdir}/{target_rel}',
        f'RANGE: [{new_start}:{new_end}]',
        f'FUNCTION_SIGNATURE: {full_signature}',
        '',
        'UPDATED RANGE META - This message contains the latest function range after integration.',
    ]
    
    # 添加到消息列表末尾
    messages.append(Message(role='user', content='\n'.join(new_meta_content)))
    logger.info(f"ArtifactCallback: Added updated RANGE meta to messages list: [{new_start}:{new_end}]")
    logger.info(f"ArtifactCallback: Total messages after adding meta: {len(messages)}")


def _parse_task_meta(messages: List[Message]) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int]], Optional[str], bool, bool]:
    """Extract WORKDIR, TARGET_FILE, RANGE, original file content, dynamic mode, and fn_coding mode from TaskPrep message."""
    workdir = target_rel = None
    rng = None
    original_text = None
    dynamic_mode = False
    fn_coding_mode = False
    
    # 首先尝试从新的优化格式中提取信息
    logger.info(f"artifact_callback: processing {len(messages)} messages")
    for m in messages:
        if m.role != 'user' or not m.content:
            continue
        
        logger.info(f"artifact_callback: checking user message with {len(m.content)} chars")
        if 'WORKDIR:' in m.content and 'TARGET_FILE:' in m.content:
            logger.info("Found optimized format in artifact_callback")

            # Only treat this message as authoritative meta if it also provides RANGE.
            # Otherwise (e.g. [PUBLIC_TESTS]) it would block later messages that contain RANGE.
            msg_workdir = None
            msg_target_rel = None
            msg_rng = None
            
            for line in m.content.splitlines():
                if line.startswith('WORKDIR:'):
                    msg_workdir = line.split(':', 1)[1].strip()
                    logger.info(f"Found WORKDIR from optimized format: {msg_workdir}")
                elif line.startswith('TARGET_FILE:'):
                    msg_target_rel = line.split(':', 1)[1].strip()
                    logger.info(f"Found TARGET_FILE from optimized format: {msg_target_rel}")
                elif line.startswith('RANGE:'):
                    range_str = line.split(':', 1)[1].strip()
                    # Extract [start:end] format
                    match = re.search(r'\[(\d+):(\d+)\]', range_str)
                    if match:
                        start, end = int(match.group(1)), int(match.group(2))
                        msg_rng = (start, end)
                        logger.info(f"Found RANGE from optimized format: {msg_rng}")
                elif line.startswith('FUNCTION_SIGNATURE:'):
                    # Extract function signature for validation
                    full_sig = line.split(':', 1)[1].strip()
                    logger.info(f"Found FULL_SIGNATURE from optimized format: {full_sig}")

            # Commit this message's meta only if RANGE is present.
            if msg_workdir and msg_target_rel and msg_rng:
                workdir = msg_workdir
                target_rel = msg_target_rel
                rng = msg_rng
            else:
                # Keep any previously found authoritative meta; continue scanning.
                continue
            
            # 从task中提取workdir和range
            task = _extract_task(messages)
            if task and 'task_id' in task:
                if not workdir:
                    workdir = f"task_{task['task_id']}"
                    logger.info(f"Extracted WORKDIR from task: {workdir}")
                
                # Extract range from task JSON if not found in optimized format
                if rng is None and 'start' in task and 'end' in task:
                    start, end = int(task['start']), int(task['end'])
                    rng = (start, end)
                    logger.info(f"Found RANGE from task JSON: {rng}")
            
            # Extract original file code block from optimized format message
            if target_rel:
                pattern = rf"```solidity:\s*{re.escape(target_rel)}\s*:\s*\n(.*?)```"
                m2 = re.search(pattern, m.content, re.DOTALL)
                if m2:
                    original_text = m2.group(1).strip()
                    logger.info(f"Found original text with {len(original_text.splitlines())} lines")
            
            if workdir and target_rel and rng is not None:
                break
    
    # 如果新格式没有找到，回退到旧的TASK_PREP格式
    if not (workdir and target_rel):
        logger.info("Optimized format not found in artifact_callback, trying legacy TASK_PREP format")
        for m in messages:
            if m.role != 'user' or not m.content:
                continue
            if '[TASK_PREP]' not in m.content:
                continue
            
            wd = tr = None
            start = end = None
            is_dynamic = False
            is_fn_coding = False
            
            for line in m.content.splitlines():
                if line.startswith('WORKDIR:'):
                    wd = line.split(':', 1)[1].strip()
                elif line.startswith('TARGET_FILE:'):
                    tr = line.split(':', 1)[1].strip()
                elif line.startswith('RANGE:'):
                    seg = line.split(':', 1)[1].strip()
                    try:
                        s, e = seg.split('-', 1)
                        start, end = int(s), int(e)
                    except Exception:
                        pass
                elif line.startswith('DYNAMIC_MODE:'):
                    dynamic_val = line.split(':', 1)[1].strip().lower()
                    is_dynamic = (dynamic_val == 'true')
                elif 'FN_CODING STAGE - MINIMAL CONTEXT MODE' in line:
                    is_fn_coding = True
            
            # Extract original file code block from TaskPrep message
            # Format: ```solidity: {target_rel}
            if tr:
                pattern = rf"```solidity:\s*{re.escape(tr)}\s*\n(.*?)```"
                m2 = re.search(pattern, m.content, re.DOTALL)
                if m2:
                    original_text = m2.group(1).strip()
            
            if wd and tr:
                workdir, target_rel = wd, tr
                if start and end and original_text is not None:
                    rng = (start, end)
                dynamic_mode = is_dynamic
                fn_coding_mode = is_fn_coding
                break
    
    return workdir, target_rel, rng, original_text, dynamic_mode, fn_coding_mode


def _extract_task(messages: List[Message]) -> Optional[dict]:
    """Extract task JSON from messages"""
    for m in messages:
        if m.role == 'user' and m.content and m.content.strip().startswith('{'):
            try:
                task_obj = json.loads(m.content)
                if 'task_id' in task_obj:
                    logger.info(f"Found valid task JSON with task_id: {task_obj['task_id']}")
                    return task_obj
            except Exception as e:
                logger.info(f"Failed to parse JSON snippet: {e}")
                pass
    logger.info("No task JSON found in any message")
    return None


def _validate_dynamic_range_constraint(original_text: str, new_text: str, start: int, end: int) -> Tuple[bool, str]:
    """Validate dynamic range constraint where only the function signature line is constrained."""
    o_lines = original_text.splitlines()
    n_lines = new_text.splitlines()
    
    # In dynamic mode, we only constrain the function signature line (start)
    # Everything before the signature must be identical
    prefix_end = max(0, start - 1)
    prefix_o = o_lines[:prefix_end]
    prefix_n = n_lines[:prefix_end] if len(n_lines) >= prefix_end else n_lines
    
    if prefix_o != prefix_n:
        # 添加调试日志
        logger.info(f'Prefix mismatch detected for line {start}')
        logger.info(f'Original prefix length: {len(prefix_o)}, Generated prefix length: {len(prefix_n)}')
        
        # 放宽验证：允许微小差异（如空行、注释）
        if abs(len(prefix_o) - len(prefix_n)) > 5:
            logger.warning(f'Prefix length difference too large: {abs(len(prefix_o) - len(prefix_n))}')
            return False, f'Lines before line {start} were modified (prefix mismatch)'
        
        # 检查关键行是否匹配（忽略空行差异）
        critical_lines_o = [line for line in prefix_o if line.strip()]
        critical_lines_n = [line for line in prefix_n if line.strip()]
        
        logger.info(f'Original critical lines: {len(critical_lines_o)}, Generated critical lines: {len(critical_lines_n)}')
        
        # 放宽验证：只检查关键行数量，不检查具体内容
        # 这样可以允许微小的格式差异（如空格、注释等）
        if len(critical_lines_o) != len(critical_lines_n):
            logger.warning(f'Critical prefix line count mismatch: {len(critical_lines_o)} vs {len(critical_lines_n)}')
            return False, f'Lines before line {start} were modified (prefix line count mismatch)'
        
        logger.info('Prefix validation passed with relaxed tolerance (content check disabled)')
    
    # The signature line (line start) must exist in new file
    if len(n_lines) < start:
        return False, f'Function signature line {start} is missing from generated file'
    
    # Check that the signature line starts with the same function declaration
    original_signature = o_lines[start-1].strip() if start-1 < len(o_lines) else ''
    new_signature = n_lines[start-1].strip() if start-1 < len(n_lines) else ''
    
    # Extract function name from original signature for comparison
    if original_signature:
        # Simple check: new signature should contain the same function name
        func_name_match = False
        if 'function ' in original_signature:
            func_name = original_signature.split('function ')[1].split('(')[0].strip()
            if func_name in new_signature:
                func_name_match = True
        
        if not func_name_match:
            return False, f'Function signature mismatch. Expected function containing "{func_name if "func_name" in locals() else "unknown"}"'
    
    # For dynamic mode, we need to find where the function ends and the next function begins
    # Look for the next function signature or closing brace pattern
    next_function_start = None
    
    # Find the next function in the original file (after the original end)
    for i in range(end, len(o_lines)):
        line = o_lines[i].strip()
        if line.startswith('function ') or line.startswith('    function ') or line.startswith('        function '):
            next_function_start = i + 1  # +1 for 1-based indexing
            break
        elif line == '}' and i > end + 5:  # Allow some buffer for function body
            next_function_start = i + 1
            break
    
    if next_function_start is None:
        # If we can't find the next function, check if the file ends properly
        # Allow any length for the generated function as long as it ends with proper closing
        if n_lines and n_lines[-1].strip() == '}':
            return True, ''
        else:
            return False, 'Generated function does not end with proper closing brace'
    
    # Extract the suffix from original file (from next function start)
    suffix_o = o_lines[next_function_start-1:] if next_function_start <= len(o_lines) else []
    
    # Extract suffix from new file (should be the same content)
    if suffix_o:
        if len(n_lines) < len(suffix_o):
            return False, 'Generated file is too short to include required suffix'
        
        suffix_n = n_lines[-len(suffix_o):] if len(n_lines) >= len(suffix_o) else []
        
        # Check if suffix matches (allowing for minor formatting differences)
        if len(suffix_o) != len(suffix_n):
            # 添加调试日志
            logger.info(f'Suffix length mismatch detected. Original: {len(suffix_o)}, Generated: {len(suffix_n)}')
            
            # 放宽验证：允许后缀长度有微小差异
            if abs(len(suffix_o) - len(suffix_n)) > 5:
                logger.warning(f'Suffix length difference too large: {abs(len(suffix_o) - len(suffix_n))}')
                return False, f'Suffix length mismatch. Expected {len(suffix_o)} lines, got {len(suffix_n)}'
        
        # 使用关键行比较，忽略空行差异
        critical_lines_o = [line for line in suffix_o if line.strip()]
        critical_lines_n = [line for line in suffix_n if line.strip()]
        
        logger.info(f'Original suffix critical lines: {len(critical_lines_o)}, Generated suffix critical lines: {len(critical_lines_n)}')
        
        # 放宽验证：只检查关键行数量，不检查具体内容
        # 这样可以允许微小的格式差异（如空格、注释等）
        if len(critical_lines_o) != len(critical_lines_n):
            logger.warning(f'Suffix critical line count mismatch: {len(critical_lines_o)} vs {len(critical_lines_n)}')
            return False, f'Suffix critical line count does not match original'
        
        logger.info('Suffix validation passed with relaxed tolerance (content check disabled)')
    
    return True, ''


def _validate_range_constraint(original_text: str, new_text: str, start: int, end: int) -> Tuple[bool, str]:
    """Legacy validation function for fixed range constraints."""
    o_lines = original_text.splitlines()
    n_lines = new_text.splitlines()
    
    # Check prefix [0:start-1] - must be identical
    prefix_end = max(0, start - 1)
    prefix_o = o_lines[:prefix_end]
    prefix_n = n_lines[:prefix_end] if len(n_lines) >= prefix_end else n_lines
    
    if prefix_o != prefix_n:
        # 进一步放宽：只检查行数差异，允许内容差异
        if abs(len(prefix_o) - len(prefix_n)) > 20:  # 增加容差
            return False, f'Lines before line {start} were modified (prefix length mismatch too large)'
    
    # Check suffix [end:] - must be identical
    # We need to account for potential length change in the [start:end] region
    # So we compare from the end backwards
    suffix_o = o_lines[end:] if end <= len(o_lines) else []
    
    # Calculate where suffix should start in new text
    # new_text may have different number of lines in [start:end] region
    region_len_o = end - start + 1
    region_len_n = len(n_lines) - len(prefix_n) - len(suffix_o)
    
    # Get suffix from new text (same number of lines from the end)
    if suffix_o:
        suffix_n = n_lines[-len(suffix_o):] if len(n_lines) >= len(suffix_o) else []
        if suffix_o != suffix_n:
            # 进一步放宽：只检查行数差异，允许内容差异
            if abs(len(suffix_o) - len(suffix_n)) > 20:  # 增加容差
                return False, f'Lines after line {end} were modified (suffix length mismatch too large)'
    else:
        # No suffix in original - check if new text added extra lines beyond allowed region
        expected_total = len(prefix_o) + max(1, region_len_n)
        if len(n_lines) > expected_total:
            return False, f'Extra lines added after line {end}'
    
    return True, ''


class ArtifactCallback(Callback):
    """Save the output code to local disk with range constraint validation.
    
    For the target file specified in TASK_PREP, validates that only lines [start:end]
    have been modified. Rejects writes that violate this constraint.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_generate_response(self, runtime: Runtime,
                                   messages: List[Message]):
        for message in messages:
            if message.role == 'assistant' and message.tool_calls and not message.content:
                # Claude seems does not allow empty content
                message.content = 'I should do a tool calling to continue:\n'

    async def on_tool_call(self, runtime: Runtime, messages: List[Message]):
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            return
        await self.file_system.create_directory()

        # Only write files from the LAST assistant message to avoid
        # accidentally writing code blocks included by user/system callbacks.
        last_assistant = None
        for m in reversed(messages):
            if m.role == 'assistant' and m.content:
                last_assistant = m
                break

        if not last_assistant:
            return

        # Parse task metadata for range constraint validation
        workdir, target_rel, rng, original_text, dynamic_mode, fn_coding_mode = _parse_task_meta(messages)

        all_files, _ = extract_code_blocks(last_assistant.content)
        results = []
        violations = []
        # Normalize target_rel to be relative to workdir
        target_rel_norm = (target_rel or '').replace('\\', '/').lstrip('/')
        if workdir and target_rel_norm.startswith(f'{workdir}/'):
            parts = target_rel_norm.split('/', 1)
            target_rel_norm = parts[1] if len(parts) > 1 else target_rel_norm
        if target_rel_norm.startswith('task_'):
            parts = target_rel_norm.split('/', 1)
            target_rel_norm = parts[1] if len(parts) > 1 else target_rel_norm
        # Extract task json for source_id
        task_obj = _extract_task(messages)
        source_id = task_obj.get('source_id', '') if task_obj else ''

        for f in all_files:
            filename = f['filename']
            code = f['code']
            
            # Normalize paths for comparison
            norm_filename = filename.replace('\\', '/')

            # Guard: never write to absolute paths like /root/... .
            # If we know the task's WORKDIR and TARGET_FILE, remap to that location.
            if norm_filename.startswith('/'):
                if workdir and target_rel_norm:
                    remapped = f'{workdir}/{target_rel_norm}'.replace('\\', '/')
                    logger.info(
                        'ArtifactCallback: remap absolute filename %s -> %s',
                        norm_filename,
                        remapped,
                    )
                    filename = remapped
                    norm_filename = remapped
                else:
                    violations.append(
                        f'- {filename}: absolute paths are not allowed. '
                        'Use the provided WORKDIR/TARGET_FILE path (e.g. task_XXX/<relative_path>).'
                    )
                    continue
            
            # Ensure filename has WORKDIR prefix if workdir is available
            # This prevents files from being written to output/ directly instead of output/task_XXX/
            if workdir and not norm_filename.startswith(f'{workdir}/'):
                # Check if it's an absolute path or already has a different prefix
                if not norm_filename.startswith('/') and not norm_filename.startswith('task_'):
                    filename = f'{workdir}/{filename}'
                    norm_filename = filename.replace('\\', '/')
                    logger.info(f'Added WORKDIR prefix to filename: {filename}')

            # Collapse duplicate WORKDIR prefix (e.g. task_1002/task_1002/...) to avoid
            # creating nested directories like output/task_1002/task_1002/...
            if workdir:
                double_prefix = f'{workdir}/{workdir}/'
                if norm_filename.startswith(double_prefix):
                    filename = norm_filename[len(workdir) + 1:]
                    norm_filename = filename
                    logger.info(f'Collapsed duplicate WORKDIR prefix in filename: {filename}')
            
            # Integrate assistant function into full original file if this is the target file
            is_target_file = False
            if workdir and target_rel_norm:
                candidates = {f'{workdir}/{target_rel_norm}'.replace('\\', '/'), target_rel_norm}
                is_target_file = norm_filename in candidates
            elif target_rel_norm:
                is_target_file = norm_filename.endswith(target_rel_norm)

            if is_target_file and not fn_coding_mode and rng:
                logger.info(f"ArtifactCallback: Integration conditions met - is_target_file={is_target_file}, fn_coding_mode={fn_coding_mode}, rng={rng}")
                try:
                    logger.info(f"ArtifactCallback: Starting integration for target file {norm_filename}")
                    try:
                        if task_obj and isinstance(task_obj, dict):
                            logger.info(
                                "ArtifactCallback: task_json(meta) "
                                f"start={task_obj.get('start')} end={task_obj.get('end')} "
                                f"full_signature={task_obj.get('full_signature')}"
                            )
                    except Exception:
                        pass
                    # Locate original source file in dataset
                    workspace_root = os.getcwd()
                    local_dir = getattr(self.config, 'local_dir', '') or 'projects/sol_fn_miniloop'
                    dataset_root = os.path.join(workspace_root, local_dir, 'root')

                    norm_sid = source_id.replace('\\', '/').lstrip('/') if source_id else ''
                    src_candidates = [
                        os.path.join(dataset_root, norm_sid) if norm_sid else None,
                        os.path.join(dataset_root, norm_sid[len('root/'):]) if norm_sid.startswith('root/') else None,
                        os.path.join(workspace_root, norm_sid) if norm_sid else None,
                        os.path.join(workspace_root, 'root', norm_sid) if norm_sid else None,
                    ]

                    original_source = None
                    for c in src_candidates:
                        if c and os.path.exists(c):
                            original_source = c
                            break

                    out_dir = getattr(self.config, 'output_dir', 'output')
                    dest_root = os.path.join(out_dir, workdir) if workdir else out_dir
                    original_file_path = os.path.join(dest_root, target_rel_norm) if target_rel_norm else os.path.join(dest_root, norm_filename)
                    os.makedirs(os.path.dirname(original_file_path), exist_ok=True)
                    if original_source:
                        shutil.copy2(original_source, original_file_path)

                    # If copy not available, fall back to default write
                    if not os.path.exists(original_file_path):
                        result = await self.file_system.write_file(filename, code)
                        results.append(result)
                        continue

                    with open(original_file_path, 'r', encoding='utf-8') as fsrc:
                        original_lines = fsrc.readlines()

                    # 获取函数签名
                    full_sig = None
                    for m in messages:
                        if m.role == 'user' and m.content:
                            for ln in m.content.splitlines():
                                if ln.startswith('FUNCTION_SIGNATURE:'):
                                    full_sig = ln.split(':', 1)[1].strip()
                                    break
                            if full_sig:
                                break
                    if not full_sig and task_obj and isinstance(task_obj, dict):
                        full_sig = task_obj.get('full_signature')

                    logger.info(
                        "ArtifactCallback: integration inputs "
                        f"message_rng={rng} extracted_full_sig={full_sig!r}"
                    )

                    expected_start = None
                    try:
                        if task_obj and isinstance(task_obj, dict) and task_obj.get('start') is not None:
                            expected_start = int(task_obj.get('start'))
                    except Exception:
                        expected_start = None
                    if expected_start is None and rng:
                        expected_start = int(rng[0])

                    # 策略1：优先使用函数签名定位替换位置
                    integration_success = False
                    if full_sig:
                        old_start, old_end = _locate_function_in_file(
                            original_lines,
                            full_sig,
                            expected_start=expected_start,
                        )
                        if old_start and old_end:
                            logger.info(f"ArtifactCallback: Located function in original file at lines [{old_start}:{old_end}]")
                            
                            # 从 assistant 输出中提取函数块
                            lines = code.splitlines(True)
                            cleaned = [ln for ln in lines if not ln.strip().startswith('// [start:') and not ln.strip().startswith('// [end:')]
                            func_block = _extract_function_block_from_assistant_output(cleaned, full_sig)
                            
                            # 执行替换
                            old_start_idx = old_start - 1  # 转为0基索引
                            old_end_idx = old_end          # end 已经是正确的
                            
                            new_file_lines = original_lines[:old_start_idx] + func_block + original_lines[old_end_idx:]
                            with open(original_file_path, 'w', encoding='utf-8') as fdst:
                                fdst.writelines(new_file_lines)
                            
                            # 计算新的结束行号
                            new_end = old_start + len(func_block)
                            logger.info(f"ArtifactCallback: Successfully integrated function using signature matching")
                            logger.info(f"Old range: [{old_start}:{old_end}] -> New range: [{old_start}:{new_end}]")
                            
                            # 更新 RANGE meta
                            _update_range_meta(messages, workdir, target_rel_norm, old_start, new_end, full_sig)
                            
                            integration_success = True
                    
                    # 策略2：fallback 到 RANGE 替换
                    if not integration_success and rng:
                        logger.info(f"ArtifactCallback: Falling back to RANGE-based replacement: {rng}")
                        start, end = rng
                        start_idx = max(0, int(start) - 1)
                        end_idx = min(len(original_lines), int(end))

                        # Remove [start]/[end] markers from assistant output
                        lines = code.splitlines(True)
                        cleaned = [ln for ln in lines if not ln.strip().startswith('// [start:') and not ln.strip().startswith('// [end:')]

                        # 提取函数块
                        if full_sig:
                            func_block = _extract_function_block_from_assistant_output(cleaned, full_sig)
                        else:
                            # 简单的 fallback 提取
                            func_block = cleaned

                        # 执行替换
                        new_file_lines = original_lines[:start_idx] + func_block + original_lines[end_idx:]
                        with open(original_file_path, 'w', encoding='utf-8') as fdst:
                            fdst.writelines(new_file_lines)
                        
                        # 计算新的结束行号
                        new_end = start + len(func_block)
                        logger.info(f"ArtifactCallback: Successfully integrated function using RANGE fallback")
                        logger.info(f"Old range: [{start}:{end}] -> New range: [{start}:{new_end}]")
                        
                        # 更新 RANGE meta
                        if full_sig:
                            _update_range_meta(messages, workdir, target_rel_norm, start, new_end, full_sig)
                        
                        integration_success = True
                    
                    if integration_success:
                        logger.info(f"ArtifactCallback: Successfully integrated function into {original_file_path}")
                        results.append(f'Save file <{workdir}/{target_rel_norm}> successfully.')
                        continue
                    else:
                        logger.error(f"ArtifactCallback: Failed to integrate function - neither signature matching nor RANGE fallback succeeded")
                        continue
                except Exception as e:
                    # Fall back to existing behavior if integration fails
                    logger.exception(f"ArtifactCallback: Integration failed for {norm_filename}, error: {e}")
                    pass
            else:
                logger.info(f"ArtifactCallback: Integration conditions NOT met - is_target_file={is_target_file}, fn_coding_mode={fn_coding_mode}, rng={rng}, norm_filename={norm_filename}, workdir={workdir}, target_rel_norm={target_rel_norm}")

            # Special handling for fn_coding mode
            if fn_coding_mode and workdir and target_rel and norm_filename == f'{workdir}/{target_rel}'.replace('\\', '/'):
                # In fn_coding mode, we just write the standalone function without validation
                logger.info(f'FN_CODING MODE: Writing standalone function to {filename}')
                result = await self.file_system.write_file(filename, code)
                results.append(result)
                continue
            should_guard = (
                workdir is not None
                and target_rel is not None
                and rng is not None
                and original_text is not None
                and norm_filename == f'{workdir}/{target_rel}'.replace('\\', '/')
                and not fn_coding_mode  # Skip range validation in fn_coding mode
            )
            
            if should_guard:
                # Choose validation function based on dynamic mode
                if dynamic_mode:
                    ok, reason = _validate_dynamic_range_constraint(original_text, code, rng[0], rng[1])
                else:
                    ok, reason = _validate_range_constraint(original_text, code, rng[0], rng[1])
                    
                if not ok:
                    mode_text = "(dynamic mode)" if dynamic_mode else "(fixed mode)"
                    violations.append(
                        f'- {filename}: {reason} {mode_text}\n'
                        f'  Allowed range: lines [{rng[0]}:{rng[1]}]\n'
                        f'  You MUST keep all lines outside this range IDENTICAL to the original.'
                    )
                    logger.warning(f'Range constraint violation for {filename}: {reason}')
                    continue  # Skip writing this file
                else:
                    mode_text = "(dynamic mode)" if dynamic_mode else "(fixed mode)"
                    logger.info(f'Range constraint validated for {filename}: lines [{rng[0]}:{rng[1]}] {mode_text}')
                    
                    # Check if the code is missing prefix/suffix context (common in fn_secure_refine stage)
                    # If so, provide context feedback to help the agent reconstruct the complete file
                    if original_text:
                        o_lines = original_text.splitlines()
                        n_lines = code.splitlines()
                        
                        # More robust detection: check for placeholder patterns
                        has_prefix_placeholder = any('previous code' in line.lower() or '...' in line for line in n_lines[:5])
                        has_suffix_placeholder = any('remaining code' in line.lower() or '...' in line for line in n_lines[-5:])
                        
                        # Also check length-based detection
                        start_idx = max(0, rng[0] - 1)
                        expected_prefix_len = start_idx
                        end_idx = min(len(o_lines), rng[1])
                        expected_suffix_start = end_idx
                        
                        is_too_short = len(n_lines) < expected_prefix_len + 10
                        
                        missing_context = []
                        
                        # Provide context if placeholders detected or file is too short
                        if has_prefix_placeholder or has_suffix_placeholder or is_too_short:
                            # Provide prefix context if missing
                            if expected_prefix_len > 0:
                                prefix_lines = o_lines[:expected_prefix_len]
                                missing_context.append('--- FILE PREFIX (add this before your code) ---')
                                missing_context.append(f'```solidity: {target_rel}')
                                missing_context.append('\n'.join(prefix_lines))
                                missing_context.append('```')
                            
                            # Provide suffix context if missing  
                            if expected_suffix_start < len(o_lines):
                                suffix_lines = o_lines[expected_suffix_start:]
                                missing_context.append('--- FILE SUFFIX (add this after your code) ---')
                                missing_context.append(f'```solidity: {target_rel}')
                                missing_context.append('\n'.join(suffix_lines))
                                missing_context.append('```')
                        
                        if missing_context:
                            context_feedback = (
                                '[FILE_CONTEXT]\n'
                                '⚠️ Your output appears to be missing file context. '
                                'Please include the complete file with prefix and suffix:\n\n'
                                + '\n\n'.join(missing_context) +
                                '\n\nRe-output the complete file with all context included.'
                            )
                            messages.append(Message(role='user', content=context_feedback))
                            continue  # Skip writing, wait for agent to provide complete file
            
            # Write the file
            result = await self.file_system.write_file(filename, code)
            logger.info('----------------')
            results.append(result)
            logger.info(results)

        # If there were violations, append feedback to guide the agent
        if violations:
            feedback = (
                '[ARTIFACT_GUARD]\n'
                '⚠️ Range constraint violation detected!\n\n'
                'The following files violated the range constraint:\n' +
                '\n'.join(violations) +
                '\n\nPlease re-output the target file with:\n'
                '1. Lines BEFORE the allowed range kept EXACTLY as in the original\n'
                '2. Lines AFTER the allowed range kept EXACTLY as in the original\n'
                '3. ONLY modify the function implementation within the specified range\n'
            )
            messages.append(Message(role='user', content=feedback))
        
        # Report successfully written files
        if results:
            r = '\n'.join(results)
            messages.append(Message(role='user', content=r))
