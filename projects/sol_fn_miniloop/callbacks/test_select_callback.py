import os
import re
import json
from typing import List, Optional, Tuple

from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.tools.filesystem_tool import FileSystemTool
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


def _extract_task(messages: List[Message]) -> Optional[dict]:
    for m in reversed(messages):
        if m.role != 'user' or not m.content:
            continue
        content = m.content.strip()
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and ('source_id' in obj or 'task_id' in obj):
                logger.info("Found valid task JSON")  # 添加这行
                # Filter out canonical_solution to prevent ground truth leakage
                obj.pop('canonical_solution', None)
                return obj
        except Exception as e:
            logger.info(f"Failed to parse JSON: {e}")  # 添加这行
            pass
        start = content.find('```json')
        if start != -1:
            end = content.find('```', start + 7)
            if end != -1:
                snippet = content[start + 7:end].strip()
                logger.info(f"Found JSON snippet: {snippet}")  # 添加这行
                try:
                    obj = json.loads(snippet)
                    logger.info(f"Parsed JSON snippet: {obj}")  # 添加这行
                    if isinstance(obj, dict) and ('source_id' in obj or 'task_id' in obj):
                        logger.info("Found valid task JSON from snippet")  # 添加这行
                        # Filter out canonical_solution to prevent ground truth leakage
                        obj.pop('canonical_solution', None)
                        return obj
                except Exception as e:
                    logger.info(f"Failed to parse JSON snippet: {e}")  # 添加这行
                    pass
    logger.info("No task JSON found in any message")  # 添加这行
    return None


def _parse_task_prep_meta(messages: List[Message]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse workdir, target_rel, and full_signature from either TASK_PREP messages or new optimized format"""
    workdir = target_rel = full_sig = None
    
    # 首先尝试从新的优化格式中提取信息
    for m in messages:
        if m.role != 'user' or not m.content:
            continue
        
        if 'WORKDIR:' in m.content and 'TARGET_FILE:' in m.content:
            for line in m.content.splitlines():
                if line.startswith('TARGET_FILE:'):
                    target_rel = line.split(':', 1)[1].strip()
                    logger.info(f"Found TARGET_FILE from optimized format: {target_rel}")
                elif line.startswith('FUNCTION_SIGNATURE:'):
                    full_sig = line.split(':', 1)[1].strip()
                    logger.info(f"Found FULL_SIGNATURE from optimized format: {full_sig}")
                elif line.startswith('WORKDIR:'):
                    workdir = line.split(':', 1)[1].strip()
                    logger.info(f"Found WORKDIR from optimized format: {workdir}")
            
            # 从task中提取workdir
            task = _extract_task(messages)
            if task and 'task_id' in task:
                workdir = f"task_{task['task_id']}"
                logger.info(f"Extracted WORKDIR from task: {workdir}")
            break
    
    # 如果新格式没有找到，回退到旧的TASK_PREP格式
    if not (workdir and target_rel):
        logger.info("Optimized format not found, trying legacy TASK_PREP format")
        for m in messages:
            if m.role != 'user' or not m.content:
                continue
            if '[TASK_PREP]' not in m.content:
                continue
            logger.info(f"Found TASK_PREP message: {m.content[:200]}...")
            for line in m.content.splitlines():
                if line.startswith('WORKDIR:'):
                    workdir = line.split(':', 1)[1].strip()
                elif line.startswith('TARGET_FILE:'):
                    target_rel = line.split(':', 1)[1].strip()
                elif line.startswith('FUNCTION_SIGNATURE:'):
                    full_sig = line.split(':', 1)[1].strip()
            break
    
    logger.info(f"Returning: workdir={workdir}, target_rel={target_rel}, full_sig={full_sig}")
    return workdir, target_rel, full_sig


def _parse_func_name(full_signature: Optional[str]) -> Optional[str]:
    if not full_signature:
        return None
    m = re.search(r"function\s+([A-Za-z0-9_]+)\s*\(", full_signature)
    if m:
        return m.group(1)
    return None


def _find_test_file(dest_root: str, mapping_val: str) -> Optional[str]:
    try:
        base = os.path.basename(mapping_val.replace('\\', '/'))
        # First try basename search under copied workspace
        for p, _, files in os.walk(os.path.join(dest_root, 'test')):
            for f in files:
                if f == base and f.endswith('.sol'):
                    return os.path.join(p, f)
        # Fallback: use subpath after '/test/' if available
        v = mapping_val.replace('\\', '/').lstrip('/')
        idx = v.find('/test/')
        if idx != -1:
            sub = v[idx + 1:]  # include 'test/...'
            cand = os.path.join(dest_root, sub)
            if os.path.exists(cand):
                return cand
    except Exception:
        return None
    return None


def _collect_selected_tests(sol_text: str, func_name: str, module_guess: Optional[str]) -> List[str]:
    # Build simple patterns to detect usage inside a test function body
    pats = [f"{func_name}(", f".{func_name}("]
    if module_guess:
        pats.append(f"{module_guess}.{func_name}(")

    # Locate test function declarations and extract their bodies via brace counting
    selected = []
    i = 0
    n = len(sol_text)
    while i < n:
        m = re.search(r"\bfunction\s+(test[A-Za-z0-9_]*)\s*\([^)]*\)\s*[^{;]*\{", sol_text[i:])
        if not m:
            break
        fn_name = m.group(1)
        start = i + m.start()
        body_start = i + m.end()  # position after '{'
        brace = 1
        j = body_start
        while j < n and brace > 0:
            if sol_text[j] == '{':
                brace += 1
            elif sol_text[j] == '}':
                brace -= 1
            j += 1
        body = sol_text[body_start:j-1] if j-1 >= body_start else ''
        body_lc = body
        if any(p in body_lc for p in pats):
            selected.append(fn_name)
        i = j
    return selected


class TestSelectCallback(Callback):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.file_system = FileSystemTool(config)
        self.output_dir = getattr(config, 'output_dir', 'output')
        # where to write selection for forge_callback to consume
        self.selection_filename = 'public_tests.json'

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        logger.info(messages)
        if messages[-1].tool_calls or messages[-1].role == 'tool':
            logger.info("TestSelectCallback triggered-2")
            return

        try:
            logger.info("TestSelectCallback triggered-3")
            task = _extract_task(messages)
            logger.info(f"Extracted task: {task}")  # 添加这行
            workdir, target_rel, full_sig_from_prep = _parse_task_prep_meta(messages)
            logger.info(f"Parsed meta: workdir={workdir}, target_rel={target_rel}")  # 添加这行
            if not task or not workdir or not target_rel:
                logger.info(f"Early return: task={bool(task)}, workdir={bool(workdir)}, target_rel={bool(target_rel)}")  # 添加这行
                return

            source_id = str(task.get('source_id', '')).strip()
            logger.info(f"Source ID: {source_id}")  # 添加这行
            full_signature = task.get('full_signature') or full_sig_from_prep or ''
            func_name = _parse_func_name(full_signature)
            logger.info(f"Function name: {func_name}")  # 添加这行
            module_guess = os.path.splitext(os.path.basename(target_rel.replace('\\', '/')))[0] if target_rel else None
            logger.info(f"Module guess: {module_guess}")  # 添加这行

            # Skip if public overlay already exists
            public_dir = os.path.join(self.output_dir, workdir, 'test', 'public')
            logger.info(f"Checking public overlay dir: {public_dir}")  # 添加这行
            try:
                if os.path.isdir(public_dir):
                    for _, _, files in os.walk(public_dir):
                        if any(f.endswith('.sol') for f in files):
                            logger.info("Public overlay exists, skipping")  # 添加这行
                            return
            except Exception:
                pass
            logger.info("No public overlay found, continuing")  # 添加这行

            # Load key.json mapping
            key_path = os.path.join(os.getcwd(), 'projects', 'sol_fn_miniloop', 'key.json')
            logger.info(f"Looking for key.json at: {key_path}")  # 添加这行
            if not os.path.exists(key_path):
                logger.info("key.json not found, skipping")  # 添加这行
                return
            with open(key_path, 'r', encoding='utf-8') as f:
                mp = json.load(f)
            logger.info(f"Looking for source_id: '{source_id}'")  # 添加这行
            if source_id not in mp:
                logger.info(f"source_id {source_id} not in key.json, attempting fuzzy match")
                # 通用模糊匹配
                source_filename = os.path.basename(source_id)
                matching_keys = [k for k in mp.keys() if source_filename in k]
                if matching_keys:
                    logger.info(f"Using fuzzy match: {matching_keys[0]}")
                    mapping_val = mp[matching_keys[0]]  # 实际使用匹配结果
                else:
                    logger.info(f"No matches found for {source_filename}, skipping")
                    return
            mapping_val = mp[source_id]
            logger.info(f"Mapping value: {mapping_val}")  # 添加这行

            dest_root = os.path.join(self.output_dir, workdir)
            logger.info(f"Dest root: {dest_root}")  # 添加这行
            test_path = _find_test_file(dest_root, mapping_val)
            logger.info(f"Found test path: {test_path}")  # 添加这行
            if not test_path or not os.path.exists(test_path):
                logger.info("Test file not found or doesn't exist, skipping")  # 添加这行
                return

            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    sol_text = f.read()
            except Exception:
                return

            if not func_name:
                return

            selected_tests = _collect_selected_tests(sol_text, func_name, module_guess)
            if not selected_tests:
                # Fallback: select first test to ensure at least one driver exists
                m = re.search(r"\bfunction\s+(test[A-Za-z0-9_]*)\s*\(", sol_text)
                if m:
                    selected_tests = [m.group(1)]

            sel = {
                'test_file': os.path.relpath(test_path, start=dest_root).replace('\\', '/'),
                'tests': selected_tests
            }
            with open(os.path.join(dest_root, self.selection_filename), 'w', encoding='utf-8') as f:
                json.dump(sel, f, ensure_ascii=False, indent=2)

            # Append a brief hint message for traceability
            msg_lines = [
                '[PUBLIC_TESTS]',
                f"WORKDIR: {workdir}",
                f"TARGET_FILE: {target_rel}",
                f"TEST_FILE: {sel['test_file']}",
                'TESTS:'
            ] + [f"- {name}" for name in selected_tests]
            messages.append(Message(role='user', content='\n'.join(msg_lines)))
            
            # 确保有完整的.sol文件（新增功能）
            # 使用源码文件路径，不是测试文件路径
            source_file_path = target_rel  # target_rel是从_parse_task_prep_meta获取的源码文件
            logger.info(f'TestSelectCallback: target_rel = {target_rel}')
            logger.info(f'TestSelectCallback: source_file_path = {source_file_path}')
            # 确保路径是相对于workdir的（不包含task_前缀）
            if source_file_path.startswith('task_'):
                # 如果路径已经包含task_前缀，提取相对路径
                # 例如：task_384/src/utils/LibBytes.sol -> src/utils/LibBytes.sol
                parts = source_file_path.split('/', 1)
                if len(parts) > 1:
                    final_path = parts[1]  # 取task_之后的部分
                else:
                    final_path = source_file_path
                logger.info(f'TestSelectCallback: Extracted relative path (removed task_ prefix): {final_path}')
            else:
                # 否则直接使用
                final_path = source_file_path
                logger.info(f'TestSelectCallback: Using relative path: {final_path}')
            self._ensure_complete_file_with_function(messages, workdir, final_path, full_signature, func_name, module_guess)
        except Exception:
            logger.warning('TestSelectCallback: selection step skipped due to error')

    def _ensure_complete_file_with_function(self, messages: List[Message], workdir: str, target_rel: str, full_signature: str, func_name: str, module_guess: Optional[str]):
        """确保目标 .sol 文件存在；若缺失则拷贝原始文件。方案1下不在这里做函数替换写回。"""
        if not workdir or not target_rel or not full_signature:
            logger.info('Skipping file replacement in TestSelectCallback - missing metadata')
            return
        
        try:
            task = _extract_task(messages)
            if not task:
                logger.info('TestSelectCallback._ensure_complete_file: No task found in messages')
                return
                
            source_id = task.get('source_id', '')
            if not source_id:
                logger.info('TestSelectCallback._ensure_complete_file: No source_id found in task')
                return
                
            logger.info(f'TestSelectCallback._ensure_complete_file: Looking for source_id: {source_id}')
                
            workspace_root = os.getcwd()
            local_dir = 'projects/sol_fn_miniloop'
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
                    logger.info(f'TestSelectCallback._ensure_complete_file: Found original source: {c}')
                    break
            
            if not original_source:
                logger.info(f'TestSelectCallback._ensure_complete_file: Could not find original file for source_id: {source_id}')
                return

            dest_root = os.path.join(self.output_dir, workdir)
            target_path = os.path.join(dest_root, target_rel)
            if os.path.exists(target_path):
                return

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            import shutil
            shutil.copy2(original_source, target_path)
            logger.info(f'TestSelectCallback._ensure_complete_file: Copied missing original file from {original_source} to {target_path}')
            
        except Exception as e:
            logger.warning(f'TestSelectCallback._ensure_complete_file: Failed to ensure complete file: {e}')
