import os
import sys
import json
import shlex
import pathlib
import subprocess

# 配置项（按需修改）
PYTHON = sys.executable
CLI = "ms_agent/cli/cli.py"  # 你的现有 CLI 路径
CONFIG = "projects/sol_fn_miniloop/"  # 等同于命令行 --config
TRUST_REMOTE_CODE = "true"
TASKS_PATH = "projects/sol_fn_miniloop/example_task.json"  # 任务集合文件路径


def iter_tasks(path: str):
    """
    迭代任务：自动兼容以下三种格式：
    1) JSON 数组：[ {...}, {...}, ... ]
    2) 单个 JSON 对象：{ ... }
    3) JSONL：每行一个 JSON 对象
    """
    text = pathlib.Path(path).read_text(encoding="utf-8").strip()

    # 尝试解析为 JSON（数组或对象）
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                yield json.dumps(obj, ensure_ascii=False)
            return
        elif isinstance(data, dict):
            yield json.dumps(data, ensure_ascii=False)
            return
    except Exception:
        pass

    # 回退：按 JSONL（逐行）处理
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        yield line


def run_one(task_json_str: str) -> int:
    """
    调用现有 CLI 跑单个任务。
    """
    cmd = [
        PYTHON, CLI, "run",
        "--config", CONFIG,
        "--query", task_json_str,
        "--trust_remote_code", TRUST_REMOTE_CODE,
    ]

    # 打印关键信息（避免输出过长）
    shown = " ".join(shlex.quote(c) for c in cmd[:6]) + " ..."
    print(f"[run] {shown}")

    # 确保 PYTHONPATH 生效，与命令行一致
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")

    proc = subprocess.run(cmd, env=env, text=True)
    return proc.returncode


def main():
    # 默认 PYTHONPATH=.
    os.environ.setdefault("PYTHONPATH", ".")

    ok = 0
    fail = 0

    for task_str in iter_tasks(TASKS_PATH):
        # 尝试从任务中取 task_id 以便标识
        try:
            obj = json.loads(task_str)
            tid = obj.get("task_id", "unknown")
        except Exception:
            tid = "unknown"
        print(f"[task] task_id={tid}")

        rc = run_one(task_str)
        if rc == 0:
            ok += 1
            print(f"[task] task_id={tid} OK")
        else:
            fail += 1
            print(f"[task] task_id={tid} FAIL (exit={rc})")

    print(f"[batch] done. ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()


