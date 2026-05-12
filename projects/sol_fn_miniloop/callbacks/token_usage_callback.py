import json
import logging
import os
import re
import sys
import atexit
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback
from ms_agent.llm.utils import Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


_USAGE_RE = re.compile(
    r"(?:\[[A-Z]+:ms_agent\]\s*)?\[(?P<stage>[^\]]+)\]\s+\[usage\]\s+prompt_tokens:\s*(?P<prompt>\d+),\s*completion_tokens:\s*(?P<completion>\d+)",
)


@dataclass
class _Agg:
    total_prompt: int = 0
    total_completion: int = 0
    by_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    raw_calls: List[Dict[str, int]] = field(default_factory=list)

    def add(self, stage: str, prompt: int, completion: int) -> None:
        self.total_prompt += int(prompt)
        self.total_completion += int(completion)
        st = self.by_stage.setdefault(stage, {"prompt": 0, "completion": 0, "calls": 0})
        st["prompt"] += int(prompt)
        st["completion"] += int(completion)
        st["calls"] += 1
        self.raw_calls.append({"stage": stage, "prompt": int(prompt), "completion": int(completion)})

    def to_dict(self, *, handler_installed: bool, stream_interceptor_installed: bool) -> Dict:
        return {
            "total": {
                "prompt": self.total_prompt,
                "completion": self.total_completion,
                "tokens": self.total_prompt + self.total_completion,
            },
            "by_stage": self.by_stage,
            "raw_calls": self.raw_calls,
            "debug": {
                "handler_installed": handler_installed,
                "stream_interceptor_installed": stream_interceptor_installed,
                "calls": len(self.raw_calls),
            },
        }


class _UsageLogHandler(logging.Handler):
    def __init__(self, cb: "TokenUsageCallback"):
        super().__init__()
        self._cb = cb

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        self._cb._consume_text(msg)


class _StreamTee:
    def __init__(self, cb: "TokenUsageCallback", wrapped):
        self._cb = cb
        self._wrapped = wrapped
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._cb._consume_text(s)
        try:
            return self._wrapped.write(s)
        except Exception:
            return 0

    def flush(self):
        try:
            return self._wrapped.flush()
        except Exception:
            return None

    def isatty(self):
        try:
            return self._wrapped.isatty()
        except Exception:
            return False


class TokenUsageCallback(Callback):
    """Collects and flushes token usage stats per stage.

    This callback listens for lines that contain:
      [<stage>] [usage] prompt_tokens: <n>, completion_tokens: <n>

    It supports both python logging and stdout/stderr output.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._lock = Lock()
        self._agg = _Agg()
        self._handler_installed = False
        self._stream_interceptor_installed = False
        self._orig_stdout = None
        self._orig_stderr = None
        self._workdir: Optional[str] = None
        self._last_runtime_tag: Optional[str] = None
        self._atexit_registered: bool = False

    def _consume_text(self, text: str) -> None:
        if not text:
            return

        with self._lock:
            lines = text.splitlines() or [text]
            for line in lines:
                m = _USAGE_RE.search(line)
                if not m:
                    continue
                stage = m.group("stage")
                prompt = int(m.group("prompt"))
                completion = int(m.group("completion"))
                self._agg.add(stage, prompt, completion)
            # If we captured at least one usage line and we know where to write, flush immediately.
            if self._agg.raw_calls and self._workdir and self._last_runtime_tag:
                try:
                    self._flush_if_ready()
                except Exception:
                    pass

    def _parse_workdir(self, messages: List[Message]) -> Optional[str]:
        # Prefer explicit WORKDIR: meta
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            for line in m.content.splitlines():
                if line.startswith('WORKDIR:'):
                    return line.split(':', 1)[1].strip()
        # Fallback: look for task_<id>
        for m in reversed(messages):
            if m.role != 'user' or not m.content:
                continue
            if '"task_id"' in m.content:
                mm = re.search(r'"task_id"\s*:\s*(\d+)', m.content)
                if mm:
                    return f"task_{mm.group(1)}"
        return None

    def _ensure_logging_handler(self) -> None:
        if self._handler_installed:
            return
        try:
            handler = _UsageLogHandler(self)
            handler.setLevel(logging.INFO)

            root = logging.getLogger()
            root.addHandler(handler)

            # Some libraries create non-propagating loggers (propagate=False) with their own handlers.
            # Attach to all known loggers to ensure we see the usage line.
            try:
                for _name, _lg in logging.Logger.manager.loggerDict.items():
                    if isinstance(_lg, logging.Logger):
                        _lg.addHandler(handler)
            except Exception:
                pass

            # Also attach to the common project logger explicitly.
            try:
                logging.getLogger('ms_agent').addHandler(handler)
            except Exception:
                pass
            self._handler_installed = True
        except Exception:
            return

    def _flush_if_ready(self) -> None:
        if not self._workdir:
            return
        tag = self._last_runtime_tag or 'stage'
        out_root = getattr(self.config, 'output_dir', 'output')
        out_dir = os.path.join(out_root, self._workdir)
        os.makedirs(out_dir, exist_ok=True)
        out_fp = os.path.join(out_dir, f"token_usage_{tag}.json")
        with open(out_fp, 'w', encoding='utf-8') as f:
            json.dump(
                self._agg.to_dict(
                    handler_installed=self._handler_installed,
                    stream_interceptor_installed=self._stream_interceptor_installed,
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _ensure_stream_interceptor(self) -> None:
        if self._stream_interceptor_installed:
            return
        try:
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _StreamTee(self, sys.stdout)
            sys.stderr = _StreamTee(self, sys.stderr)
            self._stream_interceptor_installed = True
        except Exception:
            return

    def _flush(self, runtime: Runtime) -> None:
        workdir = self._workdir
        if not workdir:
            return

        out_root = getattr(self.config, 'output_dir', 'output')
        out_dir = os.path.join(out_root, workdir)
        os.makedirs(out_dir, exist_ok=True)

        tag = getattr(runtime, 'tag', None) or self._last_runtime_tag or 'stage'
        out_fp = os.path.join(out_dir, f"token_usage_{tag}.json")
        with open(out_fp, 'w', encoding='utf-8') as f:
            json.dump(
                self._agg.to_dict(
                    handler_installed=self._handler_installed,
                    stream_interceptor_installed=self._stream_interceptor_installed,
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _flush_final_at_exit(self) -> None:
        # atexit runs after the main task loop, so usage lines printed late are captured.
        if not self._workdir:
            return
        out_root = getattr(self.config, 'output_dir', 'output')
        out_dir = os.path.join(out_root, self._workdir)
        os.makedirs(out_dir, exist_ok=True)
        tag = self._last_runtime_tag or 'stage'
        out_fp = os.path.join(out_dir, f"token_usage_{tag}.json")
        try:
            with open(out_fp, 'w', encoding='utf-8') as f:
                json.dump(
                    self._agg.to_dict(
                        handler_installed=self._handler_installed,
                        stream_interceptor_installed=self._stream_interceptor_installed,
                    ),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            return

    def _ensure_atexit_flush_registered(self) -> None:
        if self._atexit_registered:
            return
        try:
            atexit.register(self._flush_final_at_exit)
            self._atexit_registered = True
        except Exception:
            self._atexit_registered = True

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        self._agg = _Agg()
        self._workdir = self._parse_workdir(messages)
        self._last_runtime_tag = getattr(runtime, 'tag', None)
        self._ensure_logging_handler()
        self._ensure_stream_interceptor()
        self._ensure_atexit_flush_registered()

    async def on_generate_response(self, runtime: Runtime, messages: List[Message]):
        # update workdir lazily in case meta arrives later
        if not self._workdir:
            self._workdir = self._parse_workdir(messages)
        if not self._last_runtime_tag:
            self._last_runtime_tag = getattr(runtime, 'tag', None)

    async def on_task_end(self, runtime: Runtime, messages: List[Message]):
        if not self._workdir:
            self._workdir = self._parse_workdir(messages)
        try:
            self._flush(runtime)
        except Exception as e:
            logger.warning(f"TokenUsageCallback: flush failed: {e}")
