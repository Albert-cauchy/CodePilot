from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class TrajectoryLogger:
    """Generic JSONL event logger.

    Agent-specific event types are intentionally not defined here; you can add
    them when you implement AgentCore.
    """

    def __init__(self, path: str | Path, **context) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.lock = Lock()

    def log(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **self.context,
            **payload,
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
