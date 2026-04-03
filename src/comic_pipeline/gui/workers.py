from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal


@dataclass(slots=True)
class WorkerResult:
    label: str
    payload: Any


class WorkerSignals(QObject):
    started = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str, str)


class FunctionWorker(QRunnable):
    def __init__(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.label = label
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        self.signals.started.emit(self.label)
        try:
            payload = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.failed.emit(self.label, traceback.format_exc())
            return
        self.signals.succeeded.emit(WorkerResult(label=self.label, payload=payload))
