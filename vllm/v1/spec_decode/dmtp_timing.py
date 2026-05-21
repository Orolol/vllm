# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny CUDA-event-based timer for Dmtp proposers.

Enabled when ``DMTP_TIME=1`` is set in the env. Use as::

    from vllm.v1.spec_decode.dmtp_timing import time_region, report

    with time_region("greedy_sample"):
        ...

Timings are accumulated across all calls and dumped at process exit
(via atexit). Each region records a pair of CUDA events at enter/exit;
they are joined and accumulated lazily (synchronized) by ``flush`` or
``report``. Zero overhead when the env var is unset.
"""

from __future__ import annotations

import atexit
import os
import threading
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("DMTP_TIME", "") == "1"
_lock = threading.Lock()
# name -> list of (start_event, end_event)
_pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
# name -> [total_ms, count]
_acc: dict[str, list[float]] = {}


def _flush() -> None:
    if not _ENABLED:
        return
    with _lock:
        pending = _pending
        if not pending:
            return
        torch.cuda.synchronize()
        for name, events in pending.items():
            tot, cnt = _acc.get(name, [0.0, 0])
            for s, e in events:
                tot += s.elapsed_time(e)
                cnt += 1
            _acc[name] = [tot, cnt]
        _pending.clear()


@contextmanager
def time_region(name: str):
    if not _ENABLED:
        yield
        return
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    try:
        yield
    finally:
        e.record()
        with _lock:
            _pending.setdefault(name, []).append((s, e))


def add_cpu_time(name: str, ms: float) -> None:
    """Add a CPU-side measurement (no CUDA event involved)."""
    if not _ENABLED:
        return
    with _lock:
        tot, cnt = _acc.get(name, [0.0, 0])
        _acc[name] = [tot + ms, cnt + 1]


def report() -> str:
    if not _ENABLED:
        return ""
    _flush()
    lines = ["", "=== dmtp_timing report ==="]
    rows = sorted(_acc.items(), key=lambda x: -x[1][0])
    for name, (ms, n) in rows:
        mean = ms / max(n, 1)
        lines.append(
            f"  {name:32s}  total={ms:9.2f} ms  count={n:6d}  mean={mean:7.4f} ms"
        )
    return "\n".join(lines)


@atexit.register
def _dump_on_exit() -> None:
    if not _ENABLED:
        return
    msg = report()
    if msg:
        print(msg, flush=True)
