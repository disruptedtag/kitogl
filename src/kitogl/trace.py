"""A hook for the application's timeline: KitoGL reports a few milestones (context ready, first picture drawn).

Nothing is written unless the application installs its own functions with `set_tracer`.
"""

from __future__ import annotations

from typing import Callable, Optional

_trace: Optional[Callable[[str], None]] = None
_trace_memory: Optional[Callable[[str], None]] = None


def set_tracer(trace_function: Optional[Callable[[str], None]] = None,
               memory_function: Optional[Callable[[str], None]] = None) -> None:
    """`trace_function(message)` gets each milestone; `memory_function(label)` the ones worth a memory reading."""
    global _trace, _trace_memory
    _trace, _trace_memory = trace_function, memory_function


def trace(message: str) -> None:
    if _trace is not None:
        _trace(message)


def trace_memory(label: str) -> None:
    if _trace_memory is not None:
        _trace_memory(label)
