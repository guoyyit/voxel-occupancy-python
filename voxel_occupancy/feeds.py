"""Local observation feeds: an in-memory queue and a loopback replay.

These replace any real sensor/driver: observations come from a thread-safe
in-process pipe or from a deterministic scripted replay. No external devices
or services are contacted.
"""
from __future__ import annotations

import queue
from collections.abc import Iterable, Iterator

from .types import Observation


class MemoryFeed:
    """Thread-safe in-memory FIFO of observation batches.

    Producers call :meth:`push`; a consumer drains :meth:`batches`. ``close``
    terminates the consumer iterator once queued batches are delivered.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._q: queue.Queue[list[Observation] | None] = queue.Queue(maxsize)

    def push(self, batch: list[Observation]) -> None:
        """Enqueue one batch (blocks if a bounded queue is full)."""
        self._q.put(list(batch))

    def close(self) -> None:
        """Signal end-of-stream after all previously pushed batches."""
        self._q.put(None)

    def batches(self) -> Iterator[list[Observation]]:
        """Yield batches until :meth:`close` is called."""
        while True:
            batch = self._q.get()
            if batch is None:
                return
            yield batch


class LoopbackFeed:
    """Replays scripted observation batches in deterministic order.

    Useful for tests and demos: the same script always produces the same
    stream, which (with a fixed fusion policy) gives reproducible results.
    """

    def __init__(self, scripted: Iterable[list[Observation]]) -> None:
        self._scripted: list[list[Observation]] = [list(b) for b in scripted]

    def batches(self) -> Iterator[list[Observation]]:
        for batch in self._scripted:
            yield list(batch)
