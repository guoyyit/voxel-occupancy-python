"""内存管道与回环接口。"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from voxel_occupancy.config import ServiceConfig
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.transport import LoopbackTransport, MemoryPipe
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


def test_fifo_ordering_and_drain():
    p = MemoryPipe()
    for i in range(5):
        p.send(ObservationBatch((), sequence=i))
    assert p.recv(timeout=0).sequence == 0
    rest = p.drain()
    assert [b.sequence for b in rest] == [1, 2, 3, 4]
    assert len(p) == 0
    assert p.recv(timeout=0.01) is None


def test_bounded_pipe_blocks_when_full():
    p = MemoryPipe(maxsize=1)
    p.send(ObservationBatch((), sequence=0))
    with pytest.raises(queue.Full):
        p.send(ObservationBatch(()), timeout=0.02)


def test_close_wakes_waiters_and_rejects_send():
    p = MemoryPipe()
    out = []

    def wait():
        out.append(p.recv(timeout=5.0))

    t = threading.Thread(target=wait)
    t.start()
    time.sleep(0.05)
    p.close()
    t.join(1.0)
    assert out == [None]
    assert p.closed
    with pytest.raises(RuntimeError):
        p.send(ObservationBatch(()))


def test_loopback_pump_into_service():
    svc = OccupancyService(ServiceConfig())
    loop = LoopbackTransport()
    for i in range(3):
        loop.send_batch(
            ObservationBatch(
                (Observation((i, 0, 0), ObservationKind.OCCUPIED),) * 5,
                sequence=i + 1,
            )
        )
    results = loop.pump(svc.submit)
    assert all(r.committed for r in results)
    for i in range(3):
        assert svc.is_occupied((i, 0, 0))


def test_service_consumes_pipe():
    svc = OccupancyService(ServiceConfig())
    pipe = MemoryPipe()
    pipe.send(
        ObservationBatch(
            (Observation((7, 8, 9), ObservationKind.OCCUPIED),) * 5, sequence=1
        )
    )
    results = svc.serve_pipe(pipe)
    assert len(results) == 1 and results[0].committed
    assert svc.is_occupied((7, 8, 9))
    assert svc.serve_pipe(pipe) == []


def test_producer_consumer_threads_via_pipe():
    svc = OccupancyService(ServiceConfig())
    pipe = MemoryPipe(maxsize=4)

    def produce():
        for i in range(100):
            pipe.send(
                ObservationBatch(
                    (Observation((i, 0, 0), ObservationKind.OCCUPIED),),
                    sequence=i + 1,
                ),
                timeout=5.0,
            )
        pipe.close()

    pt = threading.Thread(target=produce)
    pt.start()

    consumed = 0
    for result in LoopbackTransport(pipe).stream(svc.submit):
        assert result.committed
        consumed += 1
    pt.join()
    assert consumed == 100
    assert svc.stats().committed_batches == 100
