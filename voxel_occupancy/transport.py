"""本地观测输入通道：内存管道与回环接口。

不依赖任何外部设备或服务。
- MemoryPipe: 线程安全的有界 FIFO（Condition 实现），
  send 放入批次，recv 取出（阻塞 / 超时 / 非阻塞）。
- LoopbackTransport: 在 MemoryPipe 之上的回环接口，
  观测被"发出"后再由本进程内的消费者取回处理，便于测试与离线回放。
"""

from __future__ import annotations

import queue
from collections import deque
from threading import Condition
from typing import Callable, Iterator

from .types import CommitResult, ObservationBatch


class MemoryPipe:
    """有界内存 FIFO 管道。"""

    def __init__(self, maxsize: int = 0) -> None:
        self._cv = Condition()
        self._q: deque[ObservationBatch] = deque()
        self._maxsize = max(0, maxsize)  # 0 = 无界
        self._closed = False

    def send(self, batch: ObservationBatch, *, timeout: float | None = None) -> None:
        """放入一个批次。

        有界且满时阻塞至多 timeout 秒；超时抛 queue.Full。
        管道已关闭时抛 RuntimeError。
        """
        with self._cv:
            if self._closed:
                raise RuntimeError("pipe is closed")
            if self._maxsize:
                if not self._cv.wait_for(
                    lambda: len(self._q) < self._maxsize or self._closed,
                    timeout=timeout,
                ):
                    raise queue.Full()
                if self._closed:
                    raise RuntimeError("pipe is closed")
            self._q.append(batch)
            self._cv.notify_all()

    def recv(self, *, timeout: float | None = None) -> ObservationBatch | None:
        """取出一个批次。

        timeout=None 阻塞直到有数据或关闭；timeout=0 非阻塞。
        超时返回 None；管道关闭且已排空时返回 None。
        """
        with self._cv:
            if not self._q:
                if self._closed:
                    return None
                self._cv.wait_for(
                    lambda: bool(self._q) or self._closed, timeout=timeout
                )
                if not self._q:
                    return None
            item = self._q.popleft()
            self._cv.notify_all()
            return item

    def drain(self, max_items: int | None = None) -> list[ObservationBatch]:
        """非阻塞地一次性取出当前全部（或至多 max_items 个）批次。"""
        with self._cv:
            if max_items is None:
                items = list(self._q)
                self._q.clear()
            else:
                items = [self._q.popleft() for _ in range(min(max_items, len(self._q)))]
            self._cv.notify_all()
            return items

    def close(self) -> None:
        """关闭管道，唤醒所有等待者；关闭幂等。"""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed


Handler = Callable[[ObservationBatch], CommitResult]


class LoopbackTransport:
    """回环接口：写入管道，再由消费者取回交给处理回调。"""

    def __init__(self, pipe: MemoryPipe | None = None, *, maxsize: int = 0) -> None:
        self._pipe = pipe or MemoryPipe(maxsize=maxsize)

    @property
    def pipe(self) -> MemoryPipe:
        return self._pipe

    def send_batch(self, batch: ObservationBatch, *, timeout: float | None = None) -> None:
        self._pipe.send(batch, timeout=timeout)

    def pump(
        self,
        handler: Handler,
        *,
        max_items: int | None = None,
        timeout: float | None = 0.0,
    ) -> list[CommitResult]:
        """拉取至多 max_items 个当前可取的批次并处理。

        timeout=0 时完全非阻塞；默认返回已入队的全部批次。
        """
        results: list[CommitResult] = []
        if max_items is not None:
            for batch in self._pipe.drain(max_items=max_items):
                results.append(handler(batch))
            return results
        while True:
            batch = self._pipe.recv(timeout=timeout)
            if batch is None:
                break
            results.append(handler(batch))
        return results

    def stream(
        self, handler: Handler, *, timeout: float | None = None
    ) -> Iterator[CommitResult]:
        """持续拉取并处理，直到管道关闭且排空。"""
        while True:
            batch = self._pipe.recv(timeout=timeout)
            if batch is None:
                return
            yield handler(batch)

    def close(self) -> None:
        self._pipe.close()
