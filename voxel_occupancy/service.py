"""占用栅格服务门面。

组装：
- VoxelGrid：概率融合与快照查询；
- 过期策略：提交前由 staleness 纯函数判定（丢弃 / 降权 / 接受）；
- 幂等去重：按 idempotency_key 缓存最近 N 个批次的提交结果（LRU）；
- 持久化：每次成功提交后原子落盘（配置开启时）。

submit 内部持锁完成"判定 -> 提交 -> 落盘 -> 记录元数据"，
保证服务重启恢复后状态与崩溃前最后一次成功提交一致。
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock

from .config import ServiceConfig
from .grid import VoxelGrid
from .persistence import PersistenceConfig, load_grid, save_grid
from .staleness import decide_staleness
from .transport import MemoryPipe
from .types import (
    CellCoord,
    CommitResult,
    LineTraceResult,
    ObservationBatch,
    SnapshotStats,
    VoxelInfo,
)


class OccupancyService:
    """对外的占用栅格服务。"""

    def __init__(self, config: ServiceConfig | None = None) -> None:
        self._config = config or ServiceConfig()
        self._grid = VoxelGrid(self._config.grid)
        self._submit_lock = RLock()
        self._dedup_lock = RLock()
        self._dedup: OrderedDict[str, CommitResult] = OrderedDict()
        # 用已恢复状态初始化元数据
        self._latest_timestamp = 0.0
        if self._config.persistence is not None and self._config.persistence.restore_on_start:
            load_grid(self._grid, self._config.persistence)

    @property
    def grid(self) -> VoxelGrid:
        return self._grid

    def _cache_get(self, key: str) -> CommitResult | None:
        with self._dedup_lock:
            if key in self._dedup:
                self._dedup.move_to_end(key)
                return self._dedup[key]
            return None

    def _cache_put(self, key: str, result: CommitResult) -> None:
        with self._dedup_lock:
            self._dedup[key] = result
            self._dedup.move_to_end(key)
            while len(self._dedup) > self._config.dedup_cache_size:
                self._dedup.popitem(last=False)

    def submit(self, batch: ObservationBatch) -> CommitResult:
        """提交一批观测。

        处理顺序：幂等去重 -> 过期判定 -> 栅格原子提交 -> 持久化。
        对同一 idempotency_key 的重复提交直接返回首次结果。
        """
        # 幂等去重（空键不参与）
        if batch.idempotency_key:
            cached = self._cache_get(batch.idempotency_key)
            if cached is not None:
                return CommitResult(
                    committed=False,
                    reason="duplicate",
                    touched=cached.touched,
                    sequence=cached.sequence,
                    dedup=True,
                )

        with self._submit_lock:
            # 双重检查，避免等待锁期间被其他线程提交
            if batch.idempotency_key:
                cached = self._cache_get(batch.idempotency_key)
                if cached is not None:
                    return CommitResult(
                        committed=False,
                        reason="duplicate",
                        touched=cached.touched,
                        sequence=cached.sequence,
                        dedup=True,
                    )

            decision = decide_staleness(
                batch,
                committed_sequence=self._grid.committed_sequence,
                latest_timestamp=self._latest_timestamp,
                config=self._config.staleness,
            )
            if not decision.accept:
                result = CommitResult(
                    committed=False,
                    reason=decision.reason,
                    touched=0,
                    sequence=self._grid.committed_sequence,
                )
            else:
                result = self._grid.commit(batch, weight_scale=decision.weight_scale)
                if decision.reason:  # 降权接受
                    result = CommitResult(
                        committed=True,
                        reason=decision.reason,
                        touched=result.touched,
                        sequence=result.sequence,
                    )
                # 提交成功后推进时间水位（只增不减，保证年龄判定确定）
                if batch.timestamp > self._latest_timestamp:
                    self._latest_timestamp = batch.timestamp
                # 原子持久化
                if self._config.persistence is not None:
                    save_grid(self._grid, self._config.persistence)

            if batch.idempotency_key:
                self._cache_put(batch.idempotency_key, result)
            return result

    # ---- 查询代理：直接读栅格已提交快照 ----
    def get(self, coord: CellCoord) -> VoxelInfo | None:
        return self._grid.get(coord)

    def is_occupied(self, coord: CellCoord) -> bool:
        return self._grid.is_occupied(coord)

    def query_region(
        self, center: CellCoord, max_range: float, *, occupied_only: bool = False
    ) -> list[VoxelInfo]:
        return self._grid.query_region(
            center, max_range, occupied_only=occupied_only
        )

    def trace_line(self, start: CellCoord, end: CellCoord) -> LineTraceResult:
        return self._grid.trace_line(start, end)

    @contextmanager
    def read(self):
        """一致性多查询读视图：with svc.read() as view: ..."""
        with self._grid.read() as view:
            yield view

    def stats(self) -> SnapshotStats:
        return self._grid.stats()

    def flush(self) -> None:
        """显式把当前已提交状态持久化。"""
        if self._config.persistence is None:
            raise RuntimeError("persistence is not configured")
        with self._submit_lock:
            save_grid(self._grid, self._config.persistence)

    def serve_pipe(
        self,
        pipe: MemoryPipe,
        *,
        max_items: int | None = None,
        timeout: float | None = 0.0,
    ) -> list[CommitResult]:
        """从内存管道拉取批次并依次提交。"""
        results: list[CommitResult] = []
        if max_items is not None:
            for batch in pipe.drain(max_items=max_items):
                results.append(self.submit(batch))
            return results
        while True:
            batch = pipe.recv(timeout=timeout)
            if batch is None:
                break
            results.append(self.submit(batch))
        return results
