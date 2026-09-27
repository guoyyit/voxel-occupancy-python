"""线程安全的三维体素概率栅格。

概率模型
--------
对每个已观测体素维护对数几率 L = log(p/(1-p))：
- OCCUPIED 观测：L += occupied_log_odds * weight
- FREE 观测：    L += free_log_odds * weight（负值）
- CLEAR 观测：   删除体素（回到未观测），之后可被再次占用
L 始终被夹在 [clamp_min, clamp_max]，因此 p 有界、永不为 0/1。

有界内存
--------
1. 未观测体素不占存储（稀疏分块 dict）；
2. 融合后 |L| <= prune_deadband 的体素被剪枝（与先验 0.5 不可区分），
   冲突观测相互抵消后内存会自动回收。

并发与一致性
------------
- 一把 RLock 串行化所有提交，并发提交等价于某一确定的串行顺序；
- 分块 + 顶层字典 copy-on-write：提交在副本上完成全部修改后一次性发布；
- 查询持同一把读锁并绑定一个快照，单查询与 read() 复合视图都具有
  快照隔离：只能看到已提交版本，看不到部分更新。
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from .config import GridConfig
from .types import (
    CellCoord,
    CellState,
    CommitResult,
    LineTraceResult,
    Observation,
    ObservationBatch,
    ObservationKind,
    SnapshotStats,
    VoxelInfo,
)

CHUNK_SHIFT = 4
CHUNK_SIZE = 1 << CHUNK_SHIFT  # 16
CHUNK_MASK = CHUNK_SIZE - 1

ChunkKey = tuple[int, int, int]


def chunk_key(coord: CellCoord) -> ChunkKey:
    """返回体素所属分块坐标（对负坐标同样成立，>> 为向下取整）。"""
    x, y, z = coord
    return (x >> CHUNK_SHIFT, y >> CHUNK_SHIFT, z >> CHUNK_SHIFT)


def _probability(log_odds: float) -> float:
    return 1.0 - 1.0 / (1.0 + math.exp(log_odds))


def bresenham3d(start: CellCoord, end: CellCoord) -> list[CellCoord]:
    """三维整数 Bresenham 直线，返回含起止点的体素序列。"""
    x0, y0, z0 = start
    x1, y1, z1 = end
    points: list[CellCoord] = [(x0, y0, z0)]
    if start == end:
        return points

    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    if dx >= dy and dx >= dz:
        err_y, err_z = 2 * dy - dx, 2 * dz - dx
        for _ in range(dx):
            if err_y >= 0:
                y0 += sy
                err_y -= 2 * dx
            err_y += 2 * dy
            if err_z >= 0:
                z0 += sz
                err_z -= 2 * dx
            err_z += 2 * dz
            x0 += sx
            points.append((x0, y0, z0))
    elif dy >= dx and dy >= dz:
        err_x, err_z = 2 * dx - dy, 2 * dz - dy
        for _ in range(dy):
            if err_x >= 0:
                x0 += sx
                err_x -= 2 * dy
            err_x += 2 * dx
            if err_z >= 0:
                z0 += sz
                err_z -= 2 * dy
            err_z += 2 * dz
            y0 += sy
            points.append((x0, y0, z0))
    else:
        err_x, err_y = 2 * dx - dz, 2 * dy - dz
        for _ in range(dz):
            if err_x >= 0:
                x0 += sx
                err_x -= 2 * dz
            err_x += 2 * dx
            if err_y >= 0:
                y0 += sy
                err_y -= 2 * dz
            err_y += 2 * dy
            z0 += sz
            points.append((x0, y0, z0))
    return points


class VoxelGrid:
    """并发安全的占用概率栅格。"""

    def __init__(self, config: GridConfig | None = None) -> None:
        self._config = config or GridConfig()
        self._lock = threading.RLock()
        # 已提交快照：chunk_key -> {coord: CellState}
        # 分块 dict 一经发布即不可变；提交时复制受影响分块后整体替换。
        self._chunks: dict[ChunkKey, dict[CellCoord, CellState]] = {}
        self._committed_sequence = 0
        self._committed_batches = 0

    # ================= 快照 / 恢复 =================
    def snapshot_cells(self) -> dict[CellCoord, CellState]:
        """导出全部已存储体素（供持久化）。"""
        with self._lock:
            out: dict[CellCoord, CellState] = {}
            for chunk in self._chunks.values():
                out.update(chunk)
            return out

    def load_cells(
        self,
        cells: dict[CellCoord, CellState],
        *,
        sequence: int,
        batches: int,
    ) -> None:
        """用恢复数据整体替换栅格内容。"""
        rebuilt: dict[ChunkKey, dict[CellCoord, CellState]] = {}
        for coord, state in cells.items():
            rebuilt.setdefault(chunk_key(coord), {})[coord] = state
        with self._lock:
            self._chunks = rebuilt
            self._committed_sequence = sequence
            self._committed_batches = batches

    @property
    def committed_sequence(self) -> int:
        return self._committed_sequence

    @property
    def committed_batches(self) -> int:
        return self._committed_batches

    # ================= 更新 =================
    def commit(
        self,
        batch: ObservationBatch,
        *,
        weight_scale: float = 1.0,
    ) -> CommitResult:
        """原子地融合一整批观测。

        批次内对同一体素的多条观测按顺序累加到同一副本，
        提交结束时一次性发布，故任何查询都不会看到半批结果。
        """
        scale = max(0.0, weight_scale)
        cfg = self._config

        # 按分块归组，每块只复制一次
        grouped: dict[ChunkKey, list[Observation]] = defaultdict(list)
        for obs in batch.observations:
            grouped[chunk_key(obs.coord)].append(obs)

        with self._lock:
            # 顶层字典也采用 copy-on-write：先在副本上完成所有分块替换，
            # 持锁最后一次性发布。读路径持同一把锁绑定 self._chunks，
            # 看到的必然是某个完整的已提交版本，绝不会读到部分更新。
            published = dict(self._chunks)
            touched = 0
            for key, observations in grouped.items():
                old = published.get(key)
                new_chunk = dict(old) if old else {}
                for obs in observations:
                    coord = obs.coord
                    if obs.kind is ObservationKind.CLEAR:
                        if new_chunk.pop(coord, None) is not None:
                            touched += 1
                        continue

                    weight = max(0.0, obs.weight) * scale
                    if weight == 0.0:
                        continue
                    if obs.kind is ObservationKind.OCCUPIED:
                        delta = cfg.occupied_log_odds * weight
                    else:  # FREE
                        delta = cfg.free_log_odds * weight
                    if delta == 0.0:
                        continue

                    prev = new_chunk.get(coord)
                    old_value = prev.log_odds if prev is not None else cfg.prior_log_odds
                    value = max(cfg.clamp_min, min(cfg.clamp_max, old_value + delta))

                    if abs(value) <= cfg.prune_deadband:
                        # 回到与先验不可区分：剪枝，释放内存
                        if new_chunk.pop(coord, None) is not None:
                            touched += 1
                    else:
                        new_chunk[coord] = CellState(
                            coord=coord,
                            log_odds=value,
                            last_updated=float(batch.sequence),
                        )
                        touched += 1

                if new_chunk:
                    published[key] = new_chunk
                else:
                    published.pop(key, None)

            self._chunks = published
            self._committed_batches += 1
            if batch.sequence > self._committed_sequence:
                self._committed_sequence = batch.sequence

            return CommitResult(
                committed=True,
                reason=None,
                touched=touched,
                sequence=self._committed_sequence,
            )

    # ================= 查询 =================
    def _bind_snapshot(self):
        """获取一次一致性读所需的（锁上下文, 快照）。

        读路径全程持有与提交相同的 RLock，期间发布的新版本不会被读到，
        从而保证复合查询（如逐体素扫描）对应同一个已提交快照。
        单次查询同样经过这里，语义统一。
        """
        self._lock.acquire()
        return self._lock, self._chunks

    @contextmanager
    def read(self) -> Iterator["GridView"]:
        """一致性读视图：with grid.read() as view: ...

        视图内的 get/is_occupied/query_region/trace_line 全部对应
        进入 read() 那一刻的已提交快照，期间提交被阻塞。
        """
        lock, chunks = self._bind_snapshot()
        try:
            yield GridView(self._config, chunks)
        finally:
            lock.release()

    def _view(self, state: CellState) -> VoxelInfo:
        p = _probability(state.log_odds)
        return VoxelInfo(
            coord=state.coord,
            probability=p,
            log_odds=state.log_odds,
            occupied=p >= self._config.occupied_threshold,
        )

    def get(self, coord: CellCoord) -> VoxelInfo | None:
        lock, chunks = self._bind_snapshot()
        try:
            chunk = chunks.get(chunk_key(coord))
            if chunk is None:
                return None
            state = chunk.get(coord)
            return self._view(state) if state is not None else None
        finally:
            lock.release()

    def is_occupied(self, coord: CellCoord) -> bool:
        """未观测 / 已清除 / 被剪枝的体素一律视为不占用。"""
        info = self.get(coord)
        return bool(info and info.occupied)

    def query_region(
        self,
        center: CellCoord,
        max_range: float,
        *,
        occupied_only: bool = False,
    ) -> list[VoxelInfo]:
        """球心 center、最大欧氏半径 max_range 内的已观测体素。

        只遍历与半径包围盒相交的分块，未观测区域不产生任何开销条目。
        """
        if max_range < 0:
            raise ValueError("max_range must be non-negative")
        lock, chunks = self._bind_snapshot()
        try:
            return _query_region(
                self._config, chunks, center, max_range, occupied_only
            )
        finally:
            lock.release()

    def trace_line(self, start: CellCoord, end: CellCoord) -> LineTraceResult:
        """直线可达查询。

        沿 Bresenham 直线检查（不含起点、含终点）：第一个占用体素即阻挡点。
        未观测/已清除体素不阻挡。
        """
        lock, chunks = self._bind_snapshot()
        try:
            return _trace_line(self._config, chunks, start, end)
        finally:
            lock.release()

    # ================= 统计 =================
    def stats(self) -> SnapshotStats:
        with self._lock:
            return SnapshotStats(
                stored_cells=sum(len(c) for c in self._chunks.values()),
                committed_sequence=self._committed_sequence,
                committed_batches=self._committed_batches,
            )

    def _iter_snapshot_cells(self) -> Iterator[CellState]:
        lock, chunks = self._bind_snapshot()
        try:
            for chunk in chunks.values():
                yield from chunk.values()
        finally:
            lock.release()


def _make_view(cfg: GridConfig, state: CellState) -> VoxelInfo:
    p = _probability(state.log_odds)
    return VoxelInfo(
        coord=state.coord,
        probability=p,
        log_odds=state.log_odds,
        occupied=p >= cfg.occupied_threshold,
    )


def _query_region(cfg, chunks, center, max_range, occupied_only) -> list[VoxelInfo]:
    cx, cy, cz = center
    r = int(math.ceil(max_range))
    r2 = max_range * max_range
    kx0, ky0, kz0 = (cx - r) >> CHUNK_SHIFT, (cy - r) >> CHUNK_SHIFT, (cz - r) >> CHUNK_SHIFT
    kx1, ky1, kz1 = (cx + r) >> CHUNK_SHIFT, (cy + r) >> CHUNK_SHIFT, (cz + r) >> CHUNK_SHIFT

    result: list[VoxelInfo] = []
    for kx in range(kx0, kx1 + 1):
        for ky in range(ky0, ky1 + 1):
            for kz in range(kz0, kz1 + 1):
                chunk = chunks.get((kx, ky, kz))
                if chunk is None:
                    continue
                for coord, state in chunk.items():
                    x, y, z = coord
                    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                    if d2 > r2:
                        continue
                    view = _make_view(cfg, state)
                    if occupied_only and not view.occupied:
                        continue
                    result.append(view)
    return result


def _trace_line(cfg, chunks, start: CellCoord, end: CellCoord) -> LineTraceResult:
    path = bresenham3d(start, end)
    blocked: CellCoord | None = None
    for coord in path[1:]:
        chunk = chunks.get(chunk_key(coord))
        if chunk is None:
            continue
        state = chunk.get(coord)
        if state is not None and cfg.is_occupied_log_odds(state.log_odds):
            blocked = coord
            break
    return LineTraceResult(
        start=start,
        end=end,
        reachable=blocked is None,
        blocked_coord=blocked,
        path=tuple(path),
    )


class GridView:
    """绑定到单一已提交快照的只读视图（由 VoxelGrid.read() 产生）。"""

    __slots__ = ("_config", "_chunks")

    def __init__(self, config: GridConfig, chunks) -> None:
        self._config = config
        self._chunks = chunks

    def get(self, coord: CellCoord) -> VoxelInfo | None:
        chunk = self._chunks.get(chunk_key(coord))
        if chunk is None:
            return None
        state = chunk.get(coord)
        return _make_view(self._config, state) if state is not None else None

    def is_occupied(self, coord: CellCoord) -> bool:
        info = self.get(coord)
        return bool(info and info.occupied)

    def query_region(
        self, center: CellCoord, max_range: float, *, occupied_only: bool = False
    ) -> list[VoxelInfo]:
        if max_range < 0:
            raise ValueError("max_range must be non-negative")
        return _query_region(
            self._config, self._chunks, center, max_range, occupied_only
        )

    def trace_line(self, start: CellCoord, end: CellCoord) -> LineTraceResult:
        return _trace_line(self._config, self._chunks, start, end)
