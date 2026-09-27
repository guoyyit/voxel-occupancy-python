"""核心数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# 三维整数栅格坐标
CellCoord = tuple[int, int, int]


class ObservationKind(str, Enum):
    """单次观测对体素的作用。"""

    OCCUPIED = "occupied"   # 观测到占用，推高占用概率
    FREE = "free"           # 观测到自由（如射线穿过），压低占用概率
    CLEAR = "clear"         # 显式清除：删除体素，回到未观测状态


@dataclass(frozen=True, slots=True)
class Observation:
    """对单个体素的一次观测。"""

    coord: CellCoord
    kind: ObservationKind
    timestamp: float = 0.0
    weight: float = 1.0
    """观测权重（非负，1.0 为标准观测），过期降权策略会改写它。"""


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """一批原子观测：要么整体提交，要么整体不生效。"""

    observations: tuple[Observation, ...]
    sequence: int = 0
    """单调递增批次序号，用于乱序/过期判断。"""
    timestamp: float = 0.0
    """批次观测时间。"""
    idempotency_key: str | None = None
    """可选幂等键，重复提交同一批次只生效一次。"""


@dataclass(frozen=True, slots=True)
class CellState:
    """单个已存储体素的状态。"""

    coord: CellCoord
    log_odds: float
    """对数几率，始终夹在有界区间内。"""
    last_updated: float
    """最近一次影响该体素的批次序号。"""

    @property
    def probability(self) -> float:
        """占用概率，范围 (0, 1)。"""
        from math import exp

        return 1.0 - 1.0 / (1.0 + exp(self.log_odds))


@dataclass(frozen=True, slots=True)
class CommitResult:
    """一次提交的结果。"""

    committed: bool
    """False 表示批次因过期或重复被整体丢弃。"""
    reason: str | None
    touched: int
    """实际发生变化的体素数量。"""
    sequence: int
    """服务当前已提交批次序号。"""
    dedup: bool = False
    """是否命中幂等去重。"""


@dataclass(frozen=True, slots=True)
class VoxelInfo:
    """查询返回的体素视图。"""

    coord: CellCoord
    probability: float
    log_odds: float
    occupied: bool


@dataclass(frozen=True, slots=True)
class LineTraceResult:
    """直线可达查询结果。"""

    start: CellCoord
    end: CellCoord
    reachable: bool
    """起点到终点之间（不含起点，含终点）是否没有任何占用体素。"""
    blocked_coord: CellCoord | None
    """第一个阻挡体素坐标，可达时为 None。"""
    path: tuple[CellCoord, ...]
    """直线经过的体素序列（含起止点）。"""


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    """栅格统计信息。"""

    stored_cells: int
    """已存储（观测过且未被剪枝）的体素数。"""
    committed_sequence: int
    committed_batches: int
