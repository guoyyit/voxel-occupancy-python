"""配置项。

所有融合运算只依赖这些参数与随机数无关的确定性计算，
因此同一输入与同一配置下结果可复现。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import CellCoord


class StalenessMode(str, Enum):
    """过期观测处理策略。"""

    ACCEPT = "accept"
    """不做任何过期处理，全部接受（重复批次仍按幂等键去重）。"""
    DROP = "drop"
    """过期观测整体丢弃，不影响栅格。"""
    DOWNWEIGHT = "downweight"
    """过期观测仍融合，但权重线性衰减。"""


@dataclass(frozen=True, slots=True)
class GridConfig:
    """概率栅格参数（对数几率模型）。"""

    resolution: float = 1.0
    """体素边长（米），用于连续坐标换算；本服务内部以离散坐标存储。"""
    occupied_log_odds: float = 0.85
    """单次 OCCUPIED 观测的对数几率增量。"""
    free_log_odds: float = -0.45
    """单次 FREE 观测的对数几率增量。"""
    clamp_min: float = -3.5
    """对数几率下界（强自由），保证概率有界。"""
    clamp_max: float = 3.5
    """对数几率上界（强占用），保证概率有界。"""
    occupied_threshold: float = 0.5
    """判定为占用的概率阈值（p >= threshold）。"""
    prune_deadband: float = 1e-3
    """|log_odds| 小于该值的体素被剪枝（视为未观测），有界内存的关键。"""

    @property
    def prior_log_odds(self) -> float:
        """先验为均匀分布 0.5，对应 log-odds = 0。"""
        return 0.0

    def is_occupied_log_odds(self, log_odds: float) -> bool:
        from math import exp

        p = 1.0 - 1.0 / (1.0 + exp(log_odds))
        return p >= self.occupied_threshold


@dataclass(frozen=True, slots=True)
class StalenessConfig:
    """过期/乱序观测策略配置。"""

    mode: StalenessMode = StalenessMode.ACCEPT
    max_sequence_lag: int = 0
    """允许的最大批次序号落后量；0 表示不按序号判过期。
    若当前已提交序号为 N，新批次序号 S 满足 N - S > max_sequence_lag 即过期。"""
    max_age_seconds: float = 0.0
    """允许的最大观测年龄（相对当前最新批次时间）；0 表示不按年龄判过期。"""
    downweight_min: float = 0.1
    """降权策略下的最低权重。"""


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    """持久化配置。"""

    path: str = "voxel_grid.dat"
    fsync: bool = True
    """提交后是否 fsync 目录/文件（崩溃安全）。"""
    restore_on_start: bool = True
    """启动时若存在持久化文件则恢复。"""
    corrupt_backup_suffix: str = ".corrupt"
    """校验失败时把坏文件改名保留，避免重复报错；留空则报错不改名。"""


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """服务总配置。"""

    grid: GridConfig = GridConfig()
    staleness: StalenessConfig = StalenessConfig()
    persistence: PersistenceConfig | None = None
    dedup_cache_size: int = 4096
    """幂等键 LRU 缓存容量。"""
    origin: CellCoord = (0, 0, 0)
    """连续坐标换算时的栅格原点偏移（离散坐标下仅用于信息展示）。"""
