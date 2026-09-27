"""过期 / 乱序观测的可配置丢弃与降权策略（纯函数，可复现）。

过期的两种判据（任一满足即视为过期）：
1. 序号滞后：已提交序号 N 与新批次序号 S 满足 N - S > max_sequence_lag
   （max_sequence_lag=0 表示不启用序号判据）；
2. 年龄过大：最新批次时间 latest 与本批次时间 t 满足 latest - t > max_age_seconds
   （max_age_seconds=0 表示不启用年龄判据）。

策略：
- ACCEPT：接受，权重不变；
- DROP：整体丢弃（accept=False）；
- DOWNWEIGHT：接受，但权重随"超出量"线性衰减到 downweight_min。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import StalenessConfig, StalenessMode
from .types import ObservationBatch


@dataclass(frozen=True, slots=True)
class StalenessDecision:
    """对一个批次的过期判定结果。"""

    accept: bool
    weight_scale: float
    """作用到批次内每条观测权重上的乘数。"""
    reason: str | None


def decide_staleness(
    batch: ObservationBatch,
    *,
    committed_sequence: int,
    latest_timestamp: float,
    config: StalenessConfig,
) -> StalenessDecision:
    """判断批次是否过期，以及降权乘数。

    纯函数：相同输入必定得到相同结果。
    """
    lag = 0
    seq_stale = (
        config.max_sequence_lag > 0
        and committed_sequence - batch.sequence > config.max_sequence_lag
    )
    if seq_stale:
        lag = max(lag, committed_sequence - batch.sequence - config.max_sequence_lag)

    age = latest_timestamp - batch.timestamp
    age_stale = config.max_age_seconds > 0 and age > config.max_age_seconds

    if not seq_stale and not age_stale:
        return StalenessDecision(accept=True, weight_scale=1.0, reason=None)

    if config.mode is StalenessMode.ACCEPT:
        # 配置为接受模式：即使判据命中也原样接受
        return StalenessDecision(accept=True, weight_scale=1.0, reason=None)

    if config.mode is StalenessMode.DROP:
        reasons = []
        if seq_stale:
            reasons.append(f"sequence_lag={committed_sequence - batch.sequence}")
        if age_stale:
            reasons.append(f"age={age:.6f}s")
        return StalenessDecision(
            accept=False, weight_scale=0.0, reason="stale:" + ",".join(reasons)
        )

    # DOWNWEIGHT：用归一化超出比例做线性衰减，确定性、可复现
    # 比例 0 -> 权重 1.0；比例 >=1 -> 权重 downweight_min
    ratio = 0.0
    if config.max_sequence_lag > 0:
        denom = max(config.max_sequence_lag, 1)
        ratio = max(ratio, lag / denom)
    if config.max_age_seconds > 0:
        ratio = max(ratio, age / config.max_age_seconds - 1.0)
    ratio = max(0.0, min(1.0, ratio))
    scale = 1.0 - ratio * (1.0 - config.downweight_min)
    return StalenessDecision(accept=True, weight_scale=scale, reason="stale:downweight")
