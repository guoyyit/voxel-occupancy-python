"""崩溃安全持久化：原子写、校验、重启恢复与最后提交状态一致。"""

from __future__ import annotations

import os

import pytest

from voxel_occupancy.config import (
    PersistenceConfig,
    ServiceConfig,
    StalenessConfig,
    StalenessMode,
)
from voxel_occupancy.grid import VoxelGrid
from voxel_occupancy.persistence import (
    CorruptGridError,
    load_grid,
    save_grid,
)
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


@pytest.fixture
def path_factory(tmp_path):
    counter = [0]

    def make():
        counter[0] += 1
        return str(tmp_path / f"grid-{counter[0]}.dat")

    return make


def _populate(g: VoxelGrid) -> None:
    obs = []
    for x in range(-30, 31, 3):
        for y in range(-10, 11, 5):
            obs.append(Observation((x, y, x % 7), ObservationKind.OCCUPIED, 1.0))
    g.commit(ObservationBatch(tuple(obs), sequence=42, timestamp=7.5))
    g.commit(
        ObservationBatch(
            (Observation((-3, 0, 0), ObservationKind.FREE, 2.0),) * 10,
            sequence=43,
            timestamp=8.0,
        )
    )
    g.commit(
        ObservationBatch(
            (Observation((999, 999, 999), ObservationKind.CLEAR),), sequence=44
        )
    )


def test_roundtrip_restores_exact_state(path_factory):
    path = path_factory()
    cfg = PersistenceConfig(path=path)
    g = VoxelGrid()
    _populate(g)
    save_grid(g, cfg)

    restored = VoxelGrid()
    assert load_grid(restored, cfg) is True
    before, after = g.snapshot_cells(), restored.snapshot_cells()
    assert set(before) == set(after)
    for coord in before:
        assert before[coord].log_odds == after[coord].log_odds
        assert before[coord].last_updated == after[coord].last_updated
    assert restored.committed_sequence == 44
    assert restored.committed_batches == 3


def test_missing_file_keeps_empty_grid(path_factory):
    g = VoxelGrid()
    assert load_grid(g, PersistenceConfig(path=path_factory())) is False
    assert g.stats().stored_cells == 0


def test_truncated_file_detected(path_factory):
    path = path_factory()
    cfg = PersistenceConfig(path=path)
    g = VoxelGrid()
    _populate(g)
    save_grid(g, cfg)

    blob = open(path, "rb").read()
    # 模拟写到一半崩溃
    with open(path, "wb") as f:
        f.write(blob[: len(blob) // 2])

    with pytest.raises(CorruptGridError):
        load_grid(VoxelGrid(), cfg)
    # 坏文件被改名备份，正式路径不再残留坏文件
    assert not os.path.exists(path)
    assert os.path.exists(path + ".corrupt")


def test_corrupted_byte_detected(path_factory):
    path = path_factory()
    cfg = PersistenceConfig(path=path)
    g = VoxelGrid()
    _populate(g)
    save_grid(g, cfg)

    blob = bytearray(open(path, "rb").read())
    blob[len(blob) // 2] ^= 0xFF
    with open(path, "wb") as f:
        f.write(blob)
    with pytest.raises(CorruptGridError):
        load_grid(VoxelGrid(), cfg)


def test_atomic_save_never_half_written(path_factory, monkeypatch):
    """在 write/fsync 各阶段注入崩溃，正式文件要么不存在要么完好。"""
    path = path_factory()
    cfg = PersistenceConfig(path=path, fsync=True)
    good = VoxelGrid()
    _populate(good)
    save_grid(good, cfg)

    # 第二次保存期间崩溃：旧文件必须保持完好可读
    crash_points = ["voxel_occupancy.persistence.zlib.compress", "os.fsync", "os.replace"]
    for cp in crash_points:
        if cp == "voxel_occupancy.persistence.zlib.compress":
            def boom(*a, **k):
                raise RuntimeError("crash during encode")
            monkeypatch.setattr(cp, boom)
        else:
            real = getattr(__import__("os"), cp.split(".")[1])

            def make_boom(real_fn):
                def boom(*a, **k):
                    raise RuntimeError(f"crash at {cp}")
                return boom

            monkeypatch.setattr(cp, make_boom(real))

        g2 = VoxelGrid()
        g2.commit(
            ObservationBatch(
                (Observation((1, 1, 1), ObservationKind.OCCUPIED),) * 5,
                sequence=100,
            )
        )
        with pytest.raises(RuntimeError):
            save_grid(g2, cfg)

        check = VoxelGrid()
        assert load_grid(check, cfg) is True
        assert check.committed_sequence == 44
        # 新批次绝不能部分可见
        assert check.get((1, 1, 1)) is None
        # 临时文件已清理
        leftovers = [n for n in os.listdir(os.path.dirname(path))
                     if n.startswith(".voxg-")]
        assert leftovers == []
        monkeypatch.undo()


def test_service_restart_matches_last_committed(path_factory):
    path = path_factory()
    pcfg = PersistenceConfig(path=path)

    svc = OccupancyService(ServiceConfig(persistence=pcfg))
    svc.submit(
        ObservationBatch(
            (Observation((1, 0, 0), ObservationKind.OCCUPIED, 1.0),) * 5,
            sequence=10,
            timestamp=100.0,
        )
    )
    svc.submit(
        ObservationBatch(
            (Observation((2, 0, 0), ObservationKind.OCCUPIED, 1.0),) * 5,
            sequence=11,
            timestamp=101.0,
        )
    )
    expected = svc.stats()
    expected_cells = svc.grid.snapshot_cells()

    # 模拟重启：全新服务实例从同一文件恢复
    svc2 = OccupancyService(ServiceConfig(persistence=pcfg))
    assert svc2.stats() == expected
    restored_cells = svc2.grid.snapshot_cells()
    assert set(expected_cells) == set(restored_cells)
    assert svc2.trace_line((0, 0, 0), (3, 0, 0)).blocked_coord == (1, 0, 0)
    assert svc2.get((2, 0, 0)).occupied

    # 恢复后继续工作：清除并重新占用
    svc2.submit(
        ObservationBatch(
            (Observation((1, 0, 0), ObservationKind.CLEAR),),
            sequence=12,
            timestamp=102.0,
        )
    )
    svc3 = OccupancyService(ServiceConfig(persistence=pcfg))
    assert svc3.get((1, 0, 0)) is None
    assert svc3.get((2, 0, 0)).occupied


def test_restart_preserves_staleness_watermark(path_factory):
    """重启后旧批次仍然过期：序号水位随栅格一并恢复。"""
    path = path_factory()
    scfg = StalenessConfig(mode=StalenessMode.DROP, max_sequence_lag=2)
    svc = OccupancyService(
        ServiceConfig(staleness=scfg, persistence=PersistenceConfig(path=path))
    )
    svc.submit(
        ObservationBatch(
            (Observation((0, 0, 0), ObservationKind.OCCUPIED),),
            sequence=50,
            timestamp=50.0,
        )
    )
    svc2 = OccupancyService(
        ServiceConfig(staleness=scfg, persistence=PersistenceConfig(path=path))
    )
    r = svc2.submit(
        ObservationBatch(
            (Observation((1, 0, 0), ObservationKind.OCCUPIED),),
            sequence=1,
            timestamp=1.0,
        )
    )
    assert not r.committed
    assert svc2.get((1, 0, 0)) is None
