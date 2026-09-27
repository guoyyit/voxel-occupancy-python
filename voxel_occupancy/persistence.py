"""崩溃安全的栅格持久化与恢复。

文件格式（v1，小端）：
    magic   = b"VOXG"
    version = uint16 (=1)
    payload = zlib 压缩的 JSON：
              {"version":1,"sequence":int,"batches":int,
               "cells":[[x,y,z,log_odds,last_updated], ...]}
    crc32   = uint32（对 magic..payload 全部字节的校验和）
    length  = uint64（magic..crc 的总字节数，用于定位文件尾并检测截断）

写入流程：同目录临时文件 -> fsync -> os.replace 原子替换 -> fsync 目录。
崩溃只会发生在 replace 之前（旧文件完好）或之后（新文件完整），
不会留下损坏或半写入的正式文件。
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib

from .config import PersistenceConfig
from .grid import VoxelGrid
from .types import CellState

MAGIC = b"VOXG"
VERSION = 1
_HEADER = struct.Struct("<4sH")
_TRAILER = struct.Struct("<IQ")  # crc32, length(of body+crc)


class CorruptGridError(RuntimeError):
    """持久化文件损坏（截断 / 魔数不符 / 校验失败 / 载荷非法）。"""


def _fsync_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # 某些平台不允许 fsync 目录，尽力而为
        pass
    finally:
        os.close(fd)


def _encode(grid: VoxelGrid) -> bytes:
    cells = grid.snapshot_cells()
    payload_obj = {
        "version": VERSION,
        "sequence": grid.committed_sequence,
        "batches": grid.committed_batches,
        "cells": [
            [coord[0], coord[1], coord[2], state.log_odds, state.last_updated]
            for coord, state in sorted(cells.items())
        ],
    }
    payload = zlib.compress(
        json.dumps(payload_obj, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        level=6,
    )
    body = _HEADER.pack(MAGIC, VERSION) + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + _TRAILER.pack(crc, len(body) + 4)


def _decode(blob: bytes) -> tuple[dict[tuple[int, int, int], CellState], int, int]:
    if len(blob) < _HEADER.size + _TRAILER.size:
        raise CorruptGridError("file too short / truncated")

    crc, length = _TRAILER.unpack(blob[-_TRAILER.size :])
    if length != len(blob) - 8:
        raise CorruptGridError("length field mismatch (partial write?)")

    body = blob[:-_TRAILER.size]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise CorruptGridError("checksum mismatch")

    magic, version = _HEADER.unpack(body[: _HEADER.size])
    if magic != MAGIC:
        raise CorruptGridError("bad magic")
    if version != VERSION:
        raise CorruptGridError(f"unsupported format version: {version}")

    try:
        payload = json.loads(zlib.decompress(body[_HEADER.size :]))
        cells: dict[tuple[int, int, int], CellState] = {}
        for entry in payload["cells"]:
            x, y, z, log_odds, last_updated = entry
            coord = (int(x), int(y), int(z))
            cells[coord] = CellState(
                coord=coord,
                log_odds=float(log_odds),
                last_updated=float(last_updated),
            )
        return cells, int(payload["sequence"]), int(payload["batches"])
    except (CorruptGridError, KeyError, TypeError, ValueError, zlib.error) as exc:
        raise CorruptGridError(f"invalid payload: {exc}") from exc


def save_grid(grid: VoxelGrid, config: PersistenceConfig) -> None:
    """把栅格当前已提交状态原子地写入持久化文件。"""
    blob = _encode(grid)
    directory = os.path.dirname(os.path.abspath(config.path)) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=".voxg-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            if config.fsync:
                os.fsync(f.fileno())
        os.replace(tmp_path, config.path)
        if config.fsync:
            _fsync_dir(config.path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_grid(grid: VoxelGrid, config: PersistenceConfig) -> bool:
    """从持久化文件恢复栅格。

    返回 True 表示成功恢复；False 表示文件不存在（保持空栅格）。
    校验失败抛 CorruptGridError，并按配置把坏文件改名备份。
    """
    if not os.path.exists(config.path):
        return False
    try:
        with open(config.path, "rb") as f:
            blob = f.read()
        cells, sequence, batches = _decode(blob)
    except CorruptGridError:
        if config.corrupt_backup_suffix:
            backup = config.path + config.corrupt_backup_suffix
            try:
                os.replace(config.path, backup)
            except OSError:
                pass
        raise

    grid.load_cells(cells, sequence=sequence, batches=batches)
    return True
