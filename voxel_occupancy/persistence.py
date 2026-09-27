"""Crash-safe persistence with atomic snapshot replacement.

On-disk layout (little-endian)::

    magic      4 bytes  b"VOXG"
    version    uint8
    sequence   uint64
    n_cells    uint64
    cell[n]    i64 x, i64 y, i64 z, f64 log_odds, i64 last_ts, u64 count
    crc32      uint32  over every preceding byte

Durability protocol per commit:
  1. serialize the full snapshot to ``<path>.tmp`` in the same directory
  2. flush + fsync the temp file
  3. ``os.replace`` it onto the final path (atomic rename on POSIX/Windows)
  4. fsync the directory so the rename itself is durable

A crash at any point leaves either the previous complete file or the new
complete file; a leftover ``.tmp`` is ignored on recovery. The trailing CRC
lets recovery detect and reject a genuinely corrupt/truncated file rather
than silently restoring bad state. Cell order is coordinate-sorted, so the
same state always serializes to identical bytes.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zlib

from .types import Cell, PersistenceConfig

MAGIC = b"VOXG"
VERSION = 1
_TMP_SUFFIX = ".tmp"

# magic(4) version(1) sequence(8) n(8) ... crc(4)
_HEADER = struct.Struct("<4sBQQ")
# i64 x, i64 y, i64 z, f64 log_odds, i64 last_ts, u64 count -> 48 bytes.
_CELL = struct.Struct("<qqqdqQ")
_CRC = struct.Struct("<I")


class CorruptStateError(RuntimeError):
    """Raised when durable state fails integrity validation."""


class PersistenceStore:
    """Durable voxel state with atomic full-snapshot replacement."""

    def __init__(self, config: PersistenceConfig) -> None:
        self.config = config

    # ----- serialization -------------------------------------------------

    @staticmethod
    def _encode(
        sequence: int, cells: dict[Cell, tuple[float, int, int]]
    ) -> bytes:
        body = bytearray(_HEADER.pack(MAGIC, VERSION, sequence, len(cells)))
        for (x, y, z) in sorted(cells):
            log_odds, last_ts, count = cells[(x, y, z)]
            body += _CELL.pack(x, y, z, log_odds, last_ts, count)
        body += _CRC.pack(zlib.crc32(bytes(body)) & 0xFFFFFFFF)
        return bytes(body)

    @staticmethod
    def _decode(
        raw: bytes,
    ) -> tuple[int, dict[Cell, tuple[float, int, int]]]:
        head_size = _HEADER.size
        if len(raw) < head_size + _CRC.size:
            raise CorruptStateError("snapshot truncated")
        magic, version, sequence, n = _HEADER.unpack_from(raw, 0)
        if magic != MAGIC:
            raise CorruptStateError("bad magic; not a voxel snapshot")
        if version != VERSION:
            raise CorruptStateError(
                f"incompatible snapshot version {version}, expected {VERSION}"
            )
        data_end = head_size + n * _CELL.size
        if len(raw) != data_end + _CRC.size:
            raise CorruptStateError("snapshot length does not match cell count")
        (expected_crc,) = _CRC.unpack_from(raw, data_end)
        actual_crc = zlib.crc32(raw[:data_end]) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise CorruptStateError("CRC mismatch; snapshot is corrupt")
        cells: dict[Cell, tuple[float, int, int]] = {}
        off = head_size
        for _ in range(n):
            x, y, z, log_odds, last_ts, count = _CELL.unpack_from(raw, off)
            cells[(x, y, z)] = (log_odds, last_ts, count)
            off += _CELL.size
        return sequence, cells

    # ----- durability ----------------------------------------------------

    def save(
        self, sequence: int, cells: dict[Cell, tuple[float, int, int]]
    ) -> None:
        """Atomically persist the given snapshot; crash-safe at every step."""
        path = self.config.path
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload = self._encode(sequence, cells)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".voxel-", suffix=_TMP_SUFFIX, dir=directory
        )
        try:
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(payload)
                    fh.flush()
                    if self.config.fsync:
                        os.fsync(fh.fileno())
            except BaseException:
                # fdopen owns fd even on error; ensure temp is removed.
                _unlink_quiet(tmp_path)
                raise
            os.replace(tmp_path, path)
            if self.config.fsync:
                _fsync_dir(directory)
        except BaseException:
            _unlink_quiet(tmp_path)
            raise

    def load(self) -> tuple[int, dict[Cell, tuple[float, int, int]]] | None:
        """Recover the last durable snapshot, or ``None`` if none exists."""
        path = self.config.path
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            raw = fh.read()
        return self._decode(raw)


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _fsync_dir(directory: str) -> None:
    if not hasattr(os, "O_DIRECTORY"):  # pragma: no cover - non-POSIX
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= getattr(os, "O_DIRECTORY")
    dir_fd = os.open(directory, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
