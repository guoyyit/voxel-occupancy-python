# 本地三维体素占用栅格服务（voxel-occupancy-python）

一个完全本地运行的三维占用概率栅格服务，用于维护动态环境下可查询的占用状态。
观测输入由**内存管道 / 回环接口**或 **HTTP** 模拟，不依赖任何外部设备或服务。

- Python 3.14，仅依赖 FastAPI；持久化只使用标准库。
- 启动服务：`uv run uvicorn main:app`（默认无持久化文件）。
- 运行测试：`uv run pytest`。

## 1. 状态模型（概率融合）

栅格以三维整数坐标 `(x, y, z)` 稀疏存储体素，未观测体素不占内存。
每个已存储体素维护**对数几率（log-odds）**：

```
L = log(p / (1 - p))，先验 p0 = 0.5 => L0 = 0
```

观测融合（`weight >= 0` 为观测权重）：

| 观测类型 | 更新 |
|---|---|
| `OCCUPIED` | `L += occupied_log_odds * weight`（默认 +0.85） |
| `FREE` | `L += free_log_odds * weight`（默认 -0.45） |
| `CLEAR` | 直接删除体素，回到未观测状态；之后可被再次占用 |

**有界性**：每次更新后 `L` 都被夹在 `[clamp_min, clamp_max]`（默认 [-3.5, 3.5]），
因此占用概率始终严格落在 `(0, 1)`，重复观测、乱序观测、过期降权观测都不可能越界。

**有界内存**：

1. 稀疏表示——只存储被观测过的体素，未观测区域零开销；
2. 体素按 16×16×16 分块存放，区域查询只扫描与半径包围盒相交的分块；
3. 先验死区剪枝——更新后若 `|L| <= prune_deadband`（默认 1e-3），
   说明该体素与先验 0.5 不可区分，直接删除并在分块为空时释放分块。
   冲突观测相互抵消后内存会自动回收（见 `test_memory_reclaimed_after_cancel`）。

占用判定：`p(L) >= occupied_threshold`（默认 0.5）。未观测、已清除、已剪枝体素一律视为**不占用**、不阻挡直线。

## 2. 提交、过期策略与幂等

观测以 `ObservationBatch`（一批 `Observation`）为单位**原子提交**：
要么整批生效，要么整批不生效。批次可带：

- `sequence`：单调递增批次序号；
- `timestamp`：观测时间；
- `idempotency_key`：幂等键，重复提交同一批次只生效一次（LRU 缓存，容量 `dedup_cache_size`）。

过期判定（`staleness.decide_staleness`，纯函数，可复现），满足任一判据即过期：

- 序号滞后：`committed_sequence - batch.sequence > max_sequence_lag`（0 表示不启用）；
- 年龄过大：`latest_timestamp - batch.timestamp > max_age_seconds`（0 表示不启用）。

策略 `StalenessMode`：

| 策略 | 行为 |
|---|---|
| `ACCEPT`（默认） | 忽略过期判据，全部接受 |
| `DROP` | 过期批次整体丢弃，栅格不变，返回 `committed=False` 及原因 |
| `DOWNWEIGHT` | 过期批次仍融合，但权重按超出量线性衰减到 `downweight_min` |

融合运算不含随机数、不依赖墙钟，**同一输入 + 同一策略必然得到同一结果**
（见 `test_same_input_same_policy_reproducible`）。

> 说明：序号水位 `committed_sequence` 随栅格持久化，重启后旧批次仍会被判为过期；
> 年龄水位 `latest_timestamp` 为进程内状态，重启后从 0 重新建立。

## 3. 查询模型与一致性

- `get(coord)`：单体素视图（未观测返回 `None`）；
- `query_region(center, max_range, occupied_only=False)`：球心 + 最大**欧氏**半径内的已观测体素；
- `trace_line(start, end)`：三维整数 Bresenham 直线，返回经过路径、
  是否可达以及第一个阻挡体素（不含起点、含终点）；
- `stats()`：已存储体素数、已提交序号与批次数。

**快照隔离**：栅格内部为「顶层字典 + 16³ 分块」的两级结构，提交时在副本上完成
全部修改后一次性发布（copy-on-write）；查询持同一把锁并绑定一个已提交快照。
因此任何查询都**看不到部分更新**，结果与某一次已提交状态一致。

需要在一次查询中组合多个读取（例如扫描多个体素）时，使用一致性读视图：

```python
with service.read() as view:
    if view.is_occupied(a) or view.is_occupied(b):
        ...
    line = view.trace_line(start, end)
```

## 4. 并发语义

- 一把 `RLock` 串行化所有提交：并发更新同一批体素的结果**等价于某一合法串行顺序**
  （见 `test_concurrent_result_equivalent_to_some_serial_order`）；
- 不会出现丢失更新（无夹逼时 N 个线程的累加结果恰好等于总和，见
  `test_concurrent_increments_no_lost_update_and_bounded`）；
- 概率永不越界（`test_concurrent_mixed_updates_stay_bounded`）；
- 同一幂等键的并发提交只有一次生效（`test_concurrent_idempotent_submit_applies_once`）；
- 读视图与写互斥但读之间共享，查询在并发提交下不会崩溃或读到半批数据。

## 5. 持久化与崩溃恢复

文件格式（小端，`persistence.py`）：

```
magic "VOXG" | uint16 version(=1) | zlib(JSON payload) | uint32 crc32 | uint64 length
```

payload 记录 `sequence / batches / cells([x,y,z,log_odds,last_updated])`。

写入流程：同目录临时文件 → `flush + fsync` → `os.replace` 原子替换 → `fsync` 目录。
崩溃只可能发生在替换前（旧文件完好）或替换后（新文件完整），
**不会留下损坏或半写入的正式文件**；临时文件在异常时清理。

读取时校验长度与 CRC32，截断 / 翻转字节均抛 `CorruptGridError`
并按 `corrupt_backup_suffix` 把坏文件改名备份。服务启动时自动恢复
（`restore_on_start`），恢复后查询结果与崩溃前最后一次成功提交一致，
并可继续清除/重新占用（`test_service_restart_matches_last_committed`）。

## 6. 观测输入：内存管道与回环

不依赖外部设备：

```python
from voxel_occupancy.transport import MemoryPipe, LoopbackTransport
pipe = MemoryPipe(maxsize=128)          # 有界 FIFO，线程安全
loop = LoopbackTransport(pipe)
loop.send_batch(batch)                  # 生产者（传感器模拟）
results = loop.pump(service.submit)     # 消费者取出并提交
# 或：service.serve_pipe(pipe) / loop.stream(service.submit)
```

## 7. HTTP 接口（FastAPI）

| 方法/路径 | 说明 |
|---|---|
| `POST /batches` | 提交一批观测（body：`observations[{x,y,z,kind,timestamp,weight}], sequence, timestamp, idempotency_key`） |
| `GET /voxels/{x}/{y}/{z}` | 单体素查询（未观测返回 `observed=false, occupied=false, probability=0.5`） |
| `GET /regions?x&y&z&range&occupied_only` | 最大范围区域查询 |
| `GET /lines?x0..z1` | 起止点直线可达查询 |
| `GET /stats` | 统计信息 |
| `POST /flush` | 显式落盘（未配置持久化时返回 400） |

## 8. 配置项（`config.py`）

| 配置 | 字段（默认值） |
|---|---|
| `GridConfig` | `resolution=1.0`、`occupied_log_odds=0.85`、`free_log_odds=-0.45`、`clamp_min=-3.5`、`clamp_max=3.5`、`occupied_threshold=0.5`、`prune_deadband=1e-3` |
| `StalenessConfig` | `mode=ACCEPT`、`max_sequence_lag=0`、`max_age_seconds=0.0`、`downweight_min=0.1` |
| `PersistenceConfig` | `path="voxel_grid.dat"`、`fsync=true`、`restore_on_start=true`、`corrupt_backup_suffix=".corrupt"` |
| `ServiceConfig` | `grid`、`staleness`、`persistence=None`、`dedup_cache_size=4096`、`origin=(0,0,0)` |

## 9. 代码结构

```
voxel_occupancy/
  types.py        # 坐标/观测/批次/结果等数据类型
  config.py       # 上述全部配置项
  staleness.py    # 过期/乱序：接受、丢弃、降权（纯函数）
  grid.py         # 线程安全概率栅格：融合、COW 快照、剪枝、区域/直线查询、read() 视图
  persistence.py  # 原子写 + CRC + 崩溃恢复
  transport.py    # MemoryPipe / LoopbackTransport
  service.py      # OccupancyService：幂等 + 策略 + 提交 + 持久化 + 管道消费
  server.py       # FastAPI 应用（create_app 可注入服务）
tests/            # 50 个用例：融合/有界、查询一致性、重复乱序、并发、恢复、有界内存、传输、HTTP
```
