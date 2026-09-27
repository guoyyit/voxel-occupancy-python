# Local 3D Voxel Occupancy Service

一个完全本地运行的三维体素占用栅格服务：以离散栅格维护动态环境的可查询占用状态，
支持概率融合、带范围的区域查询、直线可达查询、并发安全、过期策略、崩溃恢复，以及大
规模未观测区域的有界内存表示。观测通过**进程内内存管道**或**回环脚本**注入，不依
赖任何外部设备或在线服务；HTTP 层仅为本机可选接口。

- Python 3.14，仅依赖 FastAPI（标准库实现线程、持久化、空间算法）
- 纯整数体素坐标 `(x, y, z)`，稀疏存储，未观测单元零内存

## 快速开始

```bash
uv sync
uv run pytest                      # 40 个自动化测试
uv run python examples/demo.py     # 端到端本地演示（含"重启恢复"）
uv run uvicorn main:app --port 8000   # 启动 HTTP 服务（默认纯内存）
```

## 架构与模块

| 模块 | 职责 |
| --- | --- |
| `voxel_occupancy/types.py` | 数据类型：观测、快照、栅格/持久化配置、枚举 |
| `voxel_occupancy/fusion.py` | log-odds 概率融合原语与过期/降权策略 |
| `voxel_occupancy/grid.py` | 线程安全稀疏栅格，原子提交，先验裁剪 |
| `voxel_occupancy/queries.py` | 立方区域枚举 + 3D Bresenham 直线遍历 |
| `voxel_occupancy/persistence.py` | 临时文件 + fsync + 原子 rename 的崩溃安全快照 |
| `voxel_occupancy/feeds.py` | `MemoryFeed` 内存管道、`LoopbackFeed` 回环脚本 |
| `voxel_occupancy/service.py` | 事务编排：准备 → 原子合并 → 持久化 → 一致查询 |
| `voxel_occupancy/api.py` | FastAPI HTTP 接口 |

## 状态模型

每个**被观测过**的体素存储一条记录：

```
(log_odds, last_update_timestamp, update_count)
```

- 占用信念以 log-odds 表示：`L = ln(p / (1 - p))`。
- 一次独立二值观测的融合是加法：`L_new = clamp(L_old + (logit(p_hit) - L_prior) * w)`。
- log-odds 被夹在 `[log_odds_min, log_odds_max]`（默认 ±8，约对应概率
  `0.000336 ~ 0.999664`）。因此无论重复、乱序还是过期观测，概率**永远严格在 (0,1)**。
- 离散状态由阈值导出：
  - `p >= occupied_threshold`（默认 0.7）→ `occupied`
  - `p <= free_threshold`（默认 0.3）→ `free`
  - 其余 → `unknown`
- **未观测单元不占内存**，读取时返回先验 `p = 0.5 / unknown`。
- 当融合后的 `|L - L_prior| <= prune_log_odds_eps`（默认 1e-3）时，单元被删除回到
  先验——这保证长期运行中内存只随"当前有信息量的单元"增长；被清除的单元之后仍可被
  新观测重新占用（删除仅是稀疏化，不是墓碑）。

### 时间与过期语义

每个观测带单调整数 `timestamp` 与可选 `weight`（观测自身置信度，默认 1）。融合时以
"该单元已提交的最高时间戳"为水位线，策略可配置：

- `REJECT`（默认）：时间戳必须**严格新于**已提交水位线才接受；相等视为重复丢弃，更
  旧视为过期丢弃。同一批次内乱序到达的旧观测同样被丢弃。
- `DOWNWEIGHT`：相等时间戳按全权重融合；更旧的观测按
  `downweight_decay ** 年龄` 降权；年龄超过 `expiry_horizon` 时直接丢弃。
- `IGNORE_TIME`：忽略时间戳，全部按全权重融合。

权重最终生效为 `obs.weight × 时间权重`。相同输入 + 相同策略给出**逐位一致、可复现**
的结果（快照还按坐标排序序列化）。

## 事务、查询一致性与并发

- `submit(batch)` 是单事务：先在锁外把批次归约为每体素增量，再**一次性**获取栅格锁，
  完成合并与持久化后才推进提交序号。
- 查询（`get_cell` / `query_region` / `query_ray`）在同一把锁下对同一个映射取快照执
  行，因此只会看到某个**完整已提交版本**，永远看不到半批更新。
- 并发更新在整批边界上线性化：结果等价于某种串行顺序；log-odds 加法保证不丢更新、不
  越界（有测试 `tests/test_concurrency.py` 验证计数、版本号单调、概率有界）。
- `query_region(center, max_range)`：在切比雪夫距离 ≤ `max_range` 的闭区间立方内，遍历
  **已存储**单元做包围盒过滤返回（O(已知单元数)，而非 O(range³)；10 万单元下约 5ms）。
- `query_ray(start, end)`：3D Bresenham 整数直线（含两端、无重复），线上任何单元为
  `occupied` 即不可达，返回首个阻挡单元与完整路径；未观测/空闲单元不阻挡。

## 崩溃安全与恢复

启用持久化（传入 `PersistenceConfig(path=...)`）后，每次提交在锁内执行：

1. 序列化整快照到同目录临时文件（magic + 版本 + 序号 + 按坐标排序的单元 + CRC32）；
2. `flush + fsync` 临时文件；
3. `os.replace` 原子替换正式文件（POSIX/Windows 均原子）；
4. fsync 目录，保证 rename 本身落盘。

任意时刻崩溃，只会留下**上一份完整文件或新的完整文件**；残留 `.tmp` 在恢复时忽略；
正式文件若被撕裂/损坏会由 CRC 检测并抛出 `CorruptStateError`，绝不静默载入坏状态。
`OccupancyService.start(...)` 启动即恢复，恢复后查询结果与崩溃前最后提交一致（序号、
log-odds、状态、计数全部保留）。

## 配置项

### `GridConfig`

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `log_odds_min` / `log_odds_max` | -8.0 / 8.0 | log-odds 夹断边界，保证概率有界 |
| `occupied_threshold` | 0.7 | ≥ 该概率判为 occupied |
| `free_threshold` | 0.3 | ≤ 该概率判为 free |
| `prior_log_odds` | 0.0 | 先验 log-odds（0 ⇔ p=0.5） |
| `prune_log_odds_eps` | 1e-3 | 回到先验附近即裁剪，控制内存上界 |
| `max_observations_per_commit` | 1_000_000 | 单批观测数上限（保留配置） |
| `expiry_policy` | `REJECT` | `REJECT` / `DOWNWEIGHT` / `IGNORE_TIME` |
| `downweight_decay` | 0.5 | DOWNWEIGHT 下 `decay ** age` 的底数 |
| `expiry_horizon` | `None` | 年龄超过该值即使降权也丢弃；`None` 不限制 |

### `PersistenceConfig`

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `path` | `voxel_state.db` | 快照文件路径 |
| `fsync` | `True` | 每次提交是否 fsync 文件与目录（关闭可提性能但降低持久性保证） |
| `expected_version` | 1 | 期望的磁盘格式版本，不兼容则报错 |

## HTTP 接口

| 方法/路径 | 说明 |
| --- | --- |
| `GET /health` | 存活检查与当前提交版本号 |
| `POST /observations` | 原子提交一批观测，返回 `{sequence, applied, rejected_expired}` |
| `GET /cells/{x}/{y}/{z}` | 单体素状态（未观测返回 prior/unknown） |
| `GET /region?x&y&z&range` | 闭区间立方区域查询 |
| `GET /ray?x0&y0&z0&x1&y1&z1` | 直线可达性，返回是否可达、首个阻挡体素、完整路径 |
| `POST /recover` | 从持久化快照重新载入 |

观测体：`{x,y,z,prob_hit(0,1),timestamp:int,weight?:float}`。
`main.py` 默认创建纯内存服务；需要持久化时用 `create_app(persistence_config=...)`。

## 测试覆盖

`uv run pytest` 共 40 个用例，覆盖：

- 概率有界性、重复证据收敛、矛盾证据回摆（`test_fusion.py`）
- 三种过期策略、批次内乱序、重复丢弃、降权与水平截止、可复现性
- 稀疏内存、log-odds 夹断、先验裁剪与再占用（`test_grid_queries.py`）
- 区域查询、Bresenham 直线、占用阻挡判定
- 重复/乱序/过期批次、查询只反映最后提交（`test_service_updates.py`）
- 多线程并发：无丢失更新、版本号 1..N、概率不越界、快照一致性
- 重启恢复、原子写无残留、崩溃保留旧快照、撕裂文件 CRC 检测（`test_recovery.py`）
- 内存管道/回环脚本、HTTP 接口与持久化恢复（`test_feeds_api.py`）
