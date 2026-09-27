"""本地三维体素占用栅格服务。

模块组成：
- types: 核心数据类型
- config: 配置项
- staleness: 过期观测策略
- grid: 线程安全的概率栅格（含事务性更新）
- persistence: 崩溃安全的持久化与恢复
- transport: 内存管道 / 回环观测输入
- service: 占用服务门面
- server: FastAPI HTTP 接口
"""
