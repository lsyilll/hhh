# 观叶研究与采集平台

保留原 AgriChat 农业问答、语音唤醒“小叶小叶”和人员入口，增加 React 研究采集页面、Spring 项目权限、异步模型任务、设备通信、数据版本、分析数仓和运行监控。本目录是可独立构建的平台源码；原 AgriChat 主程序与权重由原目录管理。

## 已实际验证（2026-10-21）

- PostgreSQL、Redis、SeaweedFS / S3、Spring Security、Celery：真实接口、项目角色、对象归属、质检、Parquet 审计及临时数据清理。
- DINOv2 B/14、L/14：实际队列任务生成 768 / 1024 维归一化向量与 FAISS 索引；SAM 2.1：明确选框得到 3 个叶片分割候选。MLflow：PostgreSQL 参数/指标及 S3 文件读写。
- MQTT 5 / Mosquitto：双向 TLS、设备心跳入库、缺少证书及其他设备主题拒绝。使用软件设备验证，未冒充树莓派或摄像头实测。
- OpenTelemetry → Prometheus → Grafana：真实 Celery 指标、collector 链路数据、Spring 认证采集和 4 面板仪表板数据查询。
- Kafka → Flink SQL → Doris：真实发件箱补发、事件去重、一分钟事件时间窗口；重复投递后完成数仍为 2，ODS/DWD/DWS/ADS 查询一致。
- Airflow：匿名 API 401，认证触发 DAG，经调度器实际执行，生成并读回限定项目的 Parquet 文件。
- Git / DVC：快照、审计和不重叠划分，经 S3 上传后在没有 DVC 缓存的新 Git 检出目录恢复，清单与划分字节一致。
- Nginx：HTTPS 反向代理、双向 TLS 设备心跳到 Spring/PostgreSQL；公开端口不接受设备接口，伪造网关头被拒绝。仅为测试请求显式信任开发 CA，没有导入系统根证书。
- Qwen3.5-4B：公开官方权重固定提交并校验 SHA256；真实 CPU 队列任务及 UTF-8 报告读回通过。96 token 测试约 247 秒，输出被截断；不是完整报告或诊断准确率验收。页面默认最多 256 token，仍需人工核查内容。
- ONNX Runtime / TensorRT 10.8：RTX 3060 上的 8×768 受控数值输入比较，最大绝对误差约 2.38e-7。验证对象是数值分类头，不代表整套 AgriChat 已加速或病害识别准确率。
- Python 27 项、JUnit 7 项、Playwright 3 项通过；TypeScript 与 Vite 生产构建通过。GitHub Actions 的实际远端结果另行记录。

临时项目、对象、任务缓存与证书已清理。Kafka 技术验收事件保留至正常主题过期，不删除共享主题；MLflow 临时实验和 run 按官方 API 软删除。没有搬迁原账号、人脸、音频和个人历史。

## 使用和内存切换

32GB 内存采用分组运行，不能把所有重型服务同时常驻。默认新端口仅监听本机回环地址。使用原工作台菜单“研究与采集平台”进入页面。

| 入口（本目录） | 作用 |
| --- | --- |
| `run_remaining_stack.bat` | 启动并验证监控与软件设备通信 |
| `run_analytics_stack.bat` | 启动 Kafka / Flink / Doris；暂停研究模型 |
| `verify_analytics_stack.bat` | 只刷新 Doris 写入服务并验证分析链路 |
| `run_offline_stack.bat` | 启动并验证 Airflow；暂停研究模型和分析服务 |
| `run_qwen_report.bat` | 复用模型镜像，启动并验证 Qwen；暂停 Airflow、分析服务和 MLflow |
| `scripts/check_browser.py` | 生产构建和独立 Chromium 自动测试；不访问用户浏览器资料 |
| `scripts/check_gateway.py` | 临时 HTTPS/mTLS 验收；占用端口时拒绝替换现有服务 |

开发时先用 `scripts/init_config.py` 创建私有配置，再构建基础容器。公开模型容器使用 `compose.public.yaml` 与 `Dockerfile.public`，依赖已构建的 `guanye-worker` 镜像。扩展阶段使用 `compose.yaml` + `compose.extended.yaml`，挂载受版本控制的 worker / model_service 源码；禁止将这些挂载目录开放给不可信使用者。

原 AgriChat 仍由原目录的 `start_agrichat.bat` 启动；公开研究服务入口为 `start_public_research.bat`。关闭网页不会释放容器资源；使用原 `stop_platform.bat` 停止服务，保留数据库和存储卷。已安装的权重和 SDK 会继续占用磁盘。

## 每项技术用于哪里

| 技术 | 在项目中的用途 | 当前条件 |
| --- | --- | --- |
| React 19 | 研究与采集平台的页面与组件 | 已接入；实际运行状态以页面探测为准 |
| TypeScript | 项目、样本、任务及设备数据类型 | 已接入；实际运行状态以页面探测为准 |
| Vite | 前端开发与生产构建 | 已接入；实际运行状态以页面探测为准 |
| Ant Design | 表格、表单、角色操作、二维码及交互反馈 | 已接入；实际运行状态以页面探测为准 |
| TanStack Query | 请求缓存、加载状态、任务刷新和失效更新 | 已接入；实际运行状态以页面探测为准 |
| HTTPS API | 统一入口、登录 cookie 与设备双向 TLS 上传 | 已接入；实际运行状态以页面探测为准 |
| SSE | 原图像问答流式回答及平台任务更新 | 已接入；实际运行状态以页面探测为准 |
| ECharts | 真实样本趋势与原运行概览 | 已接入；实际运行状态以页面探测为准 |
| Java 21 | 业务服务运行时 | 已接入；实际运行状态以页面探测为准 |
| Spring Boot 4.x | 项目、设备、样本、任务、复核与审计接口 | 已接入；实际运行状态以页面探测为准 |
| Spring Security | 短时签名身份、业务/设备/监控 scope 与项目角色 | 已接入；实际运行状态以页面探测为准 |
| Python / FastAPI | 保留原农业问答，增加同源平台适配与独立模型服务 | 已接入；实际运行状态以页面探测为准 |
| PyTorch / Transformers | 原 AgriChat 和隔离的新模型加载及推理 | 已接入；实际运行状态以页面探测为准 |
| OpenCV / Pillow / NumPy | 质检、图像解码、裁剪、特征与采集终端处理 | 已接入；实际运行状态以页面探测为准 |
| DINOv2 ViT-B/14 | 公开默认视觉特征与 FAISS 项目索引 | 无需申请访问，官方公开权重 |
| DINOv2 ViT-L/14 | 公开视觉骨干对比实验 | 无需申请访问，官方公开权重 |
| SAM 2.1 Small | 根据人工框选范围生成分割候选，不处理文字提示 | 无需申请访问，官方公开权重 |
| DINOv3 ViT-B/16 | 授权后可选的视觉特征与 FAISS 项目索引 | 官方授权及权重待本人完成 |
| DINOv3 ViT-L/16 | 视觉骨干对照实验 | 官方授权及权重待本人完成 |
| MQTT 5 / Mosquitto | 设备心跳、证书身份及命令主题隔离 | MQTT 5 双向 TLS、心跳入库及主题隔离已通过；实体设备待验收 |
| TLS | 网页 HTTPS、设备证书和本地开发 CA | HTTPS / 双向 TLS 已实际验收；浏览器及跨设备证书信任仍需管理员配置 |
| PostgreSQL / SQL | 项目角色、设备、任务、样本、复核、审计和事务发件箱 | 已接入；实际运行状态以页面探测为准 |
| Redis / Celery | 任务排队、结果缓存、进度与失败重试 | 已接入；实际运行状态以页面探测为准 |
| S3 / SeaweedFS | 植物原图、遮罩、特征索引、数据清单和报告 | 已接入；实际运行状态以页面探测为准 |
| FAISS | 原个人案例库及 DINOv2 / DINOv3 项目相似检索 | 已接入；实际运行状态以页面探测为准 |
| MLflow | 模型参数、数据哈希、评估指标与实验版本 | 已接入；实际运行状态以页面探测为准 |
| DVC / Git | 数据清单、划分、数据版本和代码版本关联 | 已接入；实际运行状态以页面探测为准 |
| Linux / WSL2 | Windows 上的 Linux 容器运行环境 | 需启用 WSL2 / Linux 容器环境 |
| Node.js | React/TypeScript/Vite 构建 | 已接入；实际运行状态以页面探测为准 |
| uv | 隔离的 Python 环境和依赖安装脚本 | 已接入；实际运行状态以页面探测为准 |
| Maven | Java 构建及 JUnit 检查 | 已接入；实际运行状态以页面探测为准 |
| Docker Compose | 按 core/devices/analytics/warehouse/research/offline/monitoring/models 分组启动 | 需安装 Docker Desktop 并启动 Linux 引擎 |
| Nginx | 同源 HTTPS、长连接、设备双向 TLS 反向代理 | 已接入；实际运行状态以页面探测为准 |
| pytest | 权限适配、签名、质量与任务输入边界检查 | 已接入；实际运行状态以页面探测为准 |
| JUnit | 项目越权、设备审批、对象归属及标签确认测试 | 已接入；实际运行状态以页面探测为准 |
| Playwright | 页面入口与响应式业务流程的浏览器测试 | 已接入；实际运行状态以页面探测为准 |
| GitHub Actions | 前端构建、Java/Python 检查与 Linux 集成检查配置 | 工作流已准备，目标 lsyilll/hhh；远端运行待完成 |
| SAM3 | 文本/框提示分割候选；由人复核后用于标注 | 官方授权及权重待本人完成，建议独占 GPU 模式 |
| Kafka | 可靠业务事件主题，发件箱补发与流式消费 | 已接入；实际运行状态以页面探测为准 |
| Flink / Flink SQL | 事件去重、事件时间水位线和一分钟窗口统计 | 已接入；实际运行状态以页面探测为准 |
| Doris / SQL | ODS 原始事件、DWD 任务明细、DWS 窗口聚合、ADS 日报 | 已接入；实际运行状态以页面探测为准 |
| Airflow / Python / SQL | 每日样本快照、历史质量任务及后续重算入口 | 已接入；实际运行状态以页面探测为准 |
| Parquet | 可重复审计的数据快照与数据清单导出 | 已接入；实际运行状态以页面探测为准 |
| OpenTelemetry | 请求链路与任务观测接口，不采集人脸或音频正文 | 已接入；实际运行状态以页面探测为准 |
| Prometheus | 采集业务和算法运行指标 | 已接入；实际运行状态以页面探测为准 |
| Grafana | 耗时、失败率和任务状态仪表板 | 已接入；实际运行状态以页面探测为准 |
| ONNX Runtime | 分类头导出与真实测试样本的一致性校验 | 已接入；实际运行状态以页面探测为准 |
| TensorRT | 独占 GPU 上的数值分类头优化及与 ORT 的一致性比较 | RTX 3060 数值头一致性实测通过；独立 Windows SDK；未导出整个 AgriChat |
| Qwen3.5-4B | 根据图像和已有证据整理报告，保留不确定性 | 官方权重已校验；CPU 真实报告生成通过；96 token 验收输出截断，完整报告需更长生成 |
| 树莓派 / 摄像头 / 补光 / 按钮 / 二维码 | 双向 TLS 采集代理、GPIO 控灯和按钮、离线重试队列 | 硬件尚未连接，不能由软件安装代替 |

## 数据和模型条件

1. 默认 DINOv2 / SAM 2.1 是已获公开访问的替代方案。DINOv3 B/16、L/16 和 SAM3 的适配器保留；原模型仍需本人申请官方权重权限，尚未实际推理。SAM 2.1 只接收明确选框，不提供 SAM3 文字概念定位能力。
2. 实际摄像头、麦克风、人脸登记、树莓派、补光、GPIO 按钮和现场扫码需要硬件。软件通信测试不能证明这些实体功能已经完成。
3. 分类模型评估要求至少两个经人工确认的类别，每类至少五张独立图片，拒绝重复图片跨训练/测试划分。当前技术验收没有病害准确率结果。
4. 跨设备访问需要受控网络和匹配主机名的可信证书。Airflow 当前为仅限本机的开发认证模式；上线多人系统需按官方建议更换生产认证管理器。

模型来源：[DINOv2 B](https://huggingface.co/facebook/dinov2-base)、[DINOv2 L](https://huggingface.co/facebook/dinov2-large)、[SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-small)、[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。权重不随源码分发，保留各发布方许可和访问条件。

## 验证记录与目录

本机私有记录：`.runtime/source-validation.json`、`.runtime/core-verification.json`、`.runtime/research-verification.json`、`.runtime/extended/*-verification.json`。日志、凭证、模型、环境、数据库文件和这些本机记录均不进入 Git。

`web/` 为前端；`business/` 为 Java / SQL；`worker/` 为 Celery、事件、MQTT 和 Doris 写入；`model_service/` 为模型与数值后端；`integrations/` 为原工作台适配；`analytics/` 为 Flink、Doris 和 Airflow；`edge/` 为实体采集端代码；`infra/` 为监控和网关配置；`scripts/` 为分组启动及验证入口。
