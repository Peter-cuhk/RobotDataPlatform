# Robot Data Studio 产品需求文档（PRD）

版本：V1.0 Draft  
日期：2026-06-23  
产品形态：完全开源、本地优先的机器人数据清洗与转换平台  
首版范围：机械臂与具身操作 Episode 数据，单机数据集不超过 10 GB

## 1. 产品摘要

Robot Data Studio 是一个面向机器人学习工程师、数据工程师和研究人员的本地数据工作台。平台将目前分散在可视化工具、命令行脚本、格式转换仓库和数据质量研究代码中的能力整合成一个统一、可扩展、可复现的工作流：

1. 导入 LeRobot、ACT/robomimic 风格 HDF5 和 UMI/Zarr 数据集。
2. 同步回放视频、状态、动作、轨迹与坐标系。
3. 使用确定性规则和可插拔 VLM 检查数据质量。
4. 通过非破坏式 Pipeline 进行裁剪、筛选、重采样、字段映射和坐标变换。
5. 将处理结果一键导出为目标机器人学习格式，并生成可审计的转换报告。

产品不会把原始数据强制转存到私有数据库。原始文件始终只读，平台只维护项目元数据、本地索引、缓存、处理图和质量报告。首版为浏览器 UI 加本地 Python API，默认不上传任何机器人数据。

## 2. Motivation

机器人数据工具链高度碎片化：

- Rerun 等工具的多模态回放和空间可视化优秀，但不是完整的数据清洗平台。
- LeRobot、robomimic、UMI 等项目拥有各自的数据读取、可视化或转换代码，但格式和抽象不同。
- 部分开源项目提供数据审查、成功率判断或质量评分，但缺少统一的 Schema、坐标语义和导出闭环。
- 实际 HDF5 文件通常只有容器格式相同，内部字段命名、shape、单位、时间戳和坐标约定并不统一。
- 工程团队经常通过一次性脚本完成转换；脚本不可发现、不可预览、难以复现，也难以审计转换是否破坏数据。

当前缺少一个可靠且全面的开源平台，把“看数据、判断质量、清洗数据、转换格式、转换坐标系、验证输出”组合为一个端到端工作流。

Robot Data Studio 的长期价值不只是一个可视化界面，而是建立四个可复用的基础能力：

- Canonical Robot Data Schema：跨格式的统一语义层。
- Adapter SDK：格式 Reader/Writer 插件体系。
- Quality Rules Engine：可组合、可解释的数据质量规则。
- Transform Pipeline IR：可预览、可序列化、可复现的非破坏式处理图。

## 3. 产品原则

### 3.1 本地优先

- 数据默认留在用户电脑。
- 启动本地服务后通过浏览器访问。
- VLM、遥测和联网功能必须显式开启。
- 首版不要求账户、登录或云端项目。

### 3.2 原始数据只读

- 平台不得默认原地修改数据集。
- 所有裁剪、过滤和转换保存为 Pipeline 配置。
- 导出总是写入新目录，并提供覆盖保护。

### 3.3 可解释而非神秘评分

- 每个质量问题必须包含规则、严重度、时间范围、证据、置信度和建议动作。
- 总分必须能追溯到各维度和具体 Finding。
- VLM 结论不能覆盖确定性证据。

### 3.4 统一语义，不做 N×N 转换

- 所有 Reader 映射到 Canonical Episode View。
- 所有 Writer 从 Canonical Episode View 写出。
- 新增一种格式只需实现 Reader/Writer，而不是为每对格式编写转换器。

### 3.5 复用优先，但不外包核心资产

- 复用 Rerun 的回放和 3D 可视化。
- 复用 LeRobot、robomimic、UMI 官方实现中的格式语义与测试样本。
- 平台必须自己掌握 Canonical Schema、Pipeline IR、质量规则协议和插件 API。

## 4. 目标用户与核心场景

### 4.1 目标用户

- 机器人学习工程师：训练前检查和整理示教数据。
- 机器人数据工程师：统一多来源数据、维护格式转换模板。
- 研究人员：快速理解公开数据集并筛选实验子集。
- 算法工程师：核对 observation/action 定义、坐标系与频率。
- 数据采集团队：定位失败 Episode、传感器异常和采集流程问题。

### 4.2 核心场景

#### 场景 A：检查新采集数据

用户选择一个 LeRobot 数据目录，平台扫描结构并生成 Episode 索引。用户同步查看头部相机、腕部相机、关节状态、Action 和末端轨迹，平台自动标记模糊、冻结、时间偏移、关节跳变等问题。

#### 场景 B：陌生 HDF5 Schema Mapping

用户导入非标准 HDF5。平台展示 Group/Dataset 树、dtype、shape、属性和值预览。用户将源字段映射到标准字段，配置单位、时间轴和坐标语义，验证后保存为可复用模板。

#### 场景 C：一键 LeRobot 转 HDF5

用户点击“导出 HDF5”，选择 ACT 或 robomimic 预设。平台显示映射摘要、预计文件数量和校验结果，执行转换后抽样回读，并生成 manifest、checksum 和转换报告。

#### 场景 D：坐标系统一

一个数据集的末端位姿位于 camera frame，训练代码需要 base frame。用户在 Transform 节点中选择输入/输出 Frame、外参、单位和旋转表达，预览转换前后的轨迹叠加，然后导出。

## 5. 产品范围

### 5.1 MVP 必须完成

- 本地项目创建和目录导入。
- LeRobot v2/v3 Reader/Writer。
- ACT/robomimic 风格 HDF5 Reader/Writer。
- UMI/Zarr Reader/Writer；针对 UMI 的具体工程版本通过 profile 适配。
- HDF5 可视化 Schema Mapping。
- Episode 列表、筛选、人工状态标记。
- 基于 Rerun 的多模态同步回放。
- 第一批确定性质量规则。
- 可插拔 VLM 质量规则接口，至少提供一个本地示例适配器和一个 OpenAI-compatible API 适配器。
- 非破坏式 Pipeline 和基础节点。
- 坐标系与位姿表达转换。
- 导出任务、进度、取消、日志和结果报告。
- 插件 SDK、CLI 和基础文档。

### 5.2 MVP 明确不做

- ROS Bag、MCAP、RLDS/TFDS。
- 移动机器人、激光雷达、地图和通用 ROS TF 全覆盖。
- 多用户、权限、团队协作和云端数据仓库。
- TB 级数据、分布式执行和对象存储。
- 训练、模型部署、机器人车队运维。
- 通用 IK、自动标定和所有机器人型号的 FK。
- 自动修复所有质量问题。
- 原地修改原始数据。

### 5.3 后续扩展

- ROS Bag/MCAP、RLDS、DROID 等 Adapter。
- 10–500 GB 工作站数据集。
- URDF/FK 插件、标定辅助和完整 Frame Graph。
- Ray/Celery 执行器和对象存储。
- 团队服务器模式、审查协作和数据版本管理。

## 6. 信息架构与主工作流

### 6.1 一级入口

- Projects：创建项目、导入数据、扫描、最近任务。
- Inspect：Episode 回放、质量报告、人工裁决。
- Pipeline：节点式处理图、参数配置和预览。
- Export：格式选择、任务执行、结果校验和报告。
- Settings：插件、VLM、缓存、性能和隐私设置。

### 6.2 标准工作流

1. 选择数据目录并创建项目。
2. 平台识别格式；无法识别时进入 Schema Mapping。
3. 扫描元数据并建立轻量索引。
4. 执行快速质量扫描。
5. 用户在 Inspect 中检查 Findings 和 Episode。
6. 用户接受建议、排除 Episode 或加入清洗节点。
7. 在 Pipeline 中预览处理结果并运行验证。
8. 选择目标格式并导出。
9. 平台回读抽样、生成报告和可复现配置。

## 7. 功能需求

### 7.1 项目与导入

#### FR-IMPORT-01 目录导入

- 支持选择本地目录或文件。
- 自动探测 LeRobot、HDF5、Zarr/UMI。
- 禁止浏览器直接承担大文件解析；解析由本地 API 完成。
- 最近项目保存路径引用，不复制数据。

#### FR-IMPORT-02 扫描结果

- 显示格式、大小、Episode 数量、相机数量、采样率、字段列表和异常。
- 扫描任务可取消、可恢复显示。
- 10 GB 以下数据集首次扫描目标为 60 秒内出现可浏览的首批结果；完整统计允许后台继续。

#### FR-IMPORT-03 项目文件

项目目录包含：

- `project.yaml`：数据源、格式 profile、平台版本。
- `metadata.sqlite`：Episode 索引、Findings、任务和人工状态。
- `pipelines/*.json`：处理图。
- `reports/`：质量与导出报告。
- `.cache/`：缩略图、抽帧、Rerun recording 和统计缓存。

### 7.2 Schema Mapping

#### FR-MAP-01 HDF5 树浏览

- 展示 Group、Dataset 和 Attribute。
- 展示 path、dtype、shape、chunk、compression 和样本值。
- 对大型数组仅做切片预览。

#### FR-MAP-02 字段映射

标准目标至少包括：

- `observation.images.<camera>`
- `observation.depth.<camera>`
- `observation.state.joint_position`
- `observation.state.joint_velocity`
- `observation.state.eef_pose`
- `observation.state.gripper`
- `action`
- `timestamp`
- `episode_index`
- `task`
- `success`

映射配置支持源 path、轴选择、reshape、dtype、单位、时间字段、Frame 和语义说明。

#### FR-MAP-03 校验与模板

- 映射后立即校验长度、shape、时间单调性和必填字段。
- 可预览一个 Episode。
- 模板保存为 YAML/JSON，可导入、导出和放入 Git。
- 模板包含版本和兼容范围。

### 7.3 可视化与回放

#### FR-VIZ-01 同步回放

- 同步展示多路 RGB/Depth、状态曲线、Action 曲线、3D 轨迹和坐标轴。
- 支持播放、暂停、倍速、逐帧、时间跳转和区间选择。
- 点击 Finding 自动跳转到对应时间。
- 视频与状态使用平台统一时间轴，不以数组 index 假装时间同步。

#### FR-VIZ-02 Episode 浏览

- 支持搜索、排序和按分数、规则、人工状态、任务筛选。
- Episode 状态：未审查、通过、排除、需修复。
- 支持备注和标签。

#### FR-VIZ-03 可视化实现

- Rerun Web Viewer 负责视频、曲线、3D 和时间轴。
- React 页面负责项目导航、Episode 列表、质量面板和操作控件。
- Viewer 与业务 UI 通过 `episode_id`、`timestamp` 和 selection event 同步。
- Rerun 仅作为 Viewer Adapter，不作为平台唯一数据模型。

### 7.4 数据质量

#### FR-QA-01 Finding 模型

每个 Finding 包含：

- `rule_id`、规则版本。
- 数据集与 Episode ID。
- 严重度：info/warning/error/critical。
- 时间范围和受影响字段。
- 指标值、阈值、证据。
- 置信度。
- 建议动作。
- 来源：deterministic/vlm/human。
- 状态：open/accepted/ignored/resolved。

#### FR-QA-02 第一批规则

结构完整性：

- 必填字段缺失。
- dtype/shape 不一致。
- NaN/Inf。
- 空 Episode 或长度异常。

时间与同步：

- 时间戳倒退、重复或非单调。
- 采样间隔异常和丢帧。
- 多模态起止时间不一致。
- 视觉与状态的估计时间偏移。

视觉质量：

- 黑帧/近黑帧。
- 过曝。
- 模糊。
- 冻结或重复帧。
- 分辨率漂移。

运动学与控制：

- Joint position 超过配置范围。
- 速度和加速度尖峰。
- EEF 轨迹不连续。
- Quaternion 未归一化。
- Quaternion 符号跳变造成的假不连续。
- Action 与 state 维度不匹配。
- 长时间静止和 Episode 头尾冗余。

#### FR-QA-03 评分

- 输出结构、时间、视觉、运动、语义五个维度分。
- 默认总分为配置化加权平均。
- Critical Finding 可触发 Quality Gate 直接失败。
- 规则阈值按项目 profile 配置。
- 报告显示原始指标，不只显示分数。

#### FR-QA-04 VLM 插件

- 默认关闭，不影响离线核心功能。
- 输入为抽样帧、时间片段、任务描述和可选状态摘要。
- 输出必须符合结构化 JSON Schema。
- API Key 只保存在本地安全配置中，不写入项目文件。
- 调用前显示将发送的数据范围。
- 支持 OpenAI-compatible、用户自定义 HTTP 和本地模型适配器。
- VLM 适合任务成功判断、失败阶段、环境异常和语义标签；不负责替代时间戳和数值规则。

### 7.5 非破坏式 Pipeline

#### FR-PIPE-01 Pipeline IR

Pipeline 保存为有版本的 JSON/YAML DAG，包含：

- 节点类型和插件版本。
- 输入/输出端口。
- 参数。
- 数据 Schema 前置条件。
- 随机种子。
- 执行环境摘要。
- 创建时间和运行记录。

#### FR-PIPE-02 MVP 节点

- Source。
- Field Map/Rename。
- Select/Exclude Episodes。
- Trim Time Range。
- Trim Static Head/Tail。
- Resample/Align。
- Fill/Drop Invalid Samples。
- Unit Convert。
- Coordinate Transform。
- Quaternion Convert。
- Quality Gate。
- Export。

#### FR-PIPE-03 预览

- 节点支持对单 Episode 或短区间执行 Preview。
- 显示处理前后 shape、频率、时间范围和轨迹差异。
- Preview 结果进入临时缓存，不修改源数据。

#### FR-PIPE-04 执行

- 首版使用本地进程池执行 CPU 密集任务。
- 任务支持 queued/running/succeeded/failed/cancelled。
- 通过 SSE 或 WebSocket 推送进度和日志。
- 节点失败时保留错误上下文和已完成的安全缓存。

### 7.6 坐标系与空间变换

#### FR-TF-01 Frame 元数据

每个 Pose/Point/Vector/Action 字段可以声明：

- `frame_id`
- `parent_frame_id`
- 长度单位。
- 旋转表示。
- Quaternion 顺序。
- handedness。
- Pose 语义，例如 `T_parent_child`。

#### FR-TF-02 转换能力

- Euler、rotation matrix、axis-angle、quaternion 互换。
- Quaternion XYZW/WXYZ 互换与归一化。
- 米、厘米、毫米转换。
- 左手系/右手系轴变换。
- 静态外参和逐帧 Transform。
- Pose、Point、Vector 使用正确的数学规则分别变换。
- Absolute pose 与 delta pose/action 分开处理，禁止混用。

#### FR-TF-03 验证与预览

- 检查旋转矩阵正交性和 determinant。
- 检查 Frame Graph 是否断裂或形成非法环。
- 转换前后轨迹叠加。
- 随机抽样做 round-trip 误差测试。
- 导出报告记录变换链和数值误差。

### 7.7 格式转换与导出

#### FR-EXPORT-01 一键预设

- LeRobot → ACT HDF5。
- LeRobot → robomimic HDF5。
- HDF5 → LeRobot。
- UMI/Zarr → LeRobot。
- LeRobot → UMI/Zarr profile。

预设是可查看和复制的 Pipeline，不是隐藏脚本。

#### FR-EXPORT-02 输出校验

- 输出前检查目标 Schema 必填项。
- 写入临时目录，成功后原子性重命名。
- 默认拒绝覆盖非空目录。
- 导出后重新打开并抽样读取。
- 检查 Episode 数量、长度、shape、时间范围和关键字段统计。
- 生成 manifest、文件 checksum、平台版本、插件版本和 Pipeline。

#### FR-EXPORT-03 CLI

GUI 中保存的 Pipeline 可以通过 CLI 执行，例如：

`rds run pipeline.json --source ./dataset --output ./cleaned`

CLI 和 GUI 使用同一 Python Core，不维护两套逻辑。

## 8. Canonical Episode Schema

Canonical Schema 是逻辑视图，不要求将数据提前复制成新的物理格式。

### 8.1 顶层对象

- Dataset：数据集元数据、Feature 定义、Task、Robot profile。
- Episode：Episode ID、时间范围、样本数、标签和来源。
- Timeline：主时间轴和各模态时间轴。
- Feature：语义、dtype、shape、单位、Frame、采样方式。
- Sample Window：按字段与时间窗口读取的惰性视图。

### 8.2 核心字段约定

- 时间统一以秒表达，保留源时间戳和精度信息。
- 图像统一声明颜色空间和编码，不强制全部解码存储。
- Pose 推荐逻辑表示为 translation + normalized quaternion，同时保留源表示。
- Action 必须声明语义：joint position、joint velocity、EEF absolute、EEF delta 等。
- 所有字段都允许扩展命名空间，核心字段保持稳定。

### 8.3 Reader 接口

- `probe(path) -> confidence, format_profile`
- `scan() -> DatasetMetadata`
- `list_episodes(filter) -> EpisodeSummary`
- `schema() -> CanonicalSchema`
- `read_window(episode, fields, time_range, sampling) -> SampleBatch`
- `read_asset(asset_ref, range) -> bytes/stream`
- `close()`

### 8.4 Writer 接口

- `validate(schema, options) -> ValidationReport`
- `plan(dataset, options) -> ExportPlan`
- `write_episode(view)`
- `finalize() -> ExportManifest`
- `verify() -> VerificationReport`

## 9. 技术架构

### 9.1 总体架构

采用模块化单体，而非微服务：

- Web App：React/TypeScript。
- Local API：FastAPI。
- Core：Canonical Schema、Adapters、Pipeline、Quality、Transforms。
- Worker：本地进程池。
- Metadata：SQLite。
- Artifact/Cache：本地文件系统。
- Viewer：Rerun Web Viewer。

模块通过 Python 接口和版本化 JSON Schema 解耦。未来服务器化时可替换 TaskExecutor、ArtifactStore 和 MetadataStore。

### 9.2 前端选型

- React + TypeScript + Vite：成熟、生态完整，适合复杂本地 Web App。
- TanStack Query：API 缓存、任务轮询和失效管理。
- Zustand：当前 Episode、时间点、选择状态等轻量客户端状态。
- React Router：一级页面导航。
- React Flow：节点式 Pipeline 编辑器，MIT 许可。
- shadcn/ui + Radix primitives：可维护、可复制进仓库的 UI 基础。
- Rerun Web Viewer：多模态时间轴和 3D Viewer。
- ECharts 或 uPlot：仅用于业务仪表盘和 Rerun 不覆盖的统计图。

不选择 Streamlit/Gradio 作为正式前端，因为 Schema Mapping、复杂状态同步、节点编辑器和长期插件 UI 会快速超过其适用边界。

### 9.3 后端与数据选型

- Python 3.11+。
- FastAPI + Pydantic：本地 API、OpenAPI 和结构化配置。
- Uvicorn：本地服务。
- SQLite：项目元数据，不存大型数组和视频。
- SQLModel 或 SQLAlchemy：元数据访问；建议 SQLAlchemy 2.x 以减少框架绑定。
- h5py：HDF5 树、属性和按切片读取。
- zarr：UMI/Zarr 及未来 chunked store。
- PyArrow/Parquet：LeRobot 表格数据和统计。
- PyAV/FFmpeg：视频解码、抽帧和转码。
- NumPy + SciPy Rotation：数值和旋转变换。
- OpenCV：视觉质量规则。
- Rerun Python SDK：构建 Viewer recording。
- multiprocessing/ProcessPoolExecutor：首版后台执行。

### 9.4 API 设计

REST 负责资源与命令：

- `/api/projects`
- `/api/datasets/probe`
- `/api/datasets/{id}/scan`
- `/api/datasets/{id}/episodes`
- `/api/episodes/{id}/findings`
- `/api/schema/mappings`
- `/api/pipelines`
- `/api/tasks`
- `/api/exports`
- `/api/plugins`

SSE 优先用于单向任务进度和日志；仅在需要双向 Viewer 控制时使用 WebSocket。

### 9.5 前后端打包

- 开发模式：Vite dev server + FastAPI。
- 发布模式：构建静态前端，由 FastAPI 提供或打包到 Python wheel。
- CLI：`rds serve` 启动服务并打开浏览器。
- 后续可提供 PyInstaller/Briefcase 或桌面壳，但 MVP 不引入 Electron。

## 10. 插件体系

### 10.1 插件类型

- Reader Plugin。
- Writer Plugin。
- Quality Rule Plugin。
- Pipeline Node Plugin。
- VLM Provider Plugin。
- Robot Profile Plugin。
- Viewer Adapter Plugin。

### 10.2 发现与隔离

- 使用 Python entry points 发现插件。
- 插件声明 API version、capabilities、配置 JSON Schema 和许可证。
- 首版插件运行于同一环境；未来可增加子进程隔离。
- 插件异常不能破坏项目元数据。

### 10.3 兼容策略

- Core API 使用语义化版本。
- Pipeline 文件记录插件标识和版本约束。
- 不兼容时允许只读打开并提示迁移。

## 11. 可复用开源项目

### 11.1 直接依赖

#### Rerun

用途：多模态回放、时间轴、视频、曲线、3D 与 Transform 可视化。官方支持通过 iframe 或 JavaScript package 嵌入 Web 应用。许可证为 MIT OR Apache-2.0。

策略：作为 Viewer Adapter 使用；平台业务状态和 Canonical Schema 不依赖 Rerun entity path。

#### LeRobot

用途：LeRobotDataset 读取/写入、格式定义、测试数据和已有 episode 操作逻辑。其公开实现使用同步视频与 Parquet 状态/action 数据，许可证为 Apache-2.0。

策略：优先调用官方库或复用兼容逻辑，避免自行猜测 v2/v3 细节；通过适配层隔离版本变化。

#### robomimic

用途：HDF5 schema、数据集检查脚本、轨迹结构与测试样本。许可证为 MIT。

策略：实现 robomimic profile；不把训练框架作为核心依赖。

#### UMI

用途：UMI 数据处理约定、Zarr 结构、轨迹与相对动作语义参考。许可证为 MIT。

策略：将 UMI 定义为 versioned profile，不声称所有称作“UMI”的数据都拥有完全一致的 schema。

#### React Flow

用途：Pipeline 节点编辑器。许可证为 MIT。

### 11.2 参考而非核心依赖

- ARES：参考语义审查、组合式标签和 VLM 辅助分析。
- Forge：参考 Canonical Adapter、lint、format conversion 的抽象方向；在依赖前必须再次审查成熟度、维护状态和许可证。
- EmbodiFlow 等商业产品：参考工作流，不复制专有实现。

### 11.3 许可证治理

- 仓库采用 Apache-2.0，兼顾开源使用和专利授权条款。
- 建立 `THIRD_PARTY_NOTICES.md` 和依赖许可证扫描。
- 不直接复制来源不清或无许可证的代码。
- 对 GPL/AGPL 依赖默认拒绝进入核心分发，除非隔离并经过明确评估。

## 12. 非功能需求

### 12.1 性能

- 首版支持不超过 10 GB 的单数据集。
- 首屏不等待完整扫描；优先返回元数据和前几个 Episode。
- HDF5/Zarr 使用切片与 chunk 读取，不整体载入内存。
- 视频按时间窗口解码并缓存缩略图。
- UI 在 1,000–10,000 Episode 下使用虚拟列表。

### 12.2 可靠性

- 原始数据只读。
- 导出使用临时目录与完成标记。
- SQLite 使用事务。
- 任务失败不留下看似成功的输出。
- Pipeline 可重复运行；相同输入和配置应产生可比较结果。

### 12.3 安全与隐私

- 服务默认仅绑定 `127.0.0.1`。
- 不默认启用遥测。
- 路径访问限制在用户显式加入的项目根目录。
- VLM 外发前二次确认并提供脱敏/抽帧配置。
- API Key 不进入日志和项目导出包。

### 12.4 可测试性

- 每种 Adapter 提供 golden fixtures。
- 格式转换必须做 round-trip 或语义等价测试。
- 坐标变换使用已知矩阵和随机 property-based tests。
- 质量规则使用合成异常数据。
- UI 对核心工作流做 Playwright 端到端测试。

## 13. MVP 验收标准

### 13.1 导入与浏览

- 能在 macOS/Linux 上从命令行启动本地服务。
- 能导入三个核心格式的样例数据。
- 能在 10 GB 以下数据集上按 Episode 浏览，不因全量加载耗尽内存。

### 13.2 可视化

- 两路视频、状态、Action 和 EEF 轨迹能够基于时间同步回放。
- Finding 点击后跳转到正确时间范围。

### 13.3 质量

- 至少实现 15 条确定性规则。
- 每条规则产生结构化 Finding。
- 支持批量筛选和人工通过/排除。
- 未配置 VLM 时所有核心流程正常。

### 13.4 转换

- 完成 LeRobot ↔ ACT HDF5。
- 完成 LeRobot ↔ robomimic HDF5。
- 完成至少一个明确版本的 UMI/Zarr ↔ LeRobot profile。
- 导出后自动回读，关键字段统计与 Episode 数一致。

### 13.5 坐标

- 支持静态 Transform、单位、旋转表示与 Quaternion 顺序转换。
- 支持转换前后轨迹预览。
- 提供 round-trip 数值误差报告。

### 13.6 开源可用性

- Apache-2.0 LICENSE。
- 一条命令安装或清晰的开发安装步骤。
- 插件开发示例。
- 至少三份公开小型 fixture。
- CI 覆盖 lint、unit、integration 和 UI smoke test。

## 14. 开发阶段建议

### Phase 0：架构验证（1–2 周）

- Canonical Schema 草案。
- Reader/Writer 接口。
- 用同一个 Episode 从 LeRobot 和 HDF5 读入统一视图。
- 将统一视图送入 Rerun Web Viewer。
- 验证浏览器嵌入、时间同步和视频 codec。

### Phase 1：可用的 Inspect MVP（3–5 周）

- Projects、导入扫描、Episode 索引。
- Inspect 工作台。
- 8–10 条基础规则。
- 人工通过/排除和报告。

### Phase 2：Mapping 与 Conversion（3–4 周）

- HDF5 Tree 与 Schema Mapping。
- LeRobot/HDF5 Reader/Writer。
- 导出校验与报告。
- CLI Pipeline 执行。

### Phase 3：Pipeline 与 Coordinate Transform（3–5 周）

- React Flow 编辑器。
- 节点执行与 Preview。
- 坐标转换节点和轨迹叠加。
- UMI/Zarr profile。

### Phase 4：开源发布（2 周）

- 插件 SDK、文档、fixtures。
- 安装与打包。
- 安全、许可证和性能检查。
- GitHub issue templates、贡献指南和路线图。

## 15. 关键风险与应对

### R1：格式名称相同但 Schema 不同

应对：使用 versioned profile、可视化 Mapping 和严格 validation；避免“支持 HDF5”等同于“支持所有 HDF5”的宣传。

### R2：Rerun 嵌入能力限制产品交互

应对：Viewer Adapter 隔离；业务 UI 不依赖 Rerun 内部状态。必要时用自研统计图补充，但 MVP 不重写完整 Viewer。

### R3：视频 codec 在 Web Viewer 中兼容性不一致

应对：项目扫描时探测 codec；提供本地代理转码或缓存为兼容格式；优先支持 Chromium。

### R4：坐标转换语义错误比代码错误更危险

应对：强制声明 frame、单位、quaternion 顺序和 Pose 语义；提供可视化叠加、round-trip 和导出报告。

### R5：VLM 分数不可重复

应对：VLM 与确定性评分分离；记录 provider、model、prompt version、采样策略和结果；VLM Finding 保留置信度。

### R6：“大一统”导致 MVP 失控

应对：首版只聚焦机械臂 Episode、三种格式和 10 GB 以下数据；ROS、团队服务和分布式执行只保留接口。

## 16. 开源仓库建议

建议 monorepo：

```text
robot-data-studio/
  apps/
    web/
    api/
  packages/
    core/
    adapters/
      lerobot/
      hdf5/
      umi_zarr/
    quality/
    transforms/
    pipeline/
    viewer_rerun/
    plugin_sdk/
  fixtures/
  docs/
  examples/
  pyproject.toml
  pnpm-workspace.yaml
  LICENSE
  THIRD_PARTY_NOTICES.md
```

Python 包建议使用 `robot_data_studio` 命名空间，CLI 为 `rds`。前端与后端通过生成的 OpenAPI TypeScript client 保持类型一致。

## 17. 建议的产品名称

工作名：Robot Data Studio。

候选：

- RoboData Studio
- Robot Data Wrangler
- Embodied Data Studio
- RDS

正式开源前需检查 GitHub、PyPI、npm 和域名重名情况。

## 18. 参考资料

- Rerun 文档（嵌入、Recordings、Video）：https://rerun.io/docs/
- LeRobot：https://github.com/huggingface/lerobot
- robomimic 文档：https://robomimic.github.io/docs/
- UMI：https://github.com/real-stanford/universal_manipulation_interface
- React Flow：https://reactflow.dev/
- FastAPI 文档：https://fastapi.tiangolo.com/
- h5py Dataset：https://docs.h5py.org/en/latest/high/dataset.html
- Zarr-Python：https://zarr.readthedocs.io/
