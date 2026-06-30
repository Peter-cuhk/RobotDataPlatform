# Robot Data Studio

**[English](README.md)** | **简体中文**

Robot Data Studio（RDS）是一个 **local-first** 的机器人数据集检查、清洗、审查、回放与格式转换工作台。它面向训练前的数据质检：把本地数据集导入进来，看清每个 episode 的质量，用 Rerun 回放视频与状态/action，筛出坏数据，并导出为下游训练框架可用的格式。

**RDS 不会修改你的源数据集。** 生成的 Rerun 录制、清洗状态、导出文件和报告都写入项目根目录下的 `.rds-artifacts/`。

## 产品优势

### 全部本地，数据不外泄

RDS 在本机运行，数据集始终留在你的磁盘上，不上传云端。导入、质检、回放、导出都在本地完成，适合对数据隐私和合规有要求的团队。

### 大一统工作台

把训练前最零散的几步收进同一个界面，不必在多个脚本和工具之间来回切换：

1. **主流格式导入与互转** — 支持 LeRobot v3、ACT HDF5、robomimic HDF5、UMI Zarr 等格式的导入，并可导出为多种下游训练框架常用格式。
2. **集成数据筛选与清洗** — 内置质量流水线，覆盖视觉、运动学、元数据等多类检查；自动打分、人工复核、报告导出一气呵成。
3. **内置数据可视化** — 基于 Rerun 回放 observation 视频、state/action 曲线与时间轴，结合报告仪表盘和信号图，快速定位问题 episode。

### 前端完成全流程，代码仍可扩展

日常使用**不需要敲命令行**：导入路径、勾选检查项、运行流水线、查看报告、Rerun 回放、人工决策、导出数据，均可在 Web 界面完成。若需要定制规则、接入新格式或扩展 API，可直接修改 `apps/` 与 `packages/` 下的前后端代码。

## 能做什么

典型工作流：

```text
导入数据集 → 选择检查项与运行范围 → 执行质量流水线 → 查看清洗报告
→ 定位问题 episode → Rerun 回放与人工决策 → 导出干净数据
```

**导入格式（自动识别或手动选择）：**

- LeRobot v3
- ACT HDF5
- robomimic HDF5
- UMI Zarr

**导出格式：**

- ACT HDF5
- robomimic HDF5
- UMI Zarr
- LeRobot v3
- LeRobot v2.1

当前质量检查包括视觉质量、突变检测、state/action 对齐、极值、元数据完整性、运动学一致性、姿态对齐，以及可选的 VLM 任务完成度检查。流水线结束后可查看报告仪表盘、质量分布、问题发现、信号曲线与建议操作。

## 环境要求

- Python 3.11+
- Node.js 20+
- pnpm 11.7.0
- 推荐浏览器：Chrome / Chromium
- 可选：`ffmpeg`（VLM 视频抽帧）
- 运动学一致性检查依赖 Pinocchio（`pip install -e '.[dev]'` 已包含 `pin` 包）

如果没有 pnpm，可用 corepack 启用固定版本：

```bash
corepack enable
corepack prepare pnpm@11.7.0 --activate
```

## 安装

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pnpm install
```

`.[dev]` 会安装后端、测试依赖和 Pinocchio。平台内置 lightweight LeRobot reader，默认不需要额外安装官方 `lerobot` 包；若当前环境已安装兼容版本的官方 writer，导出时会优先使用，否则走本地 fallback。

**可选：下载样例数据**

```bash
python scripts/download_sample.py
```

样例会保存到 `data/samples/lerobot-pusht`，便于第一次试用。

## 启动

需要两个终端，均在项目根目录执行。

**终端 1 — 后端：**

```bash
source .venv/bin/activate
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

**终端 2 — 前端：**

```bash
pnpm dev:web
```

打开 [http://127.0.0.1:5173/](http://127.0.0.1:5173/) 使用界面。可用 [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health) 确认后端已就绪。

## 第一次使用

按下面顺序完成一条完整闭环：

1. **导入数据集** — 在 `Dataset path` 输入本地数据集根目录（不是 `meta/info.json` 这类子文件），点击导入。`Import format` 默认 `Auto detect`；识别失败时可手动选择格式。路径首尾多余的 `'` 或 `"` 会自动清理。
2. **选择运行范围与检查项** — 在侧栏勾选要执行的质量检查，并选择对全部 episode 还是仅选中 episode 运行流水线。
3. **可选高级配置** — 上传 URDF 并配置关节映射以启用运动学一致性检查；在 `VLM 设置` 中启用任务完成度检查（需要 API key 和 `ffmpeg`）。
4. **运行质量流水线** — 点击运行并等待进度完成。每个 episode 会被标记为通过、待审查或排除。
5. **阅读清洗报告** — 打开报告页查看质量分布、问题发现、数据集信号图、夹爪曲线与建议操作；可下载报告 JSON。
6. **定位问题 episode** — 从报告或侧栏进入问题/筛选详情页，查看具体 findings 与参数。
7. **Rerun 回放与人工决策** — 选择 episode 后点击 `Replay in Rerun`，结合曲线与视频判断，必要时标记为通过或排除。人工决策会覆盖自动评分。
8. **导出干净数据** — 在导出面板选择目标格式、输出目录和范围（选中 / 通过 / 已筛选等），生成转换文件与 `conversion_report.json`。

## 生成文件位置

| 内容 | 路径 |
| --- | --- |
| 清洗状态 | `.rds-artifacts/projects/<project_id>/cleaning_state.json` |
| Rerun 录制 | `.rds-artifacts/*.rrd` |
| 导出数据与转换报告 | `.rds-artifacts/` |

## VLM 配置

右上角 `VLM 设置` 可配置视觉任务完成度检查。支持三类 provider：

- **OpenAI-compatible** — 默认 `https://api.openai.com/v1`，也可在 UI 填写自定义 `api_base_url`
- **Gemini** — 需要 `GOOGLE_API_KEY` 或在 UI 中填写 API key
- **Local** — 当前仅返回占位结果，不会进行真实语义判断

常用环境变量：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://localhost:11434/v1   # 可选，自定义 OpenAI-compatible 端点
export GOOGLE_API_KEY=...                         # Gemini
```

视频抽帧需要系统已安装 `ffmpeg`（macOS 可用 `brew install ffmpeg`）。

## 常见问题

### 端口 8000 或 5173 被占用

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -tiTCP:8000 -sTCP:LISTEN | xargs kill
```

`pnpm dev:web` 使用 strict port；若 5173 被占用，对 5173 执行同样操作后重启前端。

### 导入路径存在但识别失败

确认输入的是数据集根目录，例如 `/path/to/dataset`，而不是 `/path/to/dataset/meta/info.json`。

### Rerun 只有曲线、没有视频

重新点击 `Replay in Rerun` 生成新的 `.rrd`。建议使用 Chrome / Chromium；若视频为 AV1 编码，浏览器也需支持 AV1 解码。

### VLM 检查失败

常见原因：未设置 `OPENAI_API_KEY` / `GOOGLE_API_KEY`、endpoint 配置错误、未安装 `ffmpeg`，或 episode 缺少本地视频文件。

## 开发验证

```bash
source .venv/bin/activate
pytest -q
ruff check apps packages tests scripts
pnpm test:web
pnpm build:web
```

## API 摘要

后端默认地址：`http://127.0.0.1:8000`。主要端点：`/api/health`、`/api/formats`、`/api/projects`、`/api/projects/{id}/cleaning`、`/api/projects/{id}/exports`、`/api/artifacts/{filename}`。完整路由见 `apps/api/main.py`。

## 当前限制

- 坐标系转换、大规模后台任务、dataset diff/merge 等能力尚未提供
- Local VLM provider 不进行真实语义评分
- LeRobot 导出在缺少官方 writer 时使用 lightweight fallback
- 多 camera 同屏、ROS bag / RLDS 等格式不在当前支持范围内

## License

仓库当前未附带开源许可证文件。如需对外分发或二次开发，请先自行确认授权方式。
