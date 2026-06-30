# Robot Data Studio

Robot Data Studio 是一个 local-first 的机器人数据集检查、清洗和格式转换工作台。

它的目标不是训练模型，而是解决训练前最麻烦的一步：把本地机器人数据集导入进来，看清楚每个 episode 的质量，回放视频/状态/action，筛出坏数据，并导出成下游训练框架能用的格式。

当前版本已经跑通了一个完整 MVP：

- 导入本地机器人数据集，优先支持 LeRobot v3
- 查看 dataset metadata、episode 列表、任务描述、帧数和时长
- 使用 Rerun WebViewer 回放 action、observation state 和 observation video
- 运行本地清洗评分，为 episode 标记通过、待审查或排除
- 人工修改 episode 决策，保存在本地清洗状态文件中
- 配置 VLM 检查参数，用于后续视觉任务完成度评分
- 导出 ACT HDF5、robomimic HDF5、UMI Zarr 等格式
- 生成 conversion report，记录字段映射和导出信息

所有数据默认留在本机。平台不会复制或改写源数据集，生成的 `.rrd`、`.hdf5`、清洗状态和报告会写入 `.rds-artifacts/`。

## 适合谁用

这个项目适合：

- 机器人数据采集团队做数据质检
- 做 imitation learning / policy learning 前筛数据
- 把 LeRobot、HDF5、Zarr 等数据格式互转
- 快速检查 episode 的视频、state、action 是否对齐
- 给未来的数据清洗规则、坐标系转换、VLM 评分搭统一入口

## 功能状态

| 模块 | 当前状态 |
| --- | --- |
| LeRobot v3 导入 | 可用，已用 `lerobot/pusht` 和本地 ALOHA 样例验证 |
| Rerun 回放 | 可用，支持曲线和视频帧引用 |
| 本地清洗评分 | 可用，基于 action/state 平滑度、轨迹效率、时长、视频存在性等规则 |
| 人工审核 | 可用，可把 episode 标记为通过、待审查、排除 |
| VLM 设置 | UI/API 已接入；OpenAI-compatible/Gemini/Local provider 有后端接口 |
| ACT HDF5 导出 | 可用 |
| robomimic HDF5 导出 | 可用 |
| UMI Zarr 导出 | 可用 |
| LeRobot v2.1/v3 导出 | 有 lightweight fallback；官方 writer 存在时优先使用 |
| 坐标系转换 | 规划中 |
| 大规模后台任务 | 规划中 |

## 项目结构

```text
apps/
  api/                 FastAPI 后端
  web/                 React + Vite 前端
packages/
  robot_data_studio/
    formats/           数据格式 adapter 和 exporter
    lerobot/           LeRobot reader
    projects/          project/session 管理
    quality/           清洗评分、VLM 评分、状态存储
    viewer/            Rerun recording 生成
scripts/
  download_sample.py   下载 LeRobot pusht 样例数据
tests/                 后端和 reader 测试
```

## 环境要求

- Python 3.11+
- Node.js 20+
- pnpm 10+ / 11+
- 推荐浏览器：Chrome / Chromium
- Pinocchio 4.x，用于运动学一致性筛选；pip 包名是 `pin`，Python import 名是 `pinocchio`
- 如果使用 OpenAI-compatible VLM 评分，需要安装 `ffmpeg`

如果没有 pnpm，可以安装：

```bash
npm install -g pnpm
```

也可以用 corepack：

```bash
corepack enable
corepack prepare pnpm@11.7.0 --activate
```

## 快速开始

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
pnpm install
.venv/bin/python scripts/download_sample.py
```

`.[dev]` 会安装后端、测试依赖和 Pinocchio。若你只想手动补装运动学一致性依赖，可以执行：

```bash
.venv/bin/python -m pip install 'pin>=4,<5'
```

平台内置了 lightweight LeRobot reader，默认环境不需要额外安装官方 `lerobot` Python 包；官方 writer 仅在当前环境已安装且版本兼容时自动使用，否则会走本地 fallback。

下载完成后，样例数据会在：

```text
data/samples/lerobot-pusht
```

启动后端：

```bash
.venv/bin/uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

另开一个终端启动前端：

```bash
pnpm dev:web
```

打开：

```text
http://127.0.0.1:5173/
```

默认输入框会填入：

```text
data/samples/lerobot-pusht
```

点击 `Import dataset` 即可导入样例数据。

## 基本使用流程

### 1. 导入数据集

在 `Dataset path` 输入本地数据集路径，例如：

```text
data/samples/lerobot-pusht
```

或绝对路径：

```text
/Users/you/datasets/my_lerobot_dataset
```

路径首尾如果带了 `'` 或 `"`，后端会自动清理。

`Import format` 默认选择 `Auto detect`。如果自动识别失败，可以手动选择：

- LeRobot v3
- ACT HDF5
- robomimic HDF5
- UMI Zarr

### 2. 查看 episode

导入成功后，页面会展示：

- 数据格式和版本
- episode 数量
- frame 总数
- fps
- 清洗状态

左侧是 episode 列表，可以搜索、筛选、选择待审查 episode。

### 3. 运行清洗评分

点击清洗按钮后，平台会对每个 episode 生成质量分数，并分成：

- 通过
- 待审查
- 排除

当前评分规则主要看：

- action/state 是否过于跳变
- 轨迹是否异常
- episode 时长是否异常
- 视频文件是否存在
- 可选 VLM 检查是否通过

清洗状态会保存到：

```text
.rds-artifacts/projects/<project_id>/cleaning_state.json
```

### 4. 用 Rerun 回放

选择一个 episode，点击 `Replay in Rerun`。

后端会生成 `.rrd` 文件，前端用 Rerun WebViewer 打开。正常可以看到：

- observation video
- observation/state 曲线
- action 曲线
- timeline

生成文件保存在：

```text
.rds-artifacts/
```

### 5. 人工审核 episode

在待审查 episode 上，可以结合 Rerun 回放人工判断：

- 标记为通过
- 标记为排除
- 保持待审查

人工决策会覆盖自动评分，并在后续清洗时默认保留。

### 6. 导出数据集

选择 episode 和目标格式，点击导出。

当前支持：

- `act_hdf5`
- `robomimic_hdf5`
- `umi_zarr`
- `lerobot_v3`
- `lerobot_v2_1`

导出结果和转换报告会写到：

```text
.rds-artifacts/
```

## VLM 评分

右上角 `VLM 设置` 可以配置视觉任务完成度检查。

当前支持三类 provider：

- OpenAI-compatible：默认走 `https://api.openai.com/v1`，也可以填自定义 `api_base_url`
- Gemini：需要 `GOOGLE_API_KEY` 或在 UI 中填 API key
- Local：当前是占位 adapter，适合后续接本地视觉模型

OpenAI-compatible provider 需要：

```bash
export OPENAI_API_KEY=...
```

如果需要自定义 endpoint：

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
```

视频抽帧依赖 `ffmpeg`：

```bash
brew install ffmpeg
```

## API 概览

后端默认运行在：

```text
http://127.0.0.1:8000
```

常用接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/formats` | 列出支持的格式 |
| POST | `/api/projects` | 导入数据集 |
| GET | `/api/projects/{project_id}` | 查看 project |
| GET | `/api/projects/{project_id}/episodes` | 列 episode |
| GET | `/api/projects/{project_id}/episodes/{episode_index}/frames` | 查看 episode frame 数据 |
| POST | `/api/projects/{project_id}/episodes/{episode_index}/recording` | 生成 Rerun `.rrd` |
| POST | `/api/projects/{project_id}/cleaning/runs` | 运行清洗评分 |
| GET | `/api/projects/{project_id}/cleaning` | 获取清洗结果 |
| PATCH | `/api/projects/{project_id}/episodes/{episode_index}/decision` | 修改人工审核结果 |
| GET/PATCH | `/api/projects/{project_id}/vlm-settings` | 查看或修改 VLM 设置 |
| POST | `/api/projects/{project_id}/exports` | 导出数据 |
| GET | `/api/artifacts/{filename}` | 下载生成的 artifact |

导入数据集示例：

```bash
curl -X POST http://127.0.0.1:8000/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"path":"data/samples/lerobot-pusht"}'
```

生成 Rerun recording：

```bash
curl -X POST http://127.0.0.1:8000/api/projects/<project_id>/episodes/0/recording
```

导出 ACT HDF5：

```bash
curl -X POST http://127.0.0.1:8000/api/projects/<project_id>/exports \
  -H 'Content-Type: application/json' \
  -d '{"episode_indexes":[0],"format":"act_hdf5"}'
```

## 验证

```bash
.venv/bin/pytest -q
.venv/bin/ruff check apps packages tests scripts
pnpm test:web
pnpm build:web
```

当前本地验证包括：

- API / service 测试
- LeRobot reader 测试
- Rerun viewer URL 测试
- React UI 测试
- TypeScript build
- Vite production build

## 常见问题

### 端口 8000 被占用

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

停止占用进程：

```bash
lsof -tiTCP:8000 -sTCP:LISTEN | xargs kill
```

然后重新启动后端。

### 端口 5173 被占用

当前 `pnpm dev:web` 使用 strict port。如果 5173 被占用，Vite 会直接报错。

查看并停止：

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
lsof -tiTCP:5173 -sTCP:LISTEN | xargs kill
```

### `pnpm: command not found`

安装 pnpm：

```bash
npm install -g pnpm
```

或者使用 corepack：

```bash
corepack enable
```

### 导入路径明明存在但识别失败

检查输入的是数据集根目录，而不是 `meta/info.json` 文件本身。

正确：

```text
/path/to/dataset
```

错误：

```text
/path/to/dataset/meta/info.json
```

如果路径里有空格，直接粘贴即可。首尾单引号或双引号会自动清理。

### Rerun 只看到曲线，看不到视频

请重新点击 `Replay in Rerun` 生成新的 `.rrd`。旧版本生成的 `.rrd` 可能没有视频帧引用。

另外，当前 Rerun WebViewer 对视频 codec 有要求；建议使用 Chrome / Chromium。如果是 AV1 MP4，浏览器本身也需要支持 AV1 解码。

### VLM 评分失败

常见原因：

- 没有设置 `OPENAI_API_KEY` 或 `GOOGLE_API_KEY`
- OpenAI-compatible endpoint 配错
- 没有安装 `ffmpeg`
- episode 没有本地视频文件

### 本地沙箱无法跑 localhost

某些受限环境会禁止本地端口监听，表现为：

```text
listen EPERM
```

这种情况下无法做真实浏览器冒烟，只能跑单测、lint、build 和 service 级验证。完整 UI 流程建议在本机终端运行。

## 当前边界

这个版本还是 MVP，不是完整生产平台。

已完成的核心闭环是：

```text
导入数据集 → 清洗评分 → 人工审查 → Rerun 回放 → 导出格式
```

后续重点：

- 坐标系转换
- 更完整的数据质量规则
- 大数据集后台任务和进度条
- dataset diff / merge
- 更正式的 VLM 评分 adapter
- 多 camera 同屏布局
- ROS bag / RLDS / 更多 HDF5 schema
- 插件式 reader/exporter/validator API

## License

TBD
