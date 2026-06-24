# Robot Data Studio

一个 local-first 的机器人数据工作台 MVP。当前版本已经支持：

- 导入本地 LeRobot v3 数据集并索引 episode
- 使用 Rerun WebViewer 回放 episode 的 action、state 和视频数据
- 将单个 episode 导出为 ACT 风格 HDF5
- 在浏览器中完成导入、选择、回放和导出

## 使用教程

完整中文教程见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

## 启动

需要 Python 3.11+、Node.js 和 pnpm。

如果本机没有全局 pnpm：

```bash
npm install -g pnpm
```

或者直接使用 Codex 自带的 pnpm：

```bash
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm
```

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
pnpm install
.venv/bin/python scripts/download_sample.py
```

分别启动 API 和前端：

```bash
.venv/bin/uvicorn apps.api.main:app --reload --port 8000
```

```bash
pnpm dev:web
```

如果没有全局 pnpm：

```bash
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm dev:web
```

访问 [http://127.0.0.1:5173](http://127.0.0.1:5173)，默认数据集路径已经指向
`data/samples/lerobot-pusht`。

## 验证

```bash
.venv/bin/pytest -q
.venv/bin/ruff check apps packages tests scripts
pnpm test:web
pnpm build:web
```

导出的 `.rrd` 和 `.hdf5` 位于 `.rds-artifacts/`。Rerun 的 WASM 文件会在
`dev` 或 `build` 前自动从 npm 依赖复制，不需要提交到 Git。

## 当前边界

第一版只实现 LeRobot v3 输入和单 episode ACT HDF5 输出。清洗规则、质量评分、
坐标系转换以及更多格式适配会通过 reader / exporter / validator 接口继续扩展。
