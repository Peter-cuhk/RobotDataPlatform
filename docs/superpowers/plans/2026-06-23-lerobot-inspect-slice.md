# LeRobot Inspect Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web application that imports a LeRobot v3 dataset, exposes dataset and episode metadata through FastAPI, and displays the result in a React UI.

**Architecture:** A Python package owns dataset probing and reading. FastAPI wraps that package without duplicating logic. A Vite React client calls the typed JSON endpoints. The first slice reads metadata and Parquet indexes only; video playback and cleaning rules follow after this foundation is verified.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, PyArrow, pytest, React, TypeScript, Vite, TanStack Query, Vitest.

---

### Task 1: Repository and development tooling

**Files:**
- Create: `pyproject.toml`
- Create: `package.json`
- Create: `pnpm-workspace.yaml`
- Create: `.gitignore`
- Create: `apps/web/package.json`
- Create: `apps/web/vite.config.ts`
- Create: `apps/web/tsconfig.json`

- [ ] Create the Python and pnpm workspace manifests.
- [ ] Install Python and JavaScript dependencies.
- [ ] Verify `pytest --collect-only` and `pnpm --dir apps/web test --run` execute.

### Task 2: LeRobot v3 dataset adapter

**Files:**
- Create: `packages/robot_data_studio/lerobot/models.py`
- Create: `packages/robot_data_studio/lerobot/reader.py`
- Create: `tests/lerobot/test_reader.py`
- Create: `tests/fixtures/lerobot_v3/meta/info.json`
- Create: `tests/fixtures/lerobot_v3/meta/episodes/chunk-000/file-000.parquet`

- [ ] Write a failing test that probes a LeRobot v3 directory.
- [ ] Run the test and confirm it fails because the adapter is missing.
- [ ] Implement format probing and metadata parsing.
- [ ] Write a failing test that lists episode summaries from Parquet.
- [ ] Implement lazy episode-index reading.
- [ ] Run the adapter tests and confirm they pass.

### Task 3: FastAPI project and dataset endpoints

**Files:**
- Create: `apps/api/main.py`
- Create: `packages/robot_data_studio/projects/service.py`
- Create: `tests/api/test_projects.py`

- [ ] Write failing API tests for health, import, dataset detail, and episode list.
- [ ] Implement an in-memory project registry backed by the LeRobot reader.
- [ ] Implement the four API endpoints.
- [ ] Run API tests and confirm they pass.

### Task 4: React project and inspect screen

**Files:**
- Create: `apps/web/src/main.tsx`
- Create: `apps/web/src/App.tsx`
- Create: `apps/web/src/api.ts`
- Create: `apps/web/src/App.test.tsx`
- Create: `apps/web/src/styles.css`

- [ ] Write a failing component test for importing a path and rendering dataset metadata.
- [ ] Implement the API client and minimal project screen.
- [ ] Implement dataset summary and episode list components.
- [ ] Run frontend tests and confirm they pass.

### Task 5: Download a real LeRobot sample and verify end to end

**Files:**
- Create: `scripts/download_sample.py`
- Create: `data/samples/.gitkeep`
- Create: `README.md`

- [ ] Implement a selective Hugging Face download script for metadata, one Parquet data file, and one video file.
- [ ] Download a small public LeRobot v3 sample.
- [ ] Start the API and web app.
- [ ] Import the downloaded path through the UI.
- [ ] Verify metadata and episodes appear.
- [ ] Run all Python and frontend tests plus production builds.
