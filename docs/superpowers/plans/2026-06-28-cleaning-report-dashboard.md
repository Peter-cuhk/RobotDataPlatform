# Cleaning Report Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix non-monotonic pipeline progress and replace the post-cleaning placeholder with a compact, trustworthy dataset-level cleaning report.

**Architecture:** Keep the existing inspection workspace intact and add an explicit report mode in `App`. Build report values from `Project.dataset`, episode summaries, cleaning results, and filter summaries through small pure helper functions; render them in a dedicated `CleaningReportDashboard` component. Unknown semantic metadata is displayed as “Not declared”.

**Tech Stack:** React 19, TypeScript, TanStack Query, Vitest, Testing Library, CSS, FastAPI/Pydantic.

---

### Task 1: Make pipeline progress monotonic

**Files:**
- Modify: `apps/web/src/App.test.tsx`
- Modify: `apps/web/src/App.tsx`

- [x] **Step 1: Write the failing test**

Update the streaming test to emit the backend’s real phase order: `filters` followed by `cleaning`. Assert that half-complete filters render 25%, then one-third-complete cleaning renders 67%.

- [x] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test --run src/App.test.tsx -t "renders a progress line"`

Expected: FAIL because the old mapping renders filters at 75%.

- [x] **Step 3: Implement the phase mapping**

Map filters to `0–50%` and cleaning to `50–100%`.

- [x] **Step 4: Run test to verify it passes**

Run: `pnpm --dir apps/web test --run src/App.test.tsx -t "renders a progress line"`

Expected: PASS.

### Task 2: Specify report mode behavior with failing tests

**Files:**
- Modify: `apps/web/src/App.test.tsx`

- [x] **Step 1: Add a report rendering test**

Run a full pipeline and assert the report contains `Cleaning Report`, Episodes, Frames, Duration, Overall score, Data Contract, Task & Sensors, and Quality findings.

- [x] **Step 2: Add compact status strip assertions**

Assert `#000000`, `#000001`, and `#000002` appear in a single `.report-status-strip`, and assert `Lowest score episodes` is absent.

- [x] **Step 3: Add metadata trust assertions**

Provide an action feature with motor names and assert `Joint action`, `[14] · float32`, and `Not declared` are shown.

- [x] **Step 4: Add navigation assertions**

Click a report episode ID and assert inspection mode opens. Click `View report` and assert the dashboard returns.

- [x] **Step 5: Run focused tests to verify they fail**

Run: `pnpm --dir apps/web test --run src/App.test.tsx -t "cleaning report"`

Expected: FAIL because the report dashboard does not exist.

### Task 3: Implement report data helpers and dashboard

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/api.ts`
- Modify: `apps/web/src/i18n.ts`

- [x] **Step 1: Add report mode state**

Open report mode after full-dataset pipeline success, keep selected-episode runs in inspection mode, and reset report mode on import.

- [x] **Step 2: Add pure report helpers**

Implement duration formatting, feature schema reading, action representation inference from explicit motor names, score averaging, status grouping/truncation, task deduplication, and quality metric aggregation.

- [x] **Step 3: Render `CleaningReportDashboard`**

Render the header, snapshot row, compact score chart, one-line status strip, Data Contract, Task & Sensors, and Quality findings. Episode controls call the existing selection path and exit report mode.

- [x] **Step 4: Add report/inspection controls**

Add `Inspect episodes` in report mode and `View report` in inspection mode.

- [x] **Step 5: Run focused tests**

Run: `pnpm --dir apps/web test --run src/App.test.tsx -t "cleaning report"`

Expected: PASS.

### Task 4: Style the confirmed layout

**Files:**
- Modify: `apps/web/src/styles.css`

- [x] **Step 1: Add full-width report layout**

Implement the dense dark report surface, six snapshot metrics, compact chart, three-part horizontal status strip, two-column contract/task cards, and three-column findings section.

- [x] **Step 2: Add responsive behavior**

At narrower desktop widths, wrap the snapshot metrics and collapse report detail cards without horizontal clipping.

- [x] **Step 3: Build the web app**

Run: `pnpm build:web`

Expected: TypeScript and Vite build succeed.

### Task 5: Regression and visual verification

**Files:**
- Modify if required by verified defects only.

- [x] **Step 1: Run the complete web suite**

Run: `pnpm test:web`

Expected: all Vitest tests pass.

- [x] **Step 2: Run the complete Python suite**

Run: `uv run pytest`

Expected: all pytest tests pass.

- [x] **Step 3: Run browser visual verification**

Launch the local API and web app, import the sample dataset, run the pipeline or inject a completed sample response, and verify the report has no clipping, overlapping, or unintended tall episode lists.

- [x] **Step 4: Check the final diff**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; generated `output/` artifacts remain untracked and untouched.
