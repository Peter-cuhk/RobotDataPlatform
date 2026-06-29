# Dynamic Report Signal Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a switchable per-episode gripper curve and dataset-wide episode duration distribution to the Robot Data Studio cleaning report, backed by live dataset values.

**Architecture:** A focused Python reporting module converts a `DatasetAdapter` into typed Pydantic report-signal models. `ProjectService` exposes that module through a read-only FastAPI route, while the React report queries the selected Episode and renders both datasets as responsive SVG. The currently loaded signal response is also included in the downloadable JSON report.

**Tech Stack:** Python 3.11, Pydantic, FastAPI, pytest, React 18, TypeScript, TanStack Query, SVG, Vitest, Testing Library

---

## File Structure

- Create `packages/robot_data_studio/reports/__init__.py`: public exports for report-signal models and builder.
- Create `packages/robot_data_studio/reports/signals.py`: gripper dimension discovery, finite point extraction, duration aggregation, and response models.
- Create `tests/reports/test_signals.py`: focused unit tests using a small in-memory adapter.
- Modify `packages/robot_data_studio/projects/service.py`: add `report_signals(project_id, episode_index)`.
- Modify `apps/api/main.py`: add the report-signals GET route.
- Modify `tests/api/test_projects.py`: verify route status and serialized response.
- Modify `apps/web/src/api.ts`: add report-signal TypeScript types and client function.
- Modify `apps/web/src/i18n.ts`: add English and Chinese report chart copy.
- Modify `apps/web/src/App.tsx`: query selected report Episode, include signals in JSON download, and render accessible SVG charts.
- Modify `apps/web/src/styles.css`: add responsive report-signal card, controls, legend, SVG, and empty-state styling.
- Modify `apps/web/src/App.test.tsx`: verify chart rendering, switching, graceful fallback, and JSON download payload.

### Task 1: Build the report-signal domain module

**Files:**
- Create: `packages/robot_data_studio/reports/__init__.py`
- Create: `packages/robot_data_studio/reports/signals.py`
- Create: `tests/reports/test_signals.py`

- [ ] **Step 1: Write failing tests for ordered durations and named gripper series**

Create an in-memory adapter in `tests/reports/test_signals.py` with two out-of-order `EpisodeSummary` rows, action metadata containing `left_gripper` and `right_gripper`, and finite `EpisodeFrame` samples. Assert:

```python
signals = build_report_signals(adapter, episode_index=0)

assert [row.episode_index for row in signals.episode_durations] == [0, 1]
assert signals.mean_episode_duration_seconds == pytest.approx(1.5)
assert [series.label for series in signals.gripper_series] == [
    "left_gripper",
    "right_gripper",
]
assert signals.gripper_series[0].points[0].model_dump() == {
    "timestamp": 0.0,
    "value": 0.1,
}
assert signals.gripper_unavailable_reason is None
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/reports/test_signals.py -q
```

Expected: collection fails because `robot_data_studio.reports` does not exist.

- [ ] **Step 3: Implement typed models and the minimal builder**

In `packages/robot_data_studio/reports/signals.py`, define:

```python
from __future__ import annotations

import math
from pydantic import BaseModel, Field
from robot_data_studio.formats.models import DatasetAdapter


class SignalPoint(BaseModel):
    timestamp: float
    value: float


class GripperSeries(BaseModel):
    label: str
    dimension_index: int
    points: list[SignalPoint] = Field(default_factory=list)


class EpisodeDurationRow(BaseModel):
    episode_index: int
    duration_seconds: float


class ReportSignals(BaseModel):
    episode_index: int
    gripper_series: list[GripperSeries] = Field(default_factory=list)
    episode_durations: list[EpisodeDurationRow] = Field(default_factory=list)
    mean_episode_duration_seconds: float
    gripper_unavailable_reason: str | None = None
```

Implement `find_named_dimensions(features, "action", "gripper")` by walking list-valued entries inside `features["action"]["names"]`. Implement `build_report_signals(adapter, episode_index)` so it validates the Episode with `adapter.episode`, sorts all Episodes by index, computes the arithmetic mean, reads only the requested Episode frames, and retains only points where both timestamp and action value are finite.

Export `ReportSignals` and `build_report_signals` from `packages/robot_data_studio/reports/__init__.py`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/reports/test_signals.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Add and verify graceful unavailable states**

Add tests proving:

```python
assert build_report_signals(adapter_without_names, 0).gripper_unavailable_reason == "no_named_gripper_dimensions"
assert build_report_signals(adapter_with_empty_frames, 0).gripper_unavailable_reason == "no_gripper_samples"
```

Also place `float("nan")` and `float("inf")` values in test frames and assert they are omitted rather than serialized.

Run:

```bash
uv run pytest tests/reports/test_signals.py -q
```

Expected: all tests pass with no Pydantic serialization warnings.

- [ ] **Step 6: Commit the domain module**

```bash
git add packages/robot_data_studio/reports tests/reports/test_signals.py
git commit -m "feat: build dynamic report signal data"
```

### Task 2: Expose report signals through the project API

**Files:**
- Modify: `packages/robot_data_studio/projects/service.py`
- Modify: `apps/api/main.py`
- Modify: `tests/api/test_projects.py`

- [ ] **Step 1: Write a failing API test**

Add a test that imports `ALOHA_SAMPLE`, calls:

```python
response = client.get(
    f"/api/projects/{project['id']}/report-signals",
    params={"episode_index": 0},
)
```

Assert status 200, `episode_index == 0`, duration rows are ordered, the mean is positive, labels include `left_gripper` and `right_gripper`, and every point contains numeric `timestamp` and `value`. Also assert an unknown Episode returns 404.

- [ ] **Step 2: Run the API test and verify RED**

Run:

```bash
uv run pytest tests/api/test_projects.py -q -k report_signals
```

Expected: FAIL with HTTP 404 because the route is absent.

- [ ] **Step 3: Add the service method and API route**

Import `ReportSignals` and `build_report_signals` into `projects/service.py`, then add:

```python
def report_signals(self, project_id: str, episode_index: int) -> ReportSignals:
    return build_report_signals(self.reader(project_id), episode_index)
```

In `apps/api/main.py`, add:

```python
@app.get(
    "/api/projects/{project_id}/report-signals",
    response_model=ReportSignals,
)
def report_signals(project_id: str, episode_index: int = Query(ge=0)) -> ReportSignals:
    try:
        return service.report_signals(project_id, episode_index)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
```

- [ ] **Step 4: Run focused API and domain tests**

Run:

```bash
uv run pytest tests/reports/test_signals.py tests/api/test_projects.py -q -k "report_signals or signals"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the API slice**

```bash
git add packages/robot_data_studio/projects/service.py apps/api/main.py tests/api/test_projects.py
git commit -m "feat: expose report signal data API"
```

### Task 3: Add the frontend data contract and report query

**Files:**
- Modify: `apps/web/src/api.ts`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/App.test.tsx`

- [ ] **Step 1: Write a failing report-fetch test**

Extend the existing “opens a compact dataset cleaning report” test fetch sequence with a report-signals response:

```ts
{
  episode_index: 0,
  gripper_series: [
    {
      label: "left_gripper",
      dimension_index: 6,
      points: [{ timestamp: 0, value: 0.1 }, { timestamp: 1, value: 0.8 }],
    },
  ],
  episode_durations: [
    { episode_index: 0, duration_seconds: 1 },
    { episode_index: 1, duration_seconds: 2 },
    { episode_index: 2, duration_seconds: 3 },
  ],
  mean_episode_duration_seconds: 2,
  gripper_unavailable_reason: null,
}
```

Assert the request URL is `/api/projects/project-1/report-signals?episode_index=0` and the “Dataset Signals” heading appears.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "compact dataset cleaning report"
```

Expected: FAIL because no report-signals request or heading exists.

- [ ] **Step 3: Add API types and the client function**

Add these exported types to `apps/web/src/api.ts`:

```ts
export type ReportSignalPoint = { timestamp: number; value: number };
export type ReportGripperSeries = {
  label: string;
  dimension_index: number;
  points: ReportSignalPoint[];
};
export type ReportSignals = {
  episode_index: number;
  gripper_series: ReportGripperSeries[];
  episode_durations: Array<{ episode_index: number; duration_seconds: number }>;
  mean_episode_duration_seconds: number;
  gripper_unavailable_reason: string | null;
};

export function getReportSignals(projectId: string, episodeIndex: number) {
  return request<ReportSignals>(
    `/api/projects/${projectId}/report-signals?episode_index=${episodeIndex}`,
  );
}
```

- [ ] **Step 4: Add selected report Episode state and query**

In `App`, derive the first Episode index, initialize/reset `reportEpisodeIndex` when a project is imported, and add:

```ts
const reportSignalsQuery = useQuery({
  queryKey: ["report-signals", project?.id, reportEpisodeIndex],
  queryFn: () => getReportSignals(project!.id, reportEpisodeIndex!),
  enabled: showCleaningReport && Boolean(project) && reportEpisodeIndex !== null,
});
```

Pass `reportSignalsQuery.data`, loading/error states, `reportEpisodeIndex`, and the setter into `CleaningReportDashboard`.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "compact dataset cleaning report"
```

Expected: the selected test passes.

- [ ] **Step 6: Commit the frontend data slice**

```bash
git add apps/web/src/api.ts apps/web/src/App.tsx apps/web/src/App.test.tsx
git commit -m "feat: load report signals in cleaning report"
```

### Task 4: Render accessible responsive SVG charts

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/src/i18n.ts`
- Modify: `apps/web/src/App.test.tsx`

- [ ] **Step 1: Write failing rendering and Episode-switch tests**

Assert the loaded report contains:

```ts
expect(screen.getByRole("heading", { name: "Dataset Signals" })).toBeInTheDocument();
expect(screen.getByRole("img", { name: /Gripper opening curve/ })).toBeInTheDocument();
expect(screen.getByRole("img", { name: /Episode duration distribution/ })).toBeInTheDocument();
expect(screen.getByText("left_gripper")).toBeInTheDocument();
expect(screen.getByText("Mean 2.00s")).toBeInTheDocument();
```

Change the `Report episode` combobox to Episode 1, return a second gripper response, and assert the new request includes `episode_index=1` and the new series label/value summary appears.

- [ ] **Step 2: Run the rendering tests and verify RED**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "report signals"
```

Expected: FAIL because the charts and selector are absent.

- [ ] **Step 3: Add localized copy**

Under both `report` translation objects, add keys for:

```ts
datasetSignals
gripperCurve
durationDistribution
reportEpisode
timeSeconds
actionValue
episodeIndex
durationSeconds
meanDuration
loadingSignals
signalsError
noNamedGripperDimensions
noGripperSamples
```

Use natural English and Chinese labels, and map the two machine-readable unavailable reasons to their localized messages in `App.tsx`.

- [ ] **Step 4: Implement SVG chart helpers**

Add pure helpers for finite extents, linear scales, tick values, SVG line paths, and sampled x-axis labels. Implement:

```tsx
function GripperCurveChart({ copy, signals }: { copy: Translation; signals: ReportSignals })
function EpisodeDurationChart({ copy, signals }: { copy: Translation; signals: ReportSignals })
```

Use a fixed SVG view box with `preserveAspectRatio="none"` on the plotting area, restore non-scaling strokes, include `<title>` and `<desc>`, render axes/ticks as SVG elements, and add visible textual legends/summaries outside the SVG. Do not introduce a charting dependency.

- [ ] **Step 5: Add the Dataset Signals report card**

Insert after `Quality Distribution`:

```tsx
<ReportSignalCharts
  copy={copy}
  episodes={episodes}
  selectedEpisodeIndex={reportEpisodeIndex}
  signals={reportSignals}
  pending={reportSignalsPending}
  error={reportSignalsError}
  onEpisodeChange={onReportEpisodeChange}
/>
```

The component renders the selector, loading/error state, the localized gripper unavailable state, and preserves the duration chart whenever duration rows exist.

- [ ] **Step 6: Add responsive styling**

Create `.report-signals`, `.report-signal-grid`, `.report-signal-panel`, `.report-signal-controls`, `.report-chart`, `.report-chart-legend`, and `.report-chart-empty` rules. Use a two-column desktop grid, collapse to one column in the existing mobile media query, and use the report's existing border/background palette.

- [ ] **Step 7: Run focused rendering tests and type checks**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "report signals"
pnpm --dir apps/web exec tsc --noEmit
```

Expected: all selected tests pass and TypeScript reports no errors.

- [ ] **Step 8: Commit the visual layer**

```bash
git add apps/web/src/App.tsx apps/web/src/styles.css apps/web/src/i18n.ts apps/web/src/App.test.tsx
git commit -m "feat: render dynamic report signal charts"
```

### Task 5: Include displayed signals in the downloaded report

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/App.test.tsx`

- [ ] **Step 1: Write a failing download payload test**

Mock `URL.createObjectURL`, capture the `Blob`, click “Export report”, read the blob text, and assert:

```ts
expect(payload.signals.episode_index).toBe(0);
expect(payload.signals.gripper_series[0].label).toBe("left_gripper");
expect(payload.signals.episode_durations).toHaveLength(3);
```

Add a second case where signal loading failed and assert the report still downloads with:

```ts
signals: { available: false, error: "signals unavailable" }
```

- [ ] **Step 2: Run download tests and verify RED**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "downloaded report"
```

Expected: FAIL because the payload has no `signals` key.

- [ ] **Step 3: Extend `downloadCleaningReport`**

Change the signature to accept `ReportSignals | null` and `unknown` error. Serialize either:

```ts
signals
```

or:

```ts
signals: {
  available: false,
  error: errorMessage(signalsError) ?? "signals unavailable",
}
```

Pass the currently displayed query data and error from the report download callback.

- [ ] **Step 4: Run download tests and verify GREEN**

Run:

```bash
pnpm --dir apps/web test -- --run App.test.tsx -t "downloaded report"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit report download support**

```bash
git add apps/web/src/App.tsx apps/web/src/App.test.tsx
git commit -m "feat: include signal values in report download"
```

### Task 6: Full regression and visual verification

**Files:**
- Modify only if verification reveals a scoped defect in the files above.

- [ ] **Step 1: Run the Python suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the web suite and build**

```bash
pnpm --dir apps/web test -- --run
pnpm --dir apps/web build
```

Expected: all Vitest tests pass and Vite builds successfully.

- [ ] **Step 3: Start the app and visually inspect the real sample**

Run the repository's documented API and web development commands, import:

```text
/Users/peterxie/Desktop/data platform /data/samples/aloha_static_coffee
```

Run the full cleaning pipeline and verify:

- Both charts appear below Quality Distribution.
- The default curve is Episode 000000.
- Switching Episodes updates only the gripper curve.
- The duration mean and bar heights match the dataset.
- Both charts fit at desktop width and stack cleanly at narrow width.
- Missing-gripper messaging does not disturb the duration chart.

- [ ] **Step 4: Run final diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional user and feature changes remain.

- [ ] **Step 5: Commit any verification-only fixes**

If Step 3 required a scoped correction:

```bash
git add apps/api/main.py apps/web/src packages/robot_data_studio/reports packages/robot_data_studio/projects/service.py tests
git commit -m "fix: polish dynamic report signal charts"
```
