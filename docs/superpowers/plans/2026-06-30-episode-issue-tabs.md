# Episode Issue Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep all navigable findings for the selected episode available as one-click tabs while a filter detail is open.

**Architecture:** Derive a de-duplicated tab model from the selected episode's existing findings and the finding-to-filter mapping. Pass that model and the active filter stage into the detail view, where an accessible tab row switches the existing filter-detail query without changing episodes.

**Tech Stack:** React 18, TypeScript, TanStack Query, Vitest, Testing Library, CSS

---

### Task 1: Add failing interaction coverage

**Files:**
- Test: `apps/web/src/App.test.tsx`

- [x] **Step 1: Add a test with three navigable findings**

Extend the issue-card test setup with findings for `visual_quality`, `sudden_change`, and `time_sync`. After opening visual quality, assert that the three filter labels are rendered as tabs and that Visual quality is selected:

```tsx
expect(await screen.findByRole("tab", { name: "Visual quality", selected: true })).toBeInTheDocument();
expect(screen.getByRole("tab", { name: "Sudden change", selected: false })).toBeInTheDocument();
expect(screen.getByRole("tab", { name: "State-action alignment", selected: false })).toBeInTheDocument();
```

Click the Sudden change tab and assert that the same episode's detail endpoint is requested:

```tsx
await user.click(screen.getByRole("tab", { name: "Sudden change" }));
expect(fetch).toHaveBeenLastCalledWith(
  "/api/projects/project-1/filters/sudden_change/episodes/0",
  expect.any(Object),
);
```

- [x] **Step 2: Run the focused test and verify failure**

Run:

```bash
pnpm --filter web test -- --run apps/web/src/App.test.tsx -t "switches between issue details"
```

Expected: FAIL because no issue tablist exists.

### Task 2: Implement the issue tab model and UI

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`

- [x] **Step 1: Derive unique navigable issue tabs**

Add a helper that preserves finding order, removes duplicate stages, and labels stages from the filter summary when available:

```tsx
function issueTabsForQuality(
  quality: EpisodeQualityResult | null,
  filterSummary: FilterSummary | null,
) {
  const labels = new Map(filterSummary?.filters.map((filter) => [filter.id, filter.label]) ?? []);
  const seen = new Set<FilterStageId>();
  return (quality?.findings ?? []).flatMap((finding) => {
    const stageId = filterStageForFindingCode(finding.code);
    if (!stageId || seen.has(stageId)) return [];
    seen.add(stageId);
    return [{ stageId, label: labels.get(stageId) ?? humanizeMetric(stageId) }];
  });
}
```

- [x] **Step 2: Pass the tabs to the detail view**

Compute the tabs for `selectedQuality`, then pass `issueTabs`, `activeFilterStage`, and the existing `onOpenFilter` callback to the filter detail area.

- [x] **Step 3: Render an accessible tab row**

When a filter detail is active and at least one navigable issue exists, render:

```tsx
<div className="episode-issue-tabs" role="tablist" aria-label="Episode issues">
  {issueTabs.map((tab) => (
    <button
      key={tab.stageId}
      type="button"
      role="tab"
      aria-selected={tab.stageId === activeFilterStage}
      className={tab.stageId === activeFilterStage ? "active" : ""}
      onClick={() => onOpenFilter(tab.stageId)}
    >
      {tab.label}
    </button>
  ))}
</div>
```

Keep the row visible for a single navigable issue. Do not create tabs for unmapped findings.

- [x] **Step 4: Style active, hover, focus, and overflow states**

Use a horizontally scrollable compact row that does not shrink the detail content. Give the active tab the existing blue accent and preserve a visible keyboard focus ring.

- [x] **Step 5: Run the focused test**

Run:

```bash
pnpm --filter web test -- --run apps/web/src/App.test.tsx -t "switches between issue details"
```

Expected: PASS.

### Task 3: Verify regressions and commit

**Files:**
- Verify: `apps/web/src/App.tsx`
- Verify: `apps/web/src/App.test.tsx`
- Verify: `apps/web/src/styles.css`

- [x] **Step 1: Run all web tests**

Run:

```bash
pnpm --filter web test -- --run
```

Expected: all tests pass.

- [x] **Step 2: Run the production build**

Run:

```bash
pnpm --filter web build
```

Expected: TypeScript and Vite build complete successfully.

- [ ] **Step 3: Commit the implementation**

```bash
git add apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/styles.css docs/superpowers/plans/2026-06-30-episode-issue-tabs.md
git commit -m "feat: add episode issue tabs"
```
