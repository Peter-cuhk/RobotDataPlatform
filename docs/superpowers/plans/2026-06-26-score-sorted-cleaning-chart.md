# Score-Sorted Cleaning Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show cleaning-result episodes as a score-ascending bar chart, colored by cleaning decision.

**Architecture:** `CleaningSummaryView` sorts each result by numeric score before rendering its interactive bar. Existing status classes map passed, review, and excluded results to green, yellow, and red fills respectively. The existing chart accessibility labels and click-to-select behavior remain unchanged.

**Tech Stack:** React 19, TypeScript, Vitest, Testing Library, CSS.

---

### Task 1: Render ordered, status-colored score bars

**Files:**
- Modify: `apps/web/src/App.test.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`

- [x] **Step 1: Write the failing test**

Add a cleaning summary fixture assertion that obtains the chart buttons by accessible label and expects their episode order to be excluded (41), review (72), then passed (91). Assert the review bar has its review class.

- [x] **Step 2: Run test to verify it fails**

Run: `cd apps/web && node node_modules/vitest/vitest.mjs --run src/App.test.tsx`
Expected: FAIL because the bars are ordered by episode index, not score.

- [x] **Step 3: Write minimal implementation**

Sort `summary.results` in `CleaningSummaryView` with null scores last and numeric scores ascending. Keep the existing score-to-height calculation and selection callback. Add a review fill rule using the existing yellow status palette.

- [x] **Step 4: Run tests and production build**

Run: `npm run test:web -- --run apps/web/src/App.test.tsx && npm run build:web`
Expected: PASS with a successful TypeScript/Vite build.

- [x] **Step 5: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/styles.css docs/superpowers/plans/2026-06-26-score-sorted-cleaning-chart.md
git commit -m "feat: sort cleaning score chart by score"
```
