# Episode Issue Tabs Design

## Goal

When a user opens one finding from an episode that has multiple quality issues, keep every issue for that episode directly accessible. The user should be able to switch between issue details with one click without leaving the episode.

## Interaction

- The issue detail panel displays a tab row above the selected issue detail.
- The row contains one tab for every finding in the current episode that maps to an available filter detail.
- The selected issue tab is visually highlighted and exposed as the active tab to assistive technology.
- Clicking another tab loads that issue's detail while keeping the current episode selected.
- The tab order matches the finding order shown in the episode quality report.
- Findings without an available detail page remain visible in the episode quality report but do not become tabs.
- If the episode has only one navigable issue, the single active tab remains visible so the user retains context.

## Data and State

The episode quality result remains the source of truth for the tab list. Each finding code is mapped to its filter stage using the existing finding-to-stage mapping. The currently open filter stage determines the active tab.

Switching tabs updates only the selected filter stage and requests the corresponding detail for the same episode. Existing loading and error behavior for filter details is preserved.

## Boundaries

- This change does not alter cleaning results, issue detection, repair actions, or episode navigation.
- Duplicate findings that map to the same filter stage produce one tab.
- Manual review actions continue to apply to the episode, not to an individual issue.

## Verification

- An episode with three navigable findings shows three issue tabs after any finding is opened.
- Opening a finding highlights its matching tab.
- Clicking another tab requests and displays that filter detail for the same episode.
- Duplicate stages do not create duplicate tabs.
- A non-navigable finding does not create a broken tab.
- Existing episode decisions and rerun actions remain available.
