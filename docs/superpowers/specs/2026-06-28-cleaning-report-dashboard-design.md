# Cleaning Report Dashboard Design

## Goal

After a cleaning pipeline run, show a compact dataset-level report instead of the replay placeholder. The report must make the pipeline outcome, dataset scale, data contract, task context, and quality findings understandable without opening individual episodes.

## Page behavior

- A successful full-dataset pipeline run opens the report.
- The report uses the full workspace width; the episode and selected-episode quality sidebars remain available in inspection mode.
- Selecting a chart bar or episode ID exits the report and opens that episode in inspection mode.
- An “Inspect episodes” action exits the report without changing the current selection.
- A “View report” action in inspection mode returns to the report.
- A selected-episode pipeline run keeps the user in inspection mode because it is not a dataset-level result.

## Report layout

1. Header: report title, dataset name, pipeline-complete state, inspect and export actions.
2. Snapshot: episodes, frames, total duration, sample rate, source format, and mean overall score.
3. Quality distribution: compact sorted score chart.
4. Episode status strip: one horizontal row for Pass, Exclude, and Review. It shows IDs only and truncates long groups with a `+N` count.
5. Data Contract: action representation, shape and dtype, translation/rotation conventions, coordinate frame, observation shape, and timestamp convention.
6. Task & Sensors: task descriptions, robot type, cameras, streams, and episode duration range.
7. Quality findings: dataset-level averages for available quality attributes and concise recommended actions.

## Metadata trust rules

- Derive values only from imported metadata and episode summaries.
- Motor-named action vectors may be labeled “Joint action”.
- Unavailable semantic fields such as coordinate frame and quaternion order display “Not declared”; they are never guessed from vector width.
- Action and observation shape/dtype come from the feature schema.
- Total duration is the sum of episode durations.
- Overall and quality attribute scores are means of available scored values.

## Interaction and accessibility

- Score bars and episode IDs are buttons with episode, score, and status labels.
- Status groups use text labels in addition to color.
- Long IDs and stream lists truncate visually but remain available through accessible labels/title text.
- The layout collapses to one column at narrower desktop widths.

## Verification

- Component tests cover real pipeline phase order, report opening, compact status groups, metadata fallbacks, report/inspection transitions, and selected-episode behavior.
- The full web test suite, TypeScript build, Python test suite, and an in-browser visual check must pass.
