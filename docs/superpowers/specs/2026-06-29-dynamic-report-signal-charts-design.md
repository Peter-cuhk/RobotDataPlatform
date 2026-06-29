# Dynamic Report Signal Charts Design

## Goal

Add two dynamically generated visualizations to the Robot Data Studio cleaning report:

1. A switchable per-episode gripper opening curve.
2. A dataset-wide episode duration distribution.

The charts must work for every imported dataset, use live dataset values rather than pre-generated image files, match the existing dark report interface, and degrade gracefully when gripper metadata or samples are unavailable.

## Scope

This change extends the existing `CleaningReportDashboard`. It does not replace the standalone `scripts/report_charts.py` utility and does not generate or persist PNG files.

The first version supports action dimensions whose metadata names contain `gripper`, case-insensitively. It does not attempt semantic inference from unnamed action dimensions.

## Architecture

### Backend

Add a read-only project report-signals API with two data shapes:

- Dataset duration rows for every episode.
- Gripper time series for one requested episode.

The endpoint receives a project ID and an episode index. It uses the project's existing reader, validates that the episode exists, discovers named gripper dimensions from the action feature metadata, and reads only the requested episode frames for the line chart.

The response contains:

- `episode_index`
- `gripper_series`, with a label and timestamp/value points for every matching dimension
- `episode_durations`, with episode index and duration
- `mean_episode_duration_seconds`
- an optional machine-readable `gripper_unavailable_reason`

Missing projects and episodes return the existing 404 behavior. A dataset without named gripper dimensions returns a successful response with an empty series and an unavailable reason, because the duration chart is still valid.

### Frontend

Add a typed API client function and query for report signals. The query is enabled only while the cleaning report is visible and is keyed by project ID and selected report episode.

Add a `Dataset Signals` report card immediately after `Quality Distribution`. It contains:

- A gripper curve panel with an Episode selector.
- An episode duration bar-chart panel.

The default selected report episode is the lowest available episode index. Changing the selector fetches the chosen episode's gripper series while retaining the dataset duration chart.

Both charts are rendered as native responsive SVG. This keeps them crisp, themeable, testable, and free from generated-file lifecycle concerns.

## Chart Behavior

### Gripper Opening Curve

- X-axis: timestamp in seconds.
- Y-axis: action value.
- One colored line per named gripper action dimension.
- Legend labels come from feature metadata.
- The title identifies the selected episode.
- Loading and request failures are shown inside the chart panel.
- If no named gripper dimensions or points are available, the panel displays a clear unavailable message without hiding the duration chart.

### Episode Duration Distribution

- X-axis: episode index.
- Y-axis: duration in seconds.
- One bar per episode.
- A contrasting horizontal line marks the dataset mean.
- The chart is dataset-wide and does not change when the gripper Episode selector changes.

For dense datasets, labels are sampled while every duration bar remains represented.

## Report Download

The downloadable cleaning-report JSON includes a `signals` section containing:

- Dataset duration rows and mean duration.
- The currently selected episode index.
- The currently displayed gripper series or its unavailable reason.

If signal data has not loaded successfully, downloading remains available and records `signals` as unavailable with an error reason. The report's existing cleaning and filter sections remain unchanged.

## Error Handling

- Invalid project or episode: API 404.
- Missing named gripper dimensions: HTTP 200 with an empty gripper series and a reason.
- Empty episode samples: HTTP 200 with an empty gripper series and a reason.
- Non-finite sample values: omit those individual points.
- Frontend request failure: show a localized inline error and keep the rest of the report usable.
- Report JSON download: never block the download solely because signal charts are unavailable.

## Internationalization and Accessibility

Add English and Chinese copy for the section title, chart titles, axes, mean label, selector, loading state, error state, and unavailable state.

The selector has an accessible label. Each SVG has an accessible title and description. The surrounding panels include concise textual summaries so the information is not available only through color or geometry.

## Testing

### Backend

- Returns ordered duration rows and their mean.
- Returns all named gripper dimensions for the requested episode.
- Rejects an unknown episode with 404.
- Returns a successful unavailable state when gripper metadata is absent.
- Filters non-finite or missing dimension values without failing the response.

### Frontend

- Renders the Dataset Signals section in the cleaning report.
- Shows both charts from the API response.
- Defaults to the lowest episode index.
- Fetches and redraws the gripper curve after Episode selection changes.
- Shows the gripper unavailable state while preserving the duration chart.
- Includes currently displayed signal data in the downloaded JSON report.

### Regression

Run the focused API and React tests, then the existing Python and web test suites. Preserve the existing report layout, report-mode behavior, Episode navigation, and export actions.

## Acceptance Criteria

- Running a full cleaning pipeline opens a report containing both dynamic charts.
- The gripper chart defaults to the first Episode and can switch Episodes.
- The duration distribution covers the complete imported dataset.
- No PNG generation or static dataset path is required.
- Datasets without named gripper channels still show the duration chart and a useful gripper unavailable message.
- The downloaded JSON report contains the signal values currently represented in the UI.
