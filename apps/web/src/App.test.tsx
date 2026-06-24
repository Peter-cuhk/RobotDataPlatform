import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import App from "./App";

vi.mock("@rerun-io/web-viewer", () => ({
  WebViewer: class {
    start = vi.fn();
    stop = vi.fn();
  },
}));

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

test("imports a dataset and renders its episodes", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: "project-1",
          dataset: {
            path: "/tmp/pusht",
            format: "lerobot",
            version: "v3.0",
            total_episodes: 206,
            total_frames: 25650,
            fps: 10,
            robot_type: "unknown",
            video_keys: ["observation.image"],
            scalar_keys: ["observation.state", "action"],
            features: {},
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          {
            episode_index: 0,
            length: 161,
            duration_seconds: 16.1,
            tasks: ["Push the T-shaped block onto the T-shaped target."],
            data_file: "data/chunk-000/file-000.parquet",
            video_files: {},
            video_start_seconds: {},
            video_end_seconds: {},
          },
        ],
      }),
  );
  renderApp();

  await user.clear(screen.getByLabelText("Dataset path"));
  await user.type(screen.getByLabelText("Dataset path"), "/tmp/pusht");
  await user.click(screen.getByRole("button", { name: "Import dataset" }));

  expect(await screen.findByText("206 episodes")).toBeInTheDocument();
  expect((await screen.findAllByText("Episode 000000")).length).toBeGreaterThan(0);
});

test("runs cleaning and renders episode status folders", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => projectResponse(),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => episodesResponse(),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          run_id: "run-1",
          status: "succeeded",
          summary: cleaningSummary(),
        }),
      }),
  );
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "运行清洗 Pipeline" }));

  expect(await screen.findByText("待审查")).toBeInTheDocument();
  expect(screen.getAllByText("排除").length).toBeGreaterThan(0);
  expect(screen.getAllByText("通过").length).toBeGreaterThan(0);
  expect(screen.getByText("#000000")).toBeInTheDocument();
  expect(screen.getByText("72 / 100")).toBeInTheDocument();
});

test("marks a review episode as passed", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => projectResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => episodesResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        run_id: "run-1",
        status: "succeeded",
        summary: cleaningSummary(),
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...cleaningSummary().results[0],
        status: "passed",
        source: "manual",
        review_note: "Looks good",
      }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "运行清洗 Pipeline" }));
  await user.click(await screen.findByRole("button", { name: "通过" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/episodes/0/decision",
    expect.objectContaining({
      method: "PATCH",
      body: JSON.stringify({ status: "passed" }),
    }),
  );
});

function projectResponse() {
  return {
    id: "project-1",
    dataset: {
      path: "/tmp/pusht",
      format: "lerobot",
      version: "v3.0",
      total_episodes: 206,
      total_frames: 25650,
      fps: 10,
      robot_type: "unknown",
      video_keys: ["observation.image"],
      scalar_keys: ["observation.state", "action"],
      features: {},
    },
  };
}

function episodesResponse() {
  return [
    {
      episode_index: 0,
      length: 161,
      duration_seconds: 16.1,
      tasks: ["Push the T-shaped block onto the T-shaped target."],
      data_file: "data/chunk-000/file-000.parquet",
      video_files: {},
      video_start_seconds: {},
      video_end_seconds: {},
    },
    {
      episode_index: 1,
      length: 120,
      duration_seconds: 12,
      tasks: ["Push the T-shaped block onto the T-shaped target."],
      data_file: "data/chunk-000/file-000.parquet",
      video_files: {},
      video_start_seconds: {},
      video_end_seconds: {},
    },
    {
      episode_index: 2,
      length: 140,
      duration_seconds: 14,
      tasks: ["Push the T-shaped block onto the T-shaped target."],
      data_file: "data/chunk-000/file-000.parquet",
      video_files: {},
      video_start_seconds: {},
      video_end_seconds: {},
    },
  ];
}

function cleaningSummary() {
  return {
    total: 3,
    passed_count: 1,
    review_count: 1,
    excluded_count: 1,
    unscored_count: 0,
    config: {
      pass_threshold: 0.8,
      review_threshold: 0.6,
      overwrite_manual: false,
    },
    scorer_version: "score_lerobot_episodes-compatible-v1",
    results: [
      {
        episode_index: 0,
        score: 0.72,
        status: "review",
        source: "auto",
        per_attribute_scores: {
          visual_clarity: 0.9,
          smoothness: 0.7,
          runtime: 0.62,
        },
        findings: [{ code: "low_runtime", severity: "warn", message: "Runtime score is low." }],
        review_note: null,
        updated_at: "2026-06-24T00:00:00Z",
      },
      {
        episode_index: 1,
        score: 0.41,
        status: "excluded",
        source: "auto",
        per_attribute_scores: { smoothness: 0.41 },
        findings: [{ code: "low_smoothness", severity: "warn", message: "Smoothness score is low." }],
        review_note: null,
        updated_at: "2026-06-24T00:00:00Z",
      },
      {
        episode_index: 2,
        score: 0.96,
        status: "passed",
        source: "auto",
        per_attribute_scores: { smoothness: 0.96 },
        findings: [],
        review_note: null,
        updated_at: "2026-06-24T00:00:00Z",
      },
    ],
  };
}
