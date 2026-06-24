import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import App from "./App";

vi.mock("@rerun-io/web-viewer", () => ({
  WebViewer: class {
    start = vi.fn();
    stop = vi.fn();
    on = vi.fn(() => vi.fn());
    set_active_timeline = vi.fn();
    set_playing = vi.fn();
    get_playing = vi.fn(() => false);
    get_current_time = vi.fn(() => 0);
    set_current_time = vi.fn();
    get_time_range = vi.fn(() => ({ min: 0, max: 10_000_000_000 }));
  },
}));

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

test("starts with the bundled sample dataset path", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    }),
  );
  renderApp();

  const input = await screen.findByLabelText("Dataset path");

  expect(input).toHaveValue("data/samples/lerobot-pusht");
  expect(input).toHaveAttribute("placeholder", "data/samples/lerobot-pusht");
});

test("imports a dataset and renders its episodes", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => formatsResponse(),
      })
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

test("sends selected import format hint", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => projectResponse({ format: "act_hdf5", version: "act_hdf5" }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => episodesResponse(),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await screen.findByRole("option", { name: "ACT HDF5" });
  await user.clear(screen.getByLabelText("Dataset path"));
  await user.type(screen.getByLabelText("Dataset path"), "data/samples/aloha_static_coffee");
  await user.selectOptions(screen.getByLabelText("Import format"), "act_hdf5");
  await user.click(screen.getByRole("button", { name: "Import dataset" }));

  expect(fetch).toHaveBeenNthCalledWith(
    2,
    "/api/projects",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ path: "data/samples/aloha_static_coffee", format_hint: "act_hdf5" }),
    }),
  );
});

test("exports selected episode to chosen format and shows report path", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
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
        output_path: "/tmp/project-1-episode-000000-umi_zarr.zarr",
        report_path: "/tmp/conversion_report.json",
        format: "umi_zarr",
        episode_count: 1,
      }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.selectOptions(await screen.findByLabelText("Export format"), "umi_zarr");
  await user.click(await screen.findByRole("button", { name: "Export selected" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ episode_indexes: [0], format: "umi_zarr", options: {} }),
    }),
  );
  expect(await screen.findByText(/conversion_report.json/)).toBeInTheDocument();
});

test("runs cleaning and renders episode status folders", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => formatsResponse(),
      })
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
      json: async () => formatsResponse(),
    })
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

test("filters cleaning results by search and finding type", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => formatsResponse(),
      })
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
  await user.type(await screen.findByLabelText("搜索 / 筛选"), "000001");

  expect(screen.queryByText("#000000")).not.toBeInTheDocument();
  expect(screen.getByText("#000001")).toBeInTheDocument();

  await user.clear(screen.getByLabelText("搜索 / 筛选"));
  await user.click(screen.getByLabelText("VLM 失败"));

  expect(screen.queryByText("#000000")).not.toBeInTheDocument();
  expect(screen.getByText("#000001")).toBeInTheDocument();
});

test("sends VLM settings when running cleaning", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
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
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "VLM 设置" }));
  await user.click(screen.getByLabelText("启用 VLM"));
  await user.clear(screen.getByLabelText("VLM API Base URL"));
  await user.type(screen.getByLabelText("VLM API Base URL"), "http://localhost:11434/v1");
  await user.clear(screen.getByLabelText("VLM 模型"));
  await user.type(screen.getByLabelText("VLM 模型"), "gpt-4o-mini");
  const promptInput = screen.getByLabelText("VLM Prompt");
  await user.clear(promptInput);
  await user.click(promptInput);
  await user.paste("Return JSON. Task: {task}");
  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "运行清洗 Pipeline" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/cleaning/runs",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        pass_threshold: 0.8,
        review_threshold: 0.6,
        vlm: {
          enabled: true,
          provider: "OpenAI",
          model: "gpt-4o-mini",
          api_base_url: "http://localhost:11434/v1",
          prompt: "Return JSON. Task: {task}",
          sample_frames: 4,
        },
      }),
    }),
  );
});

test("syncs VLM settings from the panel after a project is open", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => projectResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => episodesResponse(),
    })
    .mockResolvedValue({
      ok: true,
      json: async () => ({
        enabled: true,
        provider: "OpenAI",
        model: "gpt-4o-mini",
        api_base_url: "http://localhost:11434/v1",
        prompt: "Return JSON. Task: {task}",
        sample_frames: 4,
      }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(screen.getByRole("button", { name: "VLM 设置" }));
  await user.click(screen.getByLabelText("启用 VLM"));
  await user.clear(screen.getByLabelText("VLM API Base URL"));
  await user.type(screen.getByLabelText("VLM API Base URL"), "http://localhost:11434/v1");

  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/vlm-settings",
    expect.objectContaining({
      method: "PATCH",
      body: expect.stringContaining('"api_base_url":"http://localhost:11434/v1"'),
    }),
  );
});

test("shows three quality findings and can add the selected episode to cleaning pipeline", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
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
        run_id: "run-2",
        status: "succeeded",
        summary: cleaningSummary(),
      }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "运行清洗 Pipeline" }));

  expect(await screen.findByText("发现 3 个问题")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "加入清洗 Pipeline" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/cleaning/runs",
    expect.objectContaining({ method: "POST" }),
  );
});

test("replays the next episode from the viewer controls", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    })
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
      json: async () => ({ recording_url: "/api/artifacts/episode-000000.rrd" }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recording_url: "/api/artifacts/episode-000001.rrd" }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recording_url: "/api/artifacts/episode-000002.rrd" }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Replay in Rerun" }));

  expect(await screen.findByRole("button", { name: "上一集" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "下一集" }));

  expect(await screen.findByRole("heading", { name: "Episode 000001" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/1/recording",
    expect.objectContaining({ method: "POST" }),
  );

  await user.click(screen.getByRole("button", { name: "下一集" }));

  expect(await screen.findByRole("heading", { name: "Episode 000002" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "下一集" })).toBeDisabled();
  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/2/recording",
    expect.objectContaining({ method: "POST" }),
  );
});

function projectResponse(overrides: Partial<{ format: string; version: string }> = {}) {
  return {
    id: "project-1",
    dataset: {
      path: "/tmp/pusht",
      format: overrides.format ?? "lerobot",
      version: overrides.version ?? "v3.0",
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

function formatsResponse() {
  return [
    { id: "lerobot_v3", label: "LeRobot v3", profile: "lerobot", can_import: true, can_export: true },
    { id: "act_hdf5", label: "ACT HDF5", profile: "hdf5", can_import: true, can_export: true },
    { id: "robomimic_hdf5", label: "robomimic HDF5", profile: "hdf5", can_import: true, can_export: true },
    { id: "umi_zarr", label: "UMI Zarr", profile: "zarr", can_import: true, can_export: true },
  ];
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
      vlm: {
        enabled: false,
        provider: "OpenAI",
        model: "gpt-4o-mini",
        api_base_url: null,
        prompt:
          "You are an automated robot episode evaluator. Return only JSON with success, score, and reason. Judge whether the task was successfully completed from the visual evidence.",
        sample_frames: 4,
      },
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
        findings: [
          { code: "blur", severity: "warn", message: "Wrist camera freeze around 3.2s." },
          { code: "time_sync", severity: "warn", message: "RGB / state offset is around 67ms." },
          { code: "action_jump", severity: "warn", message: "Action has a jump near the start." },
        ],
        review_note: null,
        updated_at: "2026-06-24T00:00:00Z",
      },
      {
        episode_index: 1,
        score: 0.41,
        status: "excluded",
        source: "auto",
        per_attribute_scores: { smoothness: 0.41 },
        findings: [
          { code: "blur", severity: "warn", message: "Wrist camera freeze around 3.2s." },
          { code: "time_sync", severity: "warn", message: "RGB / state offset is around 67ms." },
          { code: "vlm_failed", severity: "warn", message: "VLM semantic check failed." },
        ],
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
