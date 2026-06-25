import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

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

function installMemoryStorage() {
  const values = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      clear: () => values.clear(),
      getItem: (key: string) => values.get(key) ?? null,
      removeItem: (key: string) => values.delete(key),
      setItem: (key: string, value: string) => values.set(key, value),
    },
  });
}

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  installMemoryStorage();
  window.localStorage.clear();
});

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

test("defaults to English and switches fixed UI copy to Chinese while preserving technical terms", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    }),
  );
  renderApp();

  expect(await screen.findByRole("button", { name: "Import dataset" })).toBeInTheDocument();
  expect(screen.getByText("Open a robot dataset")).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "中文" }));

  expect(screen.getByRole("button", { name: "导入 dataset" })).toBeInTheDocument();
  expect(screen.getByText("打开 robot dataset")).toBeInTheDocument();
  expect(screen.getByText(/LeRobot, HDF5 and UMI\/Zarr/)).toBeInTheDocument();
  expect(window.localStorage.getItem("robot-data-studio-language")).toBe("zh");
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
            subtasks: [
              {
                start_frame: 0,
                end_frame: 50,
                start_seconds: 0,
                end_seconds: 5,
                prompt: "Put the pen into the pen holder.",
                skill: "Insert",
                track: "default",
                is_mistake: false,
              },
            ],
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
  expect(screen.getByText("Put the pen into the pen holder.")).toBeInTheDocument();
});

test("filters episodes by subtask prompt", async () => {
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
      }),
  );
  renderApp();

  await user.click(await screen.findByRole("button", { name: "Import dataset" }));
  await user.type(await screen.findByLabelText("Search / filter"), "close the laptop");

  expect(screen.getByRole("button", { name: /Episode 000001/ })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Episode 000000/ })).not.toBeInTheDocument();
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
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.selectOptions(await screen.findByLabelText("Export format"), "umi_zarr");
  await user.clear(screen.getByLabelText("Output folder"));
  await user.type(screen.getByLabelText("Output folder"), "/tmp/user-exports");
  await user.click(await screen.findByRole("button", { name: "Export 1 episode" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        episode_indexes: [0],
        format: "umi_zarr",
        options: { output_dir: "/tmp/user-exports" },
      }),
    }),
  );
  expect(await screen.findByText(/conversion_report.json/)).toBeInTheDocument();
});

test("resizes the quality report panel by dragging the workspace separator", async () => {
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
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  const separator = await screen.findByRole("separator", { name: "Resize quality report panel" });
  const workspace = separator.closest(".workspace");
  expect(workspace).toHaveStyle({ "--quality-panel-width": "300px" });

  fireEvent(separator, new MouseEvent("pointerdown", { bubbles: true, clientX: 1010 }));
  fireEvent(document, new MouseEvent("pointermove", { bubbles: true, clientX: 940 }));
  fireEvent(document, new MouseEvent("pointerup", { bubbles: true }));

  expect(workspace).toHaveStyle({ "--quality-panel-width": "370px" });
});

test("exports all indexed episodes from the export panel scope", async () => {
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
      json: async () => exportResult(3),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.selectOptions(await screen.findByLabelText("Export scope"), "all");
  await user.click(await screen.findByRole("button", { name: "Export 3 episodes" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ episode_indexes: [0, 1, 2], format: "act_hdf5", options: {} }),
    }),
  );
});

test("exports manually checked episodes without changing the inspected episode", async () => {
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
      json: async () => exportResult(2),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  expect(await screen.findByRole("heading", { name: "Episode 000000" })).toBeInTheDocument();
  await user.click(screen.getByLabelText("Add Episode 000000 to export"));
  await user.click(screen.getByLabelText("Add Episode 000002 to export"));
  await user.selectOptions(screen.getByLabelText("Export scope"), "checked");
  await user.click(screen.getByRole("button", { name: "Export 2 episodes" }));

  expect(screen.getByRole("heading", { name: "Episode 000000" })).toBeInTheDocument();
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ episode_indexes: [0, 2], format: "act_hdf5", options: {} }),
    }),
  );
});

test("exports the current filtered episode list", async () => {
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
      json: async () => filterRunResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click(await screen.findByLabelText("VLM failed"));
  await user.selectOptions(screen.getByLabelText("Export scope"), "filtered");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ episode_indexes: [1], format: "act_hdf5", options: {} }),
    }),
  );
});

test("exports passed review and excluded cleaning status scopes", async () => {
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
      json: async () => filterRunResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  await user.selectOptions(screen.getByLabelText("Export scope"), "status_passed");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({ body: JSON.stringify({ episode_indexes: [2], format: "act_hdf5", options: {} }) }),
  );

  await user.selectOptions(screen.getByLabelText("Export scope"), "status_review");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({ body: JSON.stringify({ episode_indexes: [0], format: "act_hdf5", options: {} }) }),
  );

  await user.selectOptions(screen.getByLabelText("Export scope"), "status_excluded");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({ body: JSON.stringify({ episode_indexes: [1], format: "act_hdf5", options: {} }) }),
  );
});

test("disables status export scopes before cleaning has run", async () => {
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
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));

  expect(await screen.findByRole("option", { name: "Passed only" })).toBeDisabled();
  expect(screen.getByRole("option", { name: "Review only" })).toBeDisabled();
  expect(screen.getByRole("option", { name: "Excluded only" })).toBeDisabled();
  expect(fetch).toHaveBeenCalledTimes(3);
});

test("data filter checkboxes use filter summary when exporting current filtered episodes", async () => {
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
      json: async () => filterRunResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  expect(await screen.findByText("Review")).toBeInTheDocument();
  await user.click(await screen.findByLabelText("Sudden change"));

  expect(screen.getByText("#000001")).toBeInTheDocument();
  expect(screen.queryByText("#000000")).not.toBeInTheDocument();

  await user.selectOptions(screen.getByLabelText("Export scope"), "filtered");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ episode_indexes: [1], format: "act_hdf5", options: {} }),
    }),
  );
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
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => filterRunResponse(),
      }),
  );
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(await screen.findByText("Review")).toBeInTheDocument();
  expect(screen.getAllByText("Exclude").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Pass").length).toBeGreaterThan(0);
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
      json: async () => filterRunResponse(),
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
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click(await screen.findByRole("button", { name: "Pass" }));

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
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => filterRunResponse(),
      }),
  );
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.type(await screen.findByLabelText("Search / filter"), "000001");

  expect(screen.queryByText("#000000")).not.toBeInTheDocument();
  expect(screen.getByText("#000001")).toBeInTheDocument();

  await user.clear(screen.getByLabelText("Search / filter"));
  await user.click(screen.getByLabelText("VLM failed"));

  expect(screen.queryByText("#000000")).not.toBeInTheDocument();
  expect(screen.getByText("#000001")).toBeInTheDocument();
});

test("renders the five data filters without Qwen branding", async () => {
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
      }),
  );
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));

  expect(await screen.findByText("Sudden change")).toBeInTheDocument();
  expect(screen.getAllByText("Time sync").length).toBeGreaterThan(0);
  expect(screen.getByText("Extreme value")).toBeInTheDocument();
  expect(screen.getByText("Kinematic consistency")).toBeInTheDocument();
  expect(screen.getByText("Orientation alignment")).toBeInTheDocument();
  expect(screen.queryByText(/qwen/i)).not.toBeInTheDocument();
});

test("opens the extreme value detail view from the filter label", async () => {
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
      json: async () => filterDetailResponse("extreme_value"),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Extreme value" }));

  expect(await screen.findByRole("heading", { name: "极值检测" })).toBeInTheDocument();
  expect(screen.getAllByText("q01").length).toBeGreaterThan(0);
  expect(screen.getAllByText("q99").length).toBeGreaterThan(0);
  expect(screen.getByText("frame 12")).toBeInTheDocument();
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/filters/extreme_value/episodes/0",
    expect.any(Object),
  );
});

test("opens the kinematic consistency view with URDF configuration controls", async () => {
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
        json: async () => filterDetailResponse("kinematic_consistency"),
      }),
  );
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Kinematic consistency" }));

  expect(await screen.findByRole("heading", { name: "运动学一致性" })).toBeInTheDocument();
  expect(screen.getByLabelText("Import URDF")).toBeInTheDocument();
  expect(screen.getByLabelText("End-effector link")).toBeInTheDocument();
  expect(screen.getByLabelText("Joint names")).toBeInTheDocument();
  expect(screen.getByLabelText("EEF position indices")).toBeInTheDocument();
  expect(screen.getByText(/Pinocchio/)).toBeInTheDocument();
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
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => filterRunResponse(),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "VLM settings" }));
  await user.click(screen.getByLabelText("Enable VLM"));
  await user.clear(screen.getByLabelText("VLM API Base URL"));
  await user.type(screen.getByLabelText("VLM API Base URL"), "http://localhost:11434/v1");
  await user.clear(screen.getByLabelText("VLM model"));
  await user.type(screen.getByLabelText("VLM model"), "gpt-4o-mini");
  const promptInput = screen.getByLabelText("VLM Prompt");
  await user.clear(promptInput);
  await user.click(promptInput);
  await user.paste("Return JSON. Task: {task}");
  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(fetch).toHaveBeenCalledWith(
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
  await user.click(screen.getByRole("button", { name: "VLM settings" }));
  await user.click(screen.getByLabelText("Enable VLM"));
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

test("loads saved VLM settings into the panel without exposing the API key", async () => {
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
        enabled: true,
        provider: "OpenAI",
        model: "qwen-vl-max",
        api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_configured: true,
        prompt: "Strictly review task completion: {task}",
        sample_frames: 8,
      }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(screen.getByRole("button", { name: "VLM settings" }));

  expect(await screen.findByDisplayValue("qwen-vl-max")).toBeInTheDocument();
  expect(screen.getByDisplayValue("https://dashscope.aliyuncs.com/compatible-mode/v1")).toBeInTheDocument();
  expect(screen.getByDisplayValue("Strictly review task completion: {task}")).toBeInTheDocument();
  expect(screen.getByLabelText("VLM Sample Frames")).toHaveValue(8);
  expect(screen.getByText("API key configured")).toBeInTheDocument();
  expect(screen.getByLabelText("VLM API Key")).toHaveValue("");
  expect(screen.queryByDisplayValue("secret-key")).not.toBeInTheDocument();
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
      json: async () => filterRunResponse(),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        run_id: "run-2",
        status: "succeeded",
        summary: cleaningSummary(),
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => filterRunResponse(),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await user.click(screen.getByRole("button", { name: "Import dataset" }));
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(await screen.findByText("3 issues found")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Add to cleaning Pipeline" }));

  expect(fetch).toHaveBeenCalledWith(
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

  expect(await screen.findByRole("button", { name: "Previous episode" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Next episode" }));

  expect(await screen.findByRole("heading", { name: "Episode 000001" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/1/recording",
    expect.objectContaining({ method: "POST" }),
  );

  await user.click(screen.getByRole("button", { name: "Next episode" }));

  expect(await screen.findByRole("heading", { name: "Episode 000002" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Next episode" })).toBeDisabled();
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
      subtasks: [
        {
          start_frame: 0,
          end_frame: 30,
          start_seconds: 0,
          end_seconds: 3,
          prompt: "Close the laptop.",
          skill: "Rotation",
          track: "default",
          is_mistake: false,
        },
      ],
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

function exportResult(episodeCount: number) {
  return {
    output_path: "/tmp/project-1-export",
    report_path: "/tmp/conversion_report.json",
    format: "act_hdf5",
    episode_count: episodeCount,
  };
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

function filterSummaryResponse() {
  const passed = { count: 0, status: "passed", skipped_reason: null };
  const review = { count: 1, status: "review", skipped_reason: null };
  return {
    dataset_path: "/tmp/pusht",
    total_episodes: 3,
    total_frames: 421,
    stages: [
      { id: "sudden_change", label: "Sudden change", count: 1, status: "review", skipped_reason: null },
      { id: "state_action_alignment", label: "Time sync", count: 0, status: "passed", skipped_reason: null },
      { id: "extreme_value", label: "Extreme value", count: 0, status: "passed", skipped_reason: null },
      { id: "kinematic_consistency", label: "Kinematic consistency", count: 0, status: "passed", skipped_reason: null },
      { id: "orientation_alignment", label: "Orientation alignment", count: 0, status: "passed", skipped_reason: null },
    ],
    episodes: [
      {
        episode_index: 0,
        stage_status: {
          sudden_change: passed,
          state_action_alignment: passed,
          extreme_value: passed,
          kinematic_consistency: passed,
          orientation_alignment: passed,
        },
      },
      {
        episode_index: 1,
        stage_status: {
          sudden_change: review,
          state_action_alignment: passed,
          extreme_value: passed,
          kinematic_consistency: passed,
          orientation_alignment: passed,
        },
      },
      {
        episode_index: 2,
        stage_status: {
          sudden_change: passed,
          state_action_alignment: passed,
          extreme_value: passed,
          kinematic_consistency: passed,
          orientation_alignment: passed,
        },
      },
    ],
  };
}

function filterRunResponse() {
  return {
    run_id: "filter-run-1",
    status: "succeeded",
    summary: filterSummaryResponse(),
  };
}

function filterDetailResponse(stageId: "extreme_value" | "kinematic_consistency") {
  if (stageId === "kinematic_consistency") {
    return {
      stage_id: "kinematic_consistency",
      episode_index: 0,
      title: "运动学一致性",
      status: "skipped",
      series: {},
      thresholds: {},
      table_rows: [],
      parameters: {
        urdf_path: null,
        end_effector_link: null,
        joint_names: [],
        joint_state_indices: [],
        eef_position_indices: [],
      },
      findings: [{ code: "backend_missing", severity: "warn", message: "Pinocchio 未安装，运动学一致性暂不可运行。" }],
      skipped_reason: "backend_missing",
    };
  }
  return {
    stage_id: "extreme_value",
    episode_index: 0,
    title: "极值检测",
    status: "review",
    series: {
      "state[0]": [0, 0.2, 0.4, 1.2],
      "action[0]": [0, 0.1, 0.5, 1.4],
    },
    thresholds: {
      "state[0]": { q01: -0.2, q99: 1, low: -0.5, high: 1.3 },
    },
    table_rows: [{ frame: 12, dimension: "state[0]", value: 1.5, low: -0.5, high: 1.3, gripper_exempt: false }],
    parameters: { alpha: 0.5, q01: 0.01, q99: 0.99, gripper_exempt: [6, 13] },
    findings: [{ code: "extreme_value", severity: "warn", message: "检测到越界帧。数量：1" }],
    skipped_reason: null,
  };
}
