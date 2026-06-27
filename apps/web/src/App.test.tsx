import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

test("episode list fills the remaining sidebar height instead of ending early", async () => {
  // @ts-ignore Vitest runs in Node; the app tsconfig intentionally does not include Node ambient types.
  const { readFileSync } = await import("node:fs");
  const stylesCss = readFileSync("src/styles.css", "utf8");

  expect(stylesCss).toContain("contain: size;");
  expect(stylesCss).toContain("display: grid;");
  expect(stylesCss).toContain("grid-template-rows: 58px minmax(0, 1fr)");
  expect(stylesCss).toContain(".episode-list { min-height: 0; overflow: auto;");
  expect(stylesCss).not.toContain(".episode-list { height: 620px;");
  expect(stylesCss).toContain("flex-direction: column;");
  expect(stylesCss).toContain(".viewer-stage { flex: 0 0 530px; min-height: 0; height: 530px;");
  expect(stylesCss).toContain("margin-top: auto;");
});

async function importDatasetForTest(user: ReturnType<typeof userEvent.setup>, path = "/tmp/pusht") {
  fireEvent.change(screen.getByLabelText("Dataset path"), { target: { value: path } });
  await user.click(screen.getByRole("button", { name: "Import dataset" }));
}

async function clearDefaultSidebarFilters(user: ReturnType<typeof userEvent.setup>) {
  const sidebar = document.querySelector(".sidebar-tools");
  if (!sidebar) throw new Error("Sidebar filters are not rendered");
  const filters = within(sidebar as HTMLElement);
  await user.click(filters.getByLabelText("Blurred frames"));
  for (const filter of filters.getAllByLabelText("Time sync")) {
    await user.click(filter);
  }
  await user.click(filters.getByLabelText("Action jump"));
  await user.click(filters.getByLabelText("Sudden change"));
  await user.click(filters.getByLabelText("Extreme value"));
}

beforeEach(() => {
  installMemoryStorage();
  window.localStorage.clear();
});

test("starts with an empty dataset path and prompts for a local dataset", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => formatsResponse(),
    }),
  );
  renderApp();

  const input = await screen.findByLabelText("Dataset path");

  expect(input).toHaveValue("");
  expect(input).toHaveAttribute("placeholder", "Enter a local dataset path");
  expect(screen.getByRole("button", { name: "Import dataset" })).toBeDisabled();
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
  expect(screen.getByText("输入本地 dataset 路径后，即可检查并清洗 robot episodes。")).toBeInTheDocument();
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

  await importDatasetForTest(user);

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

  await importDatasetForTest(user);
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
  await user.selectOptions(screen.getByLabelText("Import format"), "act_hdf5");
  await importDatasetForTest(user, "data/samples/aloha_static_coffee");

  expect(fetch).toHaveBeenNthCalledWith(
    2,
    "/api/projects",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ path: "data/samples/aloha_static_coffee", format_hint: "act_hdf5" }),
    }),
  );
});

test("selects configured sidebar filters and leaves setup-dependent filters unchecked on import", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() }),
  );
  renderApp();

  await importDatasetForTest(user);

  const sidebar = document.querySelector(".sidebar-tools");
  expect(sidebar).not.toBeNull();
  const filters = within(sidebar as HTMLElement);
  expect(await filters.findByLabelText("Blurred frames")).toBeChecked();
  expect(filters.getAllByLabelText("Time sync")).toHaveLength(2);
  for (const filter of filters.getAllByLabelText("Time sync")) {
    expect(filter).toBeChecked();
  }
  expect(filters.getByLabelText("Action jump")).toBeChecked();
  expect(filters.getByLabelText("Sudden change")).toBeChecked();
  expect(filters.getByLabelText("Extreme value")).toBeChecked();
  expect(filters.getByLabelText("VLM failed")).not.toBeChecked();
  expect(filters.getByLabelText("Kinematic consistency")).not.toBeChecked();
  expect(filters.getByLabelText("Orientation alignment")).not.toBeChecked();
});

test("expands only one sidebar weight slider at a time", async () => {
  const user = userEvent.setup();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() }),
  );
  renderApp();

  await importDatasetForTest(user);

  const sidebar = document.querySelector(".sidebar-tools");
  expect(sidebar).not.toBeNull();
  const filters = within(sidebar as HTMLElement);
  expect(filters.queryByLabelText("Sudden change weight")).not.toBeInTheDocument();

  await user.click(filters.getByRole("button", { name: "Expand Sudden change weight" }));
  expect(filters.getByLabelText("Sudden change weight")).toHaveValue("1.5");
  expect(filters.getByRole("button", { name: "Collapse Sudden change weight" })).toHaveAttribute(
    "aria-expanded",
    "true",
  );

  await user.click(filters.getByRole("button", { name: "Expand Time sync weight" }));
  expect(filters.queryByLabelText("Sudden change weight")).not.toBeInTheDocument();
  expect(filters.getByLabelText("Time sync weight")).toHaveValue("1.5");

  await user.click(filters.getByRole("button", { name: "Collapse Time sync weight" }));
  expect(filters.queryByLabelText("Time sync weight")).not.toBeInTheDocument();
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

  await importDatasetForTest(user);
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

test("normalizes quoted output folders before exporting", async () => {
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

  await importDatasetForTest(user);
  await user.clear(await screen.findByLabelText("Output folder"));
  await user.type(screen.getByLabelText("Output folder"), "'/tmp/user exports'");
  await user.click(screen.getByRole("button", { name: "Export 1 episode" }));

  expect(fetch).toHaveBeenLastCalledWith(
    "/api/projects/project-1/exports",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        episode_indexes: [0],
        format: "act_hdf5",
        options: { output_dir: "/tmp/user exports" },
      }),
    }),
  );
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

  await importDatasetForTest(user);
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

  await importDatasetForTest(user);
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

  await importDatasetForTest(user);
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

test("shows the checked episode count as the main export count", async () => {
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

  await importDatasetForTest(user);
  await user.click(screen.getByLabelText("Add Episode 000000 to export"));
  await user.click(screen.getByLabelText("Add Episode 000001 to export"));
  await user.click(screen.getByLabelText("Add Episode 000002 to export"));

  const exportCount = document.querySelector(".export-count");
  expect(exportCount).toHaveTextContent("Episodes33 checked");
  expect(screen.getByRole("button", { name: "Export 1 episode" })).toBeInTheDocument();
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
    .mockResolvedValueOnce(pipelineJsonResponse())
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await clearDefaultSidebarFilters(user);
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
    .mockResolvedValueOnce(pipelineJsonResponse())
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

  await importDatasetForTest(user);
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

  await importDatasetForTest(user);

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
    .mockResolvedValueOnce(pipelineJsonResponse())
    .mockResolvedValueOnce({
      ok: true,
      json: async () => exportResult(1),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  expect((await screen.findAllByText("Review")).length).toBeGreaterThan(0);
  await clearDefaultSidebarFilters(user);

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  await user.click(within(episodePanel as HTMLElement).getByLabelText("Sudden change"));
  expect(within(episodePanel as HTMLElement).getByText("#000001")).toBeInTheDocument();
  expect(within(episodePanel as HTMLElement).queryByText("#000000")).not.toBeInTheDocument();

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
      .mockResolvedValueOnce(pipelineJsonResponse()),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect((await screen.findAllByText("Review")).length).toBeGreaterThan(0);
  expect(screen.getAllByText("Exclude").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Pass").length).toBeGreaterThan(0);
  expect(screen.getAllByText("#000000").length).toBeGreaterThan(0);
  expect(screen.getAllByText("72 / 100").length).toBeGreaterThan(0);
  expect(screen.getByText("Data quality")).toBeInTheDocument();
  expect(screen.getByText("Task success")).toBeInTheDocument();
  expect(screen.getByText("Not evaluated")).toBeInTheDocument();
});

test("marks saved results from an older scorer as requiring a rerun", async () => {
  const user = userEvent.setup();
  const summary = cleaningSummary();
  summary.requires_rerun = true;
  summary.previous_scorer_version = "score_lerobot_episodes-compatible-v1";
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
      .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
      .mockResolvedValueOnce(pipelineJsonResponse(summary)),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(screen.getByText("Rerun required")).toBeInTheDocument();
  expect(
    screen.getByText("These saved scores use an older scoring version. Run the pipeline again."),
  ).toBeInTheDocument();
});

test("shows passed episodes in the status folder when default issue filters are selected", async () => {
  const user = userEvent.setup();
  const allPassedSummary = cleaningSummary();
  allPassedSummary.passed_count = 3;
  allPassedSummary.review_count = 0;
  allPassedSummary.excluded_count = 0;
  allPassedSummary.results = allPassedSummary.results.map((result) => ({
    ...result,
    score: 0.82,
    status: "passed",
    findings: [],
  }));
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
      .mockResolvedValueOnce(pipelineJsonResponse(allPassedSummary)),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  const panel = within(episodePanel as HTMLElement);
  expect(panel.getByRole("button", { name: "Collapse Pass folder" })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000000/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000001/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000002/ })).toBeInTheDocument();
});

test("collapses and expands cleaning status folders independently", async () => {
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
      .mockResolvedValueOnce(pipelineJsonResponse()),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await clearDefaultSidebarFilters(user);

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  const panel = within(episodePanel as HTMLElement);
  expect(panel.getByRole("button", { name: /#000000/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000001/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000002/ })).toBeInTheDocument();

  await user.click(panel.getByRole("button", { name: "Collapse Review folder" }));
  expect(panel.queryByRole("button", { name: /#000000/ })).not.toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000001/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000002/ })).toBeInTheDocument();
  expect(panel.getByRole("button", { name: "Expand Review folder" })).toHaveAttribute("aria-expanded", "false");

  await user.click(panel.getByRole("button", { name: "Collapse Exclude folder" }));
  expect(panel.queryByRole("button", { name: /#000001/ })).not.toBeInTheDocument();
  expect(panel.getByRole("button", { name: /#000002/ })).toBeInTheDocument();

  await user.click(panel.getByRole("button", { name: "Expand Review folder" }));
  expect(panel.getByRole("button", { name: /#000000/ })).toBeInTheDocument();
  expect(panel.queryByRole("button", { name: /#000001/ })).not.toBeInTheDocument();
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
    .mockResolvedValueOnce(pipelineJsonResponse())
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

  await importDatasetForTest(user);
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

test("advances to the next review episode after a manual decision", async () => {
  const user = userEvent.setup();
  const summary = cleaningSummary();
  summary.review_count = 2;
  summary.excluded_count = 0;
  summary.results[1] = { ...summary.results[1], score: 0.68, status: "review" };
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
    .mockResolvedValueOnce(pipelineJsonResponse(summary))
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recording_url: "/api/artifacts/episode-000000.rrd" }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...summary.results[0],
        status: "passed",
        source: "manual",
      }),
    })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "warmed" }) })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recording_url: "/api/artifacts/episode-000001.rrd" }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click(await screen.findByRole("button", { name: "Replay in Rerun" }));
  await user.click(await screen.findByRole("button", { name: "Pass" }));

  expect(await screen.findByRole("heading", { name: "Episode 000001" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/1/recording",
    expect.objectContaining({ method: "POST" }),
  );
});

test("advances to the next passed episode after excluding from the pass folder", async () => {
  const user = userEvent.setup();
  const summary = cleaningSummary();
  summary.passed_count = 2;
  summary.review_count = 0;
  summary.excluded_count = 1;
  summary.results = [
    { ...summary.results[0], episode_index: 0, score: 0.7, status: "passed", findings: [] },
    { ...summary.results[1], episode_index: 1, score: 0.9, status: "passed", findings: [] },
    { ...summary.results[2], episode_index: 2, score: 0.95, status: "excluded", findings: [] },
  ];
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
    .mockResolvedValueOnce(pipelineJsonResponse(summary))
    .mockResolvedValueOnce({ ok: true, json: async () => ({ recording_url: "/api/artifacts/episode-000000.rrd" }) })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...summary.results[0], status: "excluded", source: "manual" }),
    })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "warmed" }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ recording_url: "/api/artifacts/episode-000001.rrd" }) });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click(await screen.findByRole("button", { name: "Replay in Rerun" }));
  await user.click(await screen.findByRole("button", { name: "Exclude" }));

  expect(await screen.findByRole("heading", { name: "Episode 000001" })).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/1/recording",
    expect.objectContaining({ method: "POST" }),
  );
});

test("stays on the current episode when its status folder is emptied by the decision", async () => {
  const user = userEvent.setup();
  const summary = cleaningSummary();
  summary.passed_count = 1;
  summary.review_count = 0;
  summary.excluded_count = 2;
  summary.results = [
    { ...summary.results[0], episode_index: 0, score: 0.3, status: "passed", findings: [] },
    { ...summary.results[1], episode_index: 1, score: 0.5, status: "excluded", findings: [] },
    { ...summary.results[2], episode_index: 2, score: 0.6, status: "excluded", findings: [] },
  ];
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
    .mockResolvedValueOnce(pipelineJsonResponse(summary))
    .mockResolvedValueOnce({ ok: true, json: async () => ({ recording_url: "/api/artifacts/episode-000000.rrd" }) })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...summary.results[0], status: "excluded", source: "manual" }),
    });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click(await screen.findByRole("button", { name: "Replay in Rerun" }));
  await user.click(await screen.findByRole("button", { name: "Exclude" }));

  await waitFor(() =>
    expect(fetch).toHaveBeenLastCalledWith(
      "/api/projects/project-1/episodes/0/decision",
      expect.objectContaining({ method: "PATCH" }),
    ),
  );
  expect(screen.getByRole("heading", { name: "Episode 000000" })).toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/1/recording",
    expect.objectContaining({ method: "POST" }),
  );
  expect(fetch).not.toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/2/recording",
    expect.objectContaining({ method: "POST" }),
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
      .mockResolvedValueOnce(pipelineJsonResponse()),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await clearDefaultSidebarFilters(user);
  await user.type(await screen.findByLabelText("Search / filter"), "000001");

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  expect(within(episodePanel as HTMLElement).queryByText("#000000")).not.toBeInTheDocument();
  expect(within(episodePanel as HTMLElement).getByText("#000001")).toBeInTheDocument();

  await user.clear(screen.getByLabelText("Search / filter"));
  await user.click(screen.getByLabelText("VLM failed"));

  expect(within(episodePanel as HTMLElement).queryByText("#000000")).not.toBeInTheDocument();
  expect(within(episodePanel as HTMLElement).getByText("#000001")).toBeInTheDocument();
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

  await importDatasetForTest(user);

  expect((await screen.findAllByText("Sudden change")).length).toBeGreaterThan(0);
  expect(screen.getAllByText("Time sync").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Extreme value").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Kinematic consistency").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Orientation alignment").length).toBeGreaterThan(0);
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

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Extreme value" }));

  expect(await screen.findByRole("heading", { name: "Extreme value" })).toBeInTheDocument();
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

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Kinematic consistency" }));

  expect(await screen.findByRole("heading", { name: "Kinematic consistency" })).toBeInTheDocument();
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
    .mockResolvedValueOnce(pipelineJsonResponse());
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
  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/pipeline/runs/stream",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        pass_threshold: 0.8,
        review_threshold: 0.6,
        enabled_filter_stages: [
          "sudden_change",
          "state_action_alignment",
          "extreme_value",
          "kinematic_consistency",
          "orientation_alignment",
        ],
        quality_weights: {
          sudden_change: 1.5,
          state_action_alignment: 1.5,
          extreme_value: 2,
          kinematic_consistency: 2,
          orientation_alignment: 1,
          task_success: 2,
        },
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

test("sends selected cleaning rules and slider weights when running cleaning", async () => {
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
    .mockResolvedValueOnce(pipelineJsonResponse());
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  const sidebar = document.querySelector(".sidebar-tools");
  expect(sidebar).not.toBeNull();
  const filters = within(sidebar as HTMLElement);
  expect(filters.queryByLabelText("Sudden change weight")).not.toBeInTheDocument();
  await user.click(filters.getByRole("button", { name: "Expand Sudden change weight" }));
  expect(filters.getByLabelText("Sudden change weight")).toHaveValue("1.5");
  expect(screen.queryByText("Cleaning rules")).not.toBeInTheDocument();
  await user.click(filters.getByRole("button", { name: "Expand Time sync weight" }));
  fireEvent.change(filters.getByLabelText("Time sync weight"), { target: { value: "2.5" } });
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/pipeline/runs/stream",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        pass_threshold: 0.8,
        review_threshold: 0.6,
        enabled_filter_stages: [
          "sudden_change",
          "state_action_alignment",
          "extreme_value",
          "kinematic_consistency",
          "orientation_alignment",
        ],
        quality_weights: {
          sudden_change: 1.5,
          state_action_alignment: 2.5,
          extreme_value: 2,
          kinematic_consistency: 2,
          orientation_alignment: 1,
        },
        vlm: {
          enabled: false,
          provider: "OpenAI",
          model: "gpt-4o-mini",
          api_base_url: null,
          prompt:
            "You are an automated robot episode evaluator. Return only JSON with success, score, and reason. Judge whether the task was successfully completed from the visual evidence.",
          sample_frames: 4,
        },
      }),
    }),
  );
});

test("runs cleaning only for the selected episode from the viewer toolbar", async () => {
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
    .mockResolvedValueOnce(pipelineJsonResponse());
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run selected episode" }));

  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/pipeline/runs/stream",
    expect.objectContaining({
      method: "POST",
      body: expect.stringContaining('"episode_indexes":[0]'),
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

  await importDatasetForTest(user);
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

  await importDatasetForTest(user);
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
    .mockResolvedValueOnce(pipelineJsonResponse())
    .mockResolvedValueOnce(pipelineJsonResponse());
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(await screen.findByText("3 issues found")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Add to cleaning Pipeline" }));

  expect(fetch).toHaveBeenCalledWith(
    "/api/projects/project-1/pipeline/runs/stream",
    expect.objectContaining({ method: "POST" }),
  );
});

test("shows the cleaning summary in the viewer after running the pipeline", async () => {
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
      .mockResolvedValueOnce(pipelineJsonResponse()),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  expect(await screen.findByRole("heading", { name: "Cleaning summary" })).toBeInTheDocument();
  expect(screen.getByText("Pass 1")).toBeInTheDocument();
  expect(screen.getByText("Review 1")).toBeInTheDocument();
  expect(screen.getByText("Exclude 1")).toBeInTheDocument();
  expect(screen.getByText("Lowest score episodes")).toBeInTheDocument();
  expect(screen.getByLabelText("Episode 000001 score 41 status Exclude")).toBeInTheDocument();

  const scoreBars = within(screen.getByRole("list", { name: "Episode cleaning scores" })).getAllByRole("button");
  expect(scoreBars.map((bar) => bar.getAttribute("aria-label"))).toEqual([
    "Episode 000001 score 41 status Exclude",
    "Episode 000000 score 72 status Review",
    "Episode 000002 score 96 status Pass",
  ]);
  expect(scoreBars[1]).toHaveClass("review");

  const scoreChart = document.querySelector(".score-chart");
  expect(scoreChart).not.toBeNull();
  expect(within(scoreChart as HTMLElement).queryByText("#000001")).not.toBeInTheDocument();

  const viewerStage = document.querySelector(".viewer-stage");
  expect(viewerStage).not.toBeNull();
  expect(viewerStage).toHaveClass("viewer-stage-scroll");
});

test("selecting an episode shows the ready-to-build placeholder without auto-building the replay", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
    .mockResolvedValueOnce(pipelineJsonResponse());
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  expect(await screen.findByRole("heading", { name: "Cleaning summary" })).toBeInTheDocument();

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  await user.click(within(episodePanel as HTMLElement).getByRole("button", { name: /#000000/ }));

  expect(await screen.findByRole("heading", { name: "Rerun replay is ready to build" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Cleaning summary" })).not.toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/0/recording",
    expect.objectContaining({ method: "POST" }),
  );
});

test("warms the recording for the selected episode in the background without building it", async () => {
  const user = userEvent.setup();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "warmed" }) });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);

  const episodePanel = document.querySelector(".episode-panel");
  expect(episodePanel).not.toBeNull();
  await user.click(within(episodePanel as HTMLElement).getByRole("button", { name: /Episode 000000/ }));

  await waitFor(() =>
    expect(fetch).toHaveBeenCalledWith(
      "/api/projects/project-1/episodes/0/recording/warm",
      expect.objectContaining({ method: "POST" }),
    ),
  );
  expect(await screen.findByRole("heading", { name: "Rerun replay is ready to build" })).toBeInTheDocument();
  expect(fetch).not.toHaveBeenCalledWith(
    "/api/projects/project-1/episodes/0/recording",
    expect.objectContaining({ method: "POST" }),
  );
});

test("shows only valid manual decisions for passed and excluded episodes", async () => {
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
      .mockResolvedValueOnce(pipelineJsonResponse()),
  );
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));
  await user.click((await screen.findAllByRole("button", { name: /#000002/ }))[0]);

  expect(screen.queryByRole("button", { name: "Pass" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Exclude" })).toBeInTheDocument();

  await user.click(screen.getAllByRole("button", { name: /#000001/ })[0]);

  expect(screen.getByRole("button", { name: "Pass" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Exclude" })).not.toBeInTheDocument();
});

test("renders a progress line on the Run cleaning Pipeline button while streaming", async () => {
  const user = userEvent.setup();
  const encoder = new TextEncoder();
  const controllerRef: {
    current: ReadableStreamDefaultController<Uint8Array> | null;
  } = { current: null };
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef.current = controller;
      controller.enqueue(
        encoder.encode(
          `event: progress\ndata: ${JSON.stringify({ phase: "cleaning", completed: 1, total: 3 })}\n\n` +
            `event: progress\ndata: ${JSON.stringify({ phase: "filters", completed: 3, total: 6 })}\n\n`,
        ),
      );
    },
  });
  const fetch = vi
    .fn()
    .mockResolvedValueOnce({ ok: true, json: async () => formatsResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => projectResponse() })
    .mockResolvedValueOnce({ ok: true, json: async () => episodesResponse() })
    .mockResolvedValueOnce({ ok: true, body: stream });
  vi.stubGlobal("fetch", fetch);
  renderApp();

  await importDatasetForTest(user);
  await user.click(await screen.findByRole("button", { name: "Run cleaning Pipeline" }));

  const button = document.querySelector(".viewer-toolbar .actions .has-progress") as HTMLButtonElement;
  await waitFor(() => expect(button.textContent).toContain("Cleaning"));
  const line = button.querySelector(".button-progress-line") as HTMLElement | null;
  expect(line).not.toBeNull();
  await waitFor(() => expect(line).toHaveStyle({ left: "35%" }));

  controllerRef.current?.enqueue(
    encoder.encode(
      `event: done\ndata: ${JSON.stringify({
        cleaning: { run_id: "run-1", status: "succeeded", summary: cleaningSummary() },
        filters: filterRunResponse(),
      })}\n\n`,
    ),
  );
  controllerRef.current?.close();

  await screen.findByRole("button", { name: "Run cleaning Pipeline" });
  expect(button.querySelector(".button-progress-line")).toBeNull();
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
    scorer_version: "quality-rules-v2",
    requires_rerun: false,
    previous_scorer_version: null as string | null,
    results: [
      {
        episode_index: 0,
        score: 0.72,
        data_quality_score: 0.72,
        task_success_score: null,
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
        data_quality_score: 0.41,
        task_success_score: null,
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
        data_quality_score: 0.96,
        task_success_score: null,
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

function pipelineJsonResponse(
  cleaning: ReturnType<typeof cleaningSummary> = cleaningSummary(),
  filters: ReturnType<typeof filterRunResponse> = filterRunResponse(),
) {
  return {
    ok: true,
    json: async () => ({
      cleaning: { run_id: "run-1", status: "succeeded", summary: cleaning },
      filters,
    }),
  };
}

function filterDetailResponse(stageId: "extreme_value" | "kinematic_consistency") {
  if (stageId === "kinematic_consistency") {
    return {
      stage_id: "kinematic_consistency",
      episode_index: 0,
      title: "Kinematic consistency",
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
      findings: [{ code: "backend_missing", severity: "warn", message: "Pinocchio not installed; kinematic consistency unavailable." }],
      skipped_reason: "backend_missing",
    };
  }
  return {
    stage_id: "extreme_value",
    episode_index: 0,
    title: "Extreme value",
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
    findings: [{ code: "extreme_value", severity: "warn", message: "Detected out-of-bounds frames. Count: 1" }],
    skipped_reason: null,
  };
}
