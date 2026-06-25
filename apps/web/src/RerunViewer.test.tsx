import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

import RerunViewer from "./RerunViewer";

const start = vi.fn();
const stop = vi.fn();
const on = vi.fn();
const setActiveTimeline = vi.fn();
const setPlaying = vi.fn();
const getPlaying = vi.fn();
const getCurrentTime = vi.fn();
const setCurrentTime = vi.fn();
const getTimeRange = vi.fn();
const overridePanelState = vi.fn();
const handlers = new Map<string, Array<(event: unknown) => void>>();

vi.mock("@rerun-io/web-viewer", () => ({
  WebViewer: class {
    start = start;
    stop = stop;
    on = on;
    set_active_timeline = setActiveTimeline;
    set_playing = setPlaying;
    get_playing = getPlaying;
    get_current_time = getCurrentTime;
    set_current_time = setCurrentTime;
    get_time_range = getTimeRange;
    override_panel_state = overridePanelState;
  },
}));

beforeEach(() => {
  start.mockResolvedValue(undefined);
  stop.mockReset();
  setActiveTimeline.mockReset();
  setPlaying.mockReset();
  getPlaying.mockReset();
  getPlaying.mockReturnValue(false);
  getCurrentTime.mockReset();
  setCurrentTime.mockReset();
  getTimeRange.mockReset();
  overridePanelState.mockReset();
  handlers.clear();
  on.mockReset();
  on.mockImplementation((event: string, callback: (event: unknown) => void) => {
    handlers.set(event, [...(handlers.get(event) ?? []), callback]);
    return vi.fn();
  });
});

function emitViewerEvent(event: string, payload: unknown) {
  act(() => {
    for (const handler of handlers.get(event) ?? []) handler(payload);
  });
}

test("loads the Rerun WASM bundle from the public asset directory without internal panel overrides", () => {
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  expect(start).toHaveBeenCalledWith(
    new URL("/api/artifacts/episode.rrd", window.location.href).toString(),
    expect.any(HTMLDivElement),
    expect.objectContaining({
      base_url: new URL("/rerun/", window.location.href).toString(),
    }),
  );
  expect(start.mock.calls[0][2]).not.toHaveProperty("panel_state_overrides");
});

test("keeps the built-in Rerun time panel visible for the time-series cursor", () => {
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  emitViewerEvent("ready", undefined);

  expect(overridePanelState).not.toHaveBeenCalled();
});

test("toggles playback through the active Rerun recording", async () => {
  const user = userEvent.setup();
  getTimeRange.mockReturnValue({ min: 0, max: 10_000_000_000 });
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  emitViewerEvent("recording_open", { recording_id: "recording-1" });
  await user.click(screen.getByRole("button", { name: "Play" }));

  expect(setActiveTimeline).toHaveBeenCalledWith("recording-1", "episode_time");
  expect(setPlaying).toHaveBeenCalledWith("recording-1", true);

  emitViewerEvent("play", { recording_id: "recording-1" });
  await user.click(screen.getByRole("button", { name: "Pause" }));

  expect(setPlaying).toHaveBeenLastCalledWith("recording-1", false);
});

test("seeks on episode_time with clamped one-second jumps", async () => {
  const user = userEvent.setup();
  getTimeRange.mockReturnValue({ min: 0, max: 6_000_000_000 });
  getCurrentTime.mockReturnValue(5_500_000_000);
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  emitViewerEvent("recording_open", { recording_id: "recording-1" });
  await user.click(screen.getByRole("button", { name: "Forward" }));

  expect(setCurrentTime).toHaveBeenCalledWith("recording-1", "episode_time", 6_000_000_000);
});

test("accelerates repeated same-direction seeks and resets after the burst window", async () => {
  let now = 1_000;
  vi.spyOn(Date, "now").mockImplementation(() => now);
  getTimeRange.mockReturnValue({ min: 0, max: 20_000_000_000 });
  getCurrentTime.mockReturnValue(0);
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  emitViewerEvent("recording_open", { recording_id: "recording-1" });
  const forwardButton = screen.getByRole("button", { name: "Forward" });
  fireEvent.click(forwardButton);
  now = 1_200;
  fireEvent.click(forwardButton);
  now = 2_200;
  fireEvent.click(forwardButton);

  expect(setCurrentTime).toHaveBeenNthCalledWith(1, "recording-1", "episode_time", 1_000_000_000);
  expect(setCurrentTime).toHaveBeenNthCalledWith(2, "recording-1", "episode_time", 2_000_000_000);
  expect(setCurrentTime).toHaveBeenNthCalledWith(3, "recording-1", "episode_time", 1_000_000_000);
});

test("accepts Chinese playback labels", async () => {
  const user = userEvent.setup();
  getTimeRange.mockReturnValue({ min: 0, max: 10_000_000_000 });
  render(
    <RerunViewer
      recordingUrl="/api/artifacts/episode.rrd"
      labels={{
        previousEpisode: "上一集",
        rewind: "后退",
        play: "播放",
        pause: "暂停",
        forward: "前进",
        nextEpisode: "下一集",
        host: "Rerun episode replay",
        controls: "Rerun playback controls",
      }}
    />,
  );

  emitViewerEvent("recording_open", { recording_id: "recording-1" });
  await user.click(screen.getByRole("button", { name: "播放" }));

  expect(setPlaying).toHaveBeenCalledWith("recording-1", true);
});
