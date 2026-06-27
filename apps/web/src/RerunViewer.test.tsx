import { render } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

import RerunViewer from "./RerunViewer";

const start = vi.fn();
const stop = vi.fn();
const on = vi.fn();
const setActiveTimeline = vi.fn();
const overridePanelState = vi.fn();
const handlers = new Map<string, Array<(event: unknown) => void>>();

vi.mock("@rerun-io/web-viewer", () => ({
  WebViewer: class {
    start = start;
    stop = stop;
    on = on;
    set_active_timeline = setActiveTimeline;
    override_panel_state = overridePanelState;
  },
}));

beforeEach(() => {
  start.mockResolvedValue(undefined);
  stop.mockReset();
  setActiveTimeline.mockReset();
  overridePanelState.mockReset();
  handlers.clear();
  on.mockReset();
  on.mockImplementation((event: string, callback: (event: unknown) => void) => {
    handlers.set(event, [...(handlers.get(event) ?? []), callback]);
    return vi.fn();
  });
});

function emitViewerEvent(event: string, payload: unknown) {
  for (const handler of handlers.get(event) ?? []) handler(payload);
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
