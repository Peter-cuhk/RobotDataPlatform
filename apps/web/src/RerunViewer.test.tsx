import { render } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import RerunViewer from "./RerunViewer";

const start = vi.fn();

vi.mock("@rerun-io/web-viewer", () => ({
  WebViewer: class {
    start = start;
    stop = vi.fn();
  },
}));

test("loads the Rerun WASM bundle from the public asset directory", () => {
  render(<RerunViewer recordingUrl="/api/artifacts/episode.rrd" />);

  expect(start).toHaveBeenCalledWith(
    new URL("/api/artifacts/episode.rrd", window.location.href).toString(),
    expect.any(HTMLDivElement),
    expect.objectContaining({
      base_url: new URL("/rerun/", window.location.href).toString(),
    }),
  );
});
