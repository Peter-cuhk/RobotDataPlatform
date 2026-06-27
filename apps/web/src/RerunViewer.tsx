import { WebViewer } from "@rerun-io/web-viewer";
import { useEffect, useRef } from "react";

const episodeTimeline = "episode_time";

type ViewerEventPayload = {
  recording_id: string;
  timeline?: string;
  time?: number;
};

type RerunViewerProps = {
  recordingUrl: string;
};

export default function RerunViewer({ recordingUrl }: RerunViewerProps) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!host.current) return;
    const viewer = new WebViewer();
    const options = {
      width: "100%",
      height: "100%",
      hide_welcome_screen: true,
      theme: "dark",
      base_url: new URL("/rerun/", window.location.href).toString(),
    } as const;
    const absoluteRecordingUrl = new URL(recordingUrl, window.location.href).toString();
    const offRecordingOpen = viewer.on("recording_open", (event: ViewerEventPayload) => {
      viewer.set_active_timeline(event.recording_id, episodeTimeline);
    });

    void viewer.start(absoluteRecordingUrl, host.current, options);
    return () => {
      offRecordingOpen();
      viewer.stop();
    };
  }, [recordingUrl]);

  return (
    <div className="rerun-shell">
      <div className="rerun-host" ref={host} />
    </div>
  );
}
