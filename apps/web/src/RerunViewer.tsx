import { WebViewer } from "@rerun-io/web-viewer";
import { useEffect, useRef } from "react";

export default function RerunViewer({ recordingUrl }: { recordingUrl: string }) {
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
    void viewer.start(absoluteRecordingUrl, host.current, options);
    return () => viewer.stop();
  }, [recordingUrl]);

  return <div className="rerun-host" ref={host} aria-label="Rerun episode replay" />;
}
