import { WebViewer } from "@rerun-io/web-viewer";
import { useEffect, useRef, useState } from "react";

const episodeTimeline = "episode_time";
const nanosPerSecond = 1_000_000_000;
const seekStepsSeconds = [1, 2, 5, 10] as const;
const seekBurstWindowMs = 800;

type ViewerEventPayload = {
  recording_id: string;
  timeline?: string;
  time?: number;
};

type RerunViewerProps = {
  recordingUrl: string;
  canGoPrevious?: boolean;
  canGoNext?: boolean;
  onPreviousEpisode?: () => void;
  onNextEpisode?: () => void;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export default function RerunViewer({
  recordingUrl,
  canGoPrevious = false,
  canGoNext = false,
  onPreviousEpisode,
  onNextEpisode,
}: RerunViewerProps) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<WebViewer | null>(null);
  const recordingId = useRef<string | null>(null);
  const timeRange = useRef<{ min: number; max: number } | null>(null);
  const currentTime = useRef(0);
  const seekBurst = useRef<{ direction: -1 | 1; lastAt: number; stepIndex: number } | null>(null);
  const [activeRecordingId, setActiveRecordingId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!host.current) return;
    setActiveRecordingId(null);
    setPlaying(false);
    const nextViewer = new WebViewer();
    viewer.current = nextViewer;
    const options = {
      width: "100%",
      height: "100%",
      hide_welcome_screen: true,
      theme: "dark",
      base_url: new URL("/rerun/", window.location.href).toString(),
    } as const;
    const absoluteRecordingUrl = new URL(recordingUrl, window.location.href).toString();
    const offReady = nextViewer.on("ready", () => {
      nextViewer.override_panel_state("time", "hidden");
    });
    const offRecordingOpen = nextViewer.on("recording_open", (event: ViewerEventPayload) => {
      recordingId.current = event.recording_id;
      setActiveRecordingId(event.recording_id);
      nextViewer.set_active_timeline(event.recording_id, episodeTimeline);
      timeRange.current = nextViewer.get_time_range(event.recording_id, episodeTimeline);
      currentTime.current = nextViewer.get_current_time(event.recording_id, episodeTimeline);
      setPlaying(nextViewer.get_playing(event.recording_id));
    });
    const offPlay = nextViewer.on("play", (event: ViewerEventPayload) => {
      if (event.recording_id === recordingId.current) setPlaying(true);
    });
    const offPause = nextViewer.on("pause", (event: ViewerEventPayload) => {
      if (event.recording_id === recordingId.current) setPlaying(false);
    });
    const offTimeUpdate = nextViewer.on("time_update", (event: ViewerEventPayload) => {
      if (event.recording_id === recordingId.current) currentTime.current = event.time ?? currentTime.current;
    });
    const offTimelineChange = nextViewer.on("timeline_change", (event: ViewerEventPayload) => {
      if (event.recording_id !== recordingId.current || event.timeline !== episodeTimeline) return;
      currentTime.current = event.time ?? currentTime.current;
    });

    void nextViewer.start(absoluteRecordingUrl, host.current, options);
    return () => {
      offReady();
      offRecordingOpen();
      offPlay();
      offPause();
      offTimeUpdate();
      offTimelineChange();
      nextViewer.stop();
      viewer.current = null;
      recordingId.current = null;
      timeRange.current = null;
      currentTime.current = 0;
      seekBurst.current = null;
    };
  }, [recordingUrl]);

  const timelineReady = Boolean(viewer.current && activeRecordingId);
  const togglePlayback = () => {
    if (!viewer.current || !recordingId.current) return;
    viewer.current.set_playing(recordingId.current, !playing);
  };
  const seek = (direction: -1 | 1) => {
    if (!viewer.current || !recordingId.current) return;
    const now = Date.now();
    const previousBurst = seekBurst.current;
    const stepIndex =
      previousBurst && previousBurst.direction === direction && now - previousBurst.lastAt <= seekBurstWindowMs
        ? Math.min(previousBurst.stepIndex + 1, seekStepsSeconds.length - 1)
        : 0;
    seekBurst.current = { direction, lastAt: now, stepIndex };

    const range = viewer.current.get_time_range(recordingId.current, episodeTimeline) ?? timeRange.current;
    if (!range) return;
    timeRange.current = range;
    const nextCurrentTime = viewer.current.get_current_time(recordingId.current, episodeTimeline);
    currentTime.current = nextCurrentTime;
    const offset = direction * seekStepsSeconds[stepIndex] * nanosPerSecond;
    const nextTime = clamp(nextCurrentTime + offset, range.min, range.max);
    currentTime.current = nextTime;
    viewer.current.set_current_time(recordingId.current, episodeTimeline, nextTime);
  };

  return (
    <div className="rerun-shell">
      <div className="rerun-host" ref={host} aria-label="Rerun episode replay" />
      <div className="rerun-controls" aria-label="Rerun playback controls">
        <button className="icon-button" aria-label="上一集" disabled={!canGoPrevious} onClick={onPreviousEpisode}>
          ‹‹
        </button>
        <button className="icon-button" aria-label="后退" disabled={!timelineReady} onClick={() => seek(-1)}>
          ‹
        </button>
        <button className="play-toggle" aria-label={playing ? "暂停" : "播放"} disabled={!timelineReady} onClick={togglePlayback}>
          {playing ? "暂停" : "播放"}
        </button>
        <button className="icon-button" aria-label="前进" disabled={!timelineReady} onClick={() => seek(1)}>
          ›
        </button>
        <button className="icon-button" aria-label="下一集" disabled={!canGoNext} onClick={onNextEpisode}>
          ››
        </button>
      </div>
    </div>
  );
}
