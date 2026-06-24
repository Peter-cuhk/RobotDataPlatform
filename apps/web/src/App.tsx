import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import {
  createRecording,
  exportEpisode,
  importDataset,
  listEpisodes,
  type Episode,
  type Project,
} from "./api";
import RerunViewer from "./RerunViewer";
import "./styles.css";

const defaultPath = "data/samples/lerobot-pusht";

function episodeLabel(index: number) {
  return `Episode ${index.toString().padStart(6, "0")}`;
}

export default function App() {
  const [path, setPath] = useState(defaultPath);
  const [project, setProject] = useState<Project | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [exportedPath, setExportedPath] = useState<string | null>(null);

  const importMutation = useMutation({
    mutationFn: () => importDataset(path),
    onSuccess: (value) => {
      setProject(value);
      setSelected(0);
      setRecordingUrl(null);
      setExportedPath(null);
    },
  });
  const episodesQuery = useQuery({
    queryKey: ["episodes", project?.id],
    queryFn: () => listEpisodes(project!.id),
    enabled: Boolean(project),
  });
  const recordingMutation = useMutation({
    mutationFn: (episodeIndex: number) => createRecording(project!.id, episodeIndex),
    onSuccess: ({ recording_url }) => setRecordingUrl(recording_url),
  });
  const exportMutation = useMutation({
    mutationFn: (episodeIndex: number) => exportEpisode(project!.id, episodeIndex),
    onSuccess: ({ output_path }) => setExportedPath(output_path),
  });

  const selectedEpisode = useMemo(
    () => episodesQuery.data?.find((episode) => episode.episode_index === selected) ?? null,
    [episodesQuery.data, selected],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">LOCAL-FIRST ROBOT DATA</span>
          <h1>Robot Data Studio</h1>
        </div>
        <span className="status-dot">● local</span>
      </header>

      <section className="import-bar">
        <label>
          <span>Dataset path</span>
          <input
            aria-label="Dataset path"
            value={path}
            onChange={(event) => setPath(event.target.value)}
          />
        </label>
        <button onClick={() => importMutation.mutate()} disabled={importMutation.isPending}>
          {importMutation.isPending ? "Scanning…" : "Import dataset"}
        </button>
        {importMutation.error && <p className="error">{importMutation.error.message}</p>}
      </section>

      {!project ? (
        <section className="empty-state">
          <div className="empty-icon">↳</div>
          <h2>Open a LeRobot v3 dataset</h2>
          <p>Metadata, episodes, synchronized signals and replay stay on this machine.</p>
        </section>
      ) : (
        <>
          <section className="dataset-strip">
            <div><span>Format</span><strong>{project.dataset.version}</strong></div>
            <div><span>Dataset</span><strong>{project.dataset.total_episodes} episodes</strong></div>
            <div><span>Frames</span><strong>{project.dataset.total_frames.toLocaleString()}</strong></div>
            <div><span>Rate</span><strong>{project.dataset.fps} Hz</strong></div>
            <div><span>Streams</span><strong>{project.dataset.video_keys.length} video</strong></div>
          </section>

          <section className="workspace">
            <aside className="episode-panel">
              <div className="panel-heading">
                <span>Episodes</span>
                <small>{episodesQuery.data?.length ?? 0} indexed</small>
              </div>
              <div className="episode-list">
                {episodesQuery.data?.map((episode) => (
                  <EpisodeRow
                    key={episode.episode_index}
                    episode={episode}
                    active={episode.episode_index === selected}
                    onSelect={() => {
                      setSelected(episode.episode_index);
                      setRecordingUrl(null);
                      setExportedPath(null);
                    }}
                  />
                ))}
              </div>
            </aside>

            <section className="viewer-panel">
              <div className="viewer-toolbar">
                <div>
                  <span className="eyebrow">INSPECT</span>
                  <h2>{selectedEpisode ? episodeLabel(selectedEpisode.episode_index) : "Episode"}</h2>
                </div>
                <div className="actions">
                  <button
                    className="secondary"
                    disabled={selected === null || recordingMutation.isPending}
                    onClick={() => selected !== null && recordingMutation.mutate(selected)}
                  >
                    {recordingMutation.isPending ? "Building replay…" : "Replay in Rerun"}
                  </button>
                  <button
                    disabled={selected === null || exportMutation.isPending}
                    onClick={() => selected !== null && exportMutation.mutate(selected)}
                  >
                    {exportMutation.isPending ? "Exporting…" : "Export ACT HDF5"}
                  </button>
                </div>
              </div>

              <div className="viewer-stage">
                {recordingUrl ? (
                  <RerunViewer recordingUrl={recordingUrl} />
                ) : (
                  <div className="viewer-placeholder">
                    <div className="signal-preview">
                      <i /><i /><i /><i /><i /><i /><i /><i />
                    </div>
                    <h3>Rerun replay is ready to build</h3>
                    <p>Select an episode and click “Replay in Rerun”.</p>
                  </div>
                )}
              </div>

              {selectedEpisode && (
                <footer className="episode-detail">
                  <div><span>Length</span><strong>{selectedEpisode.length} frames</strong></div>
                  <div><span>Duration</span><strong>{selectedEpisode.duration_seconds.toFixed(1)} s</strong></div>
                  <div className="task"><span>Task</span><strong>{selectedEpisode.tasks[0]}</strong></div>
                </footer>
              )}
              {exportedPath && <div className="success">Exported to {exportedPath}</div>}
            </section>
          </section>
        </>
      )}
    </main>
  );
}

function EpisodeRow({
  episode,
  active,
  onSelect,
}: {
  episode: Episode;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`episode-row ${active ? "active" : ""}`} onClick={onSelect}>
      <span>
        <strong>{episodeLabel(episode.episode_index)}</strong>
        <small>{episode.length} frames · {episode.duration_seconds.toFixed(1)} s</small>
      </span>
      <b>›</b>
    </button>
  );
}
