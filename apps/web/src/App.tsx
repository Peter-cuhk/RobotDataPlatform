import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import {
  createRecording,
  exportDataset,
  importDataset,
  listFormats,
  listEpisodes,
  runCleaning,
  saveVlmSettings,
  updateEpisodeDecision,
  type CleaningStatus,
  type CleaningSummary,
  type Episode,
  type EpisodeQualityResult,
  type ExportResult,
  type FormatInfo,
  type Project,
  type VlmSettings,
} from "./api";
import RerunViewer from "./RerunViewer";
import "./styles.css";

const defaultPath = "data/samples/lerobot-pusht";
const datasetPathPlaceholder = "data/samples/lerobot-pusht";
const defaultVlmPrompt =
  "You are an automated robot episode evaluator. Return only JSON with success, score, and reason. Judge whether the task was successfully completed from the visual evidence.";
const findingFilters = [
  { code: "blur", label: "模糊帧" },
  { code: "time_sync", label: "时间不同步" },
  { code: "action_jump", label: "Action 跳变" },
  { code: "vlm_failed", label: "VLM 失败" },
] as const;

function episodeLabel(index: number) {
  return `Episode ${index.toString().padStart(6, "0")}`;
}

function compactEpisodeLabel(index: number) {
  return `#${index.toString().padStart(6, "0")}`;
}

function scoreLabel(score: number | null) {
  return score === null ? "unscored" : `${Math.round(score * 100)} / 100`;
}

function statusLabel(status: CleaningStatus) {
  return {
    passed: "通过",
    review: "待审查",
    excluded: "排除",
    unscored: "未评分",
  }[status];
}

function pickEpisodeAfterCleaning(summary: CleaningSummary) {
  const review = summary.results.find((result) => result.status === "review");
  if (review) return review.episode_index;
  const scored = summary.results
    .filter((result) => result.score !== null)
    .sort((left, right) => (left.score ?? 1) - (right.score ?? 1));
  return scored[0]?.episode_index ?? summary.results[0]?.episode_index ?? null;
}

export default function App() {
  const [path, setPath] = useState(defaultPath);
  const [importFormat, setImportFormat] = useState("auto");
  const [exportFormat, setExportFormat] = useState("act_hdf5");
  const [exportScope, setExportScope] = useState<"selected" | "all">("selected");
  const [project, setProject] = useState<Project | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [exportedResult, setExportedResult] = useState<ExportResult | null>(null);
  const [cleaningSummary, setCleaningSummary] = useState<CleaningSummary | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFindingFilters, setActiveFindingFilters] = useState<string[]>([]);
  const [showVlmSettings, setShowVlmSettings] = useState(false);
  const [vlmSettings, setVlmSettings] = useState<VlmSettings>({
    enabled: false,
    provider: "OpenAI",
    model: "gpt-4o-mini",
    api_base_url: null,
    prompt: defaultVlmPrompt,
    sample_frames: 4,
  });

  const formatsQuery = useQuery({
    queryKey: ["formats"],
    queryFn: listFormats,
  });
  const importMutation = useMutation({
    mutationFn: () => importDataset(path, importFormat),
    onSuccess: (value) => {
      setProject(value);
      setSelected(0);
      setRecordingUrl(null);
      setExportedResult(null);
      setCleaningSummary(null);
    },
  });
  const episodesQuery = useQuery({
    queryKey: ["episodes", project?.id],
    queryFn: () => listEpisodes(project!.id),
    enabled: Boolean(project),
  });
  const cleaningMutation = useMutation({
    mutationFn: () => runCleaning(project!.id, vlmSettings),
    onSuccess: ({ summary }) => {
      setCleaningSummary(summary);
      setSelected(pickEpisodeAfterCleaning(summary));
      setRecordingUrl(null);
      setExportedResult(null);
    },
  });
  const vlmSettingsMutation = useMutation({
    mutationFn: (settings: VlmSettings) => saveVlmSettings(project!.id, settings),
  });
  const decisionMutation = useMutation({
    mutationFn: ({ episodeIndex, status }: { episodeIndex: number; status: "passed" | "review" | "excluded" }) =>
      updateEpisodeDecision(project!.id, episodeIndex, status),
    onSuccess: (updated) => {
      setCleaningSummary((summary) => {
        if (!summary) return summary;
        const results = summary.results.map((result) =>
          result.episode_index === updated.episode_index ? updated : result,
        );
        return summarizeCleaning({ ...summary, results });
      });
    },
  });
  const recordingMutation = useMutation({
    mutationFn: (episodeIndex: number) => createRecording(project!.id, episodeIndex),
    onSuccess: ({ recording_url }) => setRecordingUrl(recording_url),
  });
  const exportMutation = useMutation({
    mutationFn: () => {
      const episodeIndexes =
        exportScope === "all"
          ? (episodesQuery.data ?? []).map((episode) => episode.episode_index)
          : selected !== null
            ? [selected]
            : [];
      return exportDataset(project!.id, episodeIndexes, exportFormat);
    },
    onSuccess: (result) => setExportedResult(result),
  });

  const qualityByEpisode = useMemo(
    () => new Map(cleaningSummary?.results.map((result) => [result.episode_index, result]) ?? []),
    [cleaningSummary],
  );
  const selectedEpisode = useMemo(
    () => episodesQuery.data?.find((episode) => episode.episode_index === selected) ?? null,
    [episodesQuery.data, selected],
  );
  const selectedEpisodePosition = useMemo(
    () => (selected === null ? -1 : (episodesQuery.data ?? []).findIndex((episode) => episode.episode_index === selected)),
    [episodesQuery.data, selected],
  );
  const selectedQuality = selected === null ? null : qualityByEpisode.get(selected) ?? null;
  const formats = formatsQuery.data ?? [];
  const importFormats = formats.filter((format) => format.can_import);
  const exportFormats = formats.filter((format) => format.can_export);
  const updateVlmSettings = (nextSettings: VlmSettings) => {
    setVlmSettings(nextSettings);
    if (project) {
      vlmSettingsMutation.mutate(nextSettings);
    }
  };
  const replayEpisode = (episodeIndex: number) => {
    setSelected(episodeIndex);
    setRecordingUrl(null);
    setExportedResult(null);
    recordingMutation.mutate(episodeIndex);
  };
  const replayAdjacentEpisode = (offset: -1 | 1) => {
    const episodes = episodesQuery.data ?? [];
    if (selectedEpisodePosition < 0) return;
    const nextEpisode = episodes[selectedEpisodePosition + offset];
    if (!nextEpisode) return;
    replayEpisode(nextEpisode.episode_index);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">LOCAL-FIRST ROBOT DATA</span>
          <h1>Robot Data Studio</h1>
        </div>
        <div className="topbar-actions">
          <button className="secondary" onClick={() => setShowVlmSettings((value) => !value)}>
            VLM 设置
          </button>
          <span className="status-dot">● local</span>
          {showVlmSettings && (
            <VlmSettingsPanel
              settings={vlmSettings}
              onChange={updateVlmSettings}
            />
          )}
        </div>
      </header>

      <section className="import-bar">
        <label>
          <span>Dataset path</span>
          <input
            aria-label="Dataset path"
            placeholder={datasetPathPlaceholder}
            value={path}
            onChange={(event) => setPath(event.target.value)}
          />
        </label>
        <label>
          <span>Import format</span>
          <select
            aria-label="Import format"
            value={importFormat}
            onChange={(event) => setImportFormat(event.target.value)}
          >
            <option value="auto">Auto detect</option>
            {importFormats.map((format) => (
              <option key={format.id} value={format.id}>
                {format.label}
              </option>
            ))}
          </select>
        </label>
        <button onClick={() => importMutation.mutate()} disabled={importMutation.isPending}>
          {importMutation.isPending ? "Scanning…" : "Import dataset"}
        </button>
        {importMutation.error && <p className="error">{importMutation.error.message}</p>}
      </section>

      {!project ? (
        <section className="empty-state">
          <div className="empty-icon">↳</div>
          <h2>Open a robot dataset</h2>
          <p>LeRobot, HDF5 and UMI/Zarr data stay on this machine.</p>
        </section>
      ) : (
        <>
          <section className="dataset-strip">
            <div><span>Format</span><strong>{project.dataset.format} · {project.dataset.version}</strong></div>
            <div><span>Dataset</span><strong>{project.dataset.total_episodes} episodes</strong></div>
            <div><span>Frames</span><strong>{project.dataset.total_frames.toLocaleString()}</strong></div>
            <div><span>Rate</span><strong>{project.dataset.fps} Hz</strong></div>
            <div><span>Cleaning</span><strong>{cleaningSummary ? `${cleaningSummary.review_count} review` : "not run"}</strong></div>
          </section>

          <section className="workspace">
            <aside className="episode-panel">
              <div className="panel-heading">
                <span>Episodes</span>
                <small>{episodesQuery.data?.length ?? 0} indexed</small>
              </div>
              <div className="episode-list">
                <EpisodeNavigation
                  episodes={episodesQuery.data ?? []}
                  qualityByEpisode={qualityByEpisode}
                  selected={selected}
                  searchQuery={searchQuery}
                  activeFindingFilters={activeFindingFilters}
                  onSearchChange={setSearchQuery}
                  onToggleFindingFilter={(code) =>
                    setActiveFindingFilters((filters) =>
                      filters.includes(code)
                        ? filters.filter((filter) => filter !== code)
                        : [...filters, code],
                    )
                  }
                  onSelect={(episodeIndex) => {
                    setSelected(episodeIndex);
                    setRecordingUrl(null);
                    setExportedResult(null);
                  }}
                />
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
                    disabled={cleaningMutation.isPending}
                    onClick={() => cleaningMutation.mutate()}
                  >
                    {cleaningMutation.isPending ? "清洗中…" : "运行清洗 Pipeline"}
                  </button>
                  <button
                    className="secondary"
                    disabled={selected === null || recordingMutation.isPending}
                    onClick={() => selected !== null && replayEpisode(selected)}
                  >
                    {recordingMutation.isPending ? "Building replay…" : "Replay in Rerun"}
                  </button>
                  <button
                    disabled={
                      exportMutation.isPending ||
                      (exportScope === "selected" && selected === null) ||
                      (exportScope === "all" && !episodesQuery.data?.length)
                    }
                    onClick={() => exportMutation.mutate()}
                  >
                    {exportMutation.isPending ? "Exporting…" : exportScope === "all" ? "Export all" : "Export selected"}
                  </button>
                </div>
              </div>

              <ExportPanel
                formats={exportFormats}
                selectedFormat={exportFormat}
                scope={exportScope}
                onFormatChange={setExportFormat}
                onScopeChange={setExportScope}
              />

              <div className="viewer-stage">
                {recordingUrl ? (
                  <RerunViewer
                    recordingUrl={recordingUrl}
                    canGoPrevious={selectedEpisodePosition > 0 && !recordingMutation.isPending}
                    canGoNext={
                      selectedEpisodePosition >= 0 &&
                      selectedEpisodePosition < (episodesQuery.data?.length ?? 0) - 1 &&
                      !recordingMutation.isPending
                    }
                    onPreviousEpisode={() => replayAdjacentEpisode(-1)}
                    onNextEpisode={() => replayAdjacentEpisode(1)}
                  />
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
              {exportedResult && (
                <div className="success">
                  Exported {exportedResult.episode_count} episode(s) to {exportedResult.output_path}
                  <br />
                  Report: {exportedResult.report_path}
                </div>
              )}
            </section>

            <QualityReport
              quality={selectedQuality}
              pending={decisionMutation.isPending}
              onDecision={(status) => {
                if (selected !== null) decisionMutation.mutate({ episodeIndex: selected, status });
              }}
              onAddToPipeline={() => cleaningMutation.mutate()}
            />
          </section>
        </>
      )}
    </main>
  );
}

function ExportPanel({
  formats,
  selectedFormat,
  scope,
  onFormatChange,
  onScopeChange,
}: {
  formats: FormatInfo[];
  selectedFormat: string;
  scope: "selected" | "all";
  onFormatChange: (value: string) => void;
  onScopeChange: (value: "selected" | "all") => void;
}) {
  return (
    <div className="export-panel">
      <label>
        <span>Export format</span>
        <select
          aria-label="Export format"
          value={selectedFormat}
          onChange={(event) => onFormatChange(event.target.value)}
        >
          {formats.map((format) => (
            <option key={format.id} value={format.id}>
              {format.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>Export scope</span>
        <select
          aria-label="Export scope"
          value={scope}
          onChange={(event) => onScopeChange(event.target.value as "selected" | "all")}
        >
          <option value="selected">Selected episode</option>
          <option value="all">All indexed episodes</option>
        </select>
      </label>
    </div>
  );
}

function summarizeCleaning(summary: CleaningSummary): CleaningSummary {
  return {
    ...summary,
    passed_count: summary.results.filter((result) => result.status === "passed").length,
    review_count: summary.results.filter((result) => result.status === "review").length,
    excluded_count: summary.results.filter((result) => result.status === "excluded").length,
    unscored_count: summary.results.filter((result) => result.status === "unscored").length,
  };
}

function EpisodeNavigation({
  episodes,
  qualityByEpisode,
  selected,
  searchQuery,
  activeFindingFilters,
  onSearchChange,
  onToggleFindingFilter,
  onSelect,
}: {
  episodes: Episode[];
  qualityByEpisode: Map<number, EpisodeQualityResult>;
  selected: number | null;
  searchQuery: string;
  activeFindingFilters: string[];
  onSearchChange: (value: string) => void;
  onToggleFindingFilter: (code: string) => void;
  onSelect: (episodeIndex: number) => void;
}) {
  const visibleEpisodes = episodes.filter((episode) =>
    episodeMatchesFilters(episode, qualityByEpisode.get(episode.episode_index), searchQuery, activeFindingFilters),
  );
  if (qualityByEpisode.size === 0) {
    return (
      <>
        <SidebarFilters
          searchQuery={searchQuery}
          activeFindingFilters={activeFindingFilters}
          onSearchChange={onSearchChange}
          onToggleFindingFilter={onToggleFindingFilter}
        />
        {visibleEpisodes.map((episode) => (
          <EpisodeRow
            key={episode.episode_index}
            episode={episode}
            quality={null}
            active={episode.episode_index === selected}
            onSelect={() => onSelect(episode.episode_index)}
          />
        ))}
      </>
    );
  }
  const groups: Array<{ status: CleaningStatus; label: string; episodes: Episode[] }> = [
    { status: "review", label: "待审查", episodes: [] },
    { status: "excluded", label: "排除", episodes: [] },
    { status: "passed", label: "通过", episodes: [] },
  ];
  for (const episode of episodes) {
    if (!visibleEpisodes.includes(episode)) continue;
    const quality = qualityByEpisode.get(episode.episode_index);
    const group = groups.find((item) => item.status === quality?.status);
    if (group) group.episodes.push(episode);
  }
  return (
    <>
      <SidebarFilters
        searchQuery={searchQuery}
        activeFindingFilters={activeFindingFilters}
        onSearchChange={onSearchChange}
        onToggleFindingFilter={onToggleFindingFilter}
      />
      {groups.map((group) => (
        <section className="episode-folder" key={group.status}>
          <div className="folder-heading">
            <span>{group.label}</span>
            <small>{group.episodes.length}</small>
          </div>
          {group.episodes.map((episode) => (
            <EpisodeRow
              key={episode.episode_index}
              episode={episode}
              quality={qualityByEpisode.get(episode.episode_index) ?? null}
              active={episode.episode_index === selected}
              onSelect={() => onSelect(episode.episode_index)}
            />
          ))}
        </section>
      ))}
    </>
  );
}

function episodeMatchesFilters(
  episode: Episode,
  quality: EpisodeQualityResult | undefined,
  query: string,
  activeFilters: string[],
) {
  const normalizedQuery = query.trim().toLowerCase();
  const text = [
    compactEpisodeLabel(episode.episode_index),
    episodeLabel(episode.episode_index),
    episode.tasks.join(" "),
  ].join(" ").toLowerCase();
  if (normalizedQuery && !text.includes(normalizedQuery)) return false;
  if (activeFilters.length === 0) return true;
  return activeFilters.every((filter) => quality?.findings.some((finding) => finding.code === filter));
}

function SidebarFilters({
  searchQuery,
  activeFindingFilters,
  onSearchChange,
  onToggleFindingFilter,
}: {
  searchQuery: string;
  activeFindingFilters: string[];
  onSearchChange: (value: string) => void;
  onToggleFindingFilter: (code: string) => void;
}) {
  return (
    <div className="sidebar-tools">
      <label className="search-box">
        <span>搜索 / 筛选</span>
        <input
          aria-label="搜索 / 筛选"
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="#000184"
        />
      </label>
      <div className="filter-box">
        <span>筛选器</span>
        {findingFilters.map((filter) => (
          <label key={filter.code}>
            <input
              type="checkbox"
              checked={activeFindingFilters.includes(filter.code)}
              onChange={() => onToggleFindingFilter(filter.code)}
            />
            {filter.label}
          </label>
        ))}
      </div>
    </div>
  );
}

function EpisodeRow({
  episode,
  quality,
  active,
  onSelect,
}: {
  episode: Episode;
  quality: EpisodeQualityResult | null;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`episode-row ${active ? "active" : ""}`} onClick={onSelect}>
      <span>
        <strong>{quality ? compactEpisodeLabel(episode.episode_index) : episodeLabel(episode.episode_index)}</strong>
        <small>
          {quality
            ? `${scoreLabel(quality.score)} · ${statusLabel(quality.status)} · ${quality.findings.length} issues`
            : `${episode.length} frames · ${episode.duration_seconds.toFixed(1)} s`}
        </small>
      </span>
      <b>›</b>
    </button>
  );
}

function QualityReport({
  quality,
  pending,
  onDecision,
  onAddToPipeline,
}: {
  quality: EpisodeQualityResult | null;
  pending: boolean;
  onDecision: (status: "passed" | "excluded") => void;
  onAddToPipeline: () => void;
}) {
  return (
    <aside className="quality-panel">
      <div className="panel-heading">
        <span>Quality Report</span>
      </div>
      {!quality ? (
        <div className="quality-empty">
          <strong>Not scored</strong>
          <p>Run the cleaning pipeline to classify episodes.</p>
        </div>
      ) : (
        <div className="quality-body">
          <div className="quality-score">
            <span>{scoreLabel(quality.score)}</span>
            <small>{statusLabel(quality.status)} · {quality.source}</small>
          </div>
          <div className="score-list">
            {Object.entries(quality.per_attribute_scores).map(([name, value]) => (
              <div key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <strong>{Math.round(value * 100)}</strong>
              </div>
            ))}
          </div>
          <div className="finding-list">
            <h3>发现 {quality.findings.length} 个问题</h3>
            {quality.findings.length === 0 ? (
              <p>No quality findings.</p>
            ) : (
              quality.findings.map((finding) => (
                <div className="finding" key={`${finding.code}-${finding.message}`}>
                  {finding.message}
                </div>
              ))
            )}
          </div>
          <div className="review-actions">
            <button disabled={pending} onClick={() => onDecision("passed")}>通过</button>
            <button className="secondary danger" disabled={pending} onClick={() => onDecision("excluded")}>排除</button>
            <button className="secondary span-all" disabled={pending} onClick={onAddToPipeline}>
              加入清洗 Pipeline
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

function VlmSettingsPanel({
  settings,
  onChange,
}: {
  settings: VlmSettings;
  onChange: (settings: VlmSettings) => void;
}) {
  const update = (changes: Partial<VlmSettings>) => onChange({ ...settings, ...changes });
  return (
    <div className="vlm-popover">
      <label className="checkbox-row">
        <input
          type="checkbox"
          aria-label="启用 VLM"
          checked={settings.enabled}
          onChange={(event) => update({ enabled: event.target.checked })}
        />
        启用 VLM
      </label>
      <label>
        <span>Provider</span>
        <select
          aria-label="VLM Provider"
          value={settings.provider}
          onChange={(event) => update({ provider: event.target.value })}
        >
          <option>OpenAI</option>
          <option>Gemini</option>
          <option>Local</option>
        </select>
      </label>
      <label>
        <span>Model</span>
        <input
          aria-label="VLM 模型"
          value={settings.model}
          onChange={(event) => update({ model: event.target.value })}
        />
      </label>
      <label>
        <span>API Base URL</span>
        <input
          aria-label="VLM API Base URL"
          placeholder="https://api.openai.com/v1"
          value={settings.api_base_url ?? ""}
          onChange={(event) => update({ api_base_url: event.target.value || null })}
        />
      </label>
      <label>
        <span>API Key</span>
        <input
          aria-label="VLM API Key"
          type="password"
          placeholder="Use env var if blank"
          value={settings.api_key ?? ""}
          onChange={(event) => update({ api_key: event.target.value || null })}
        />
      </label>
      <label>
        <span>Sample Frames</span>
        <input
          aria-label="VLM Sample Frames"
          type="number"
          min={1}
          max={12}
          value={settings.sample_frames}
          onChange={(event) => update({ sample_frames: Number(event.target.value) })}
        />
      </label>
      <label>
        <span>Prompt</span>
        <textarea
          aria-label="VLM Prompt"
          value={settings.prompt}
          onChange={(event) => update({ prompt: event.target.value })}
        />
      </label>
    </div>
  );
}
