import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";

import {
  createRecording,
  exportDataset,
  getFilterDetail,
  getVlmSettings,
  importDataset,
  listFormats,
  listEpisodes,
  runPipelineStream,
  saveFilterConfig,
  saveVlmSettings,
  updateEpisodeDecision,
  uploadKinematicsUrdf,
  type CleaningRuleConfig,
  type CleaningStatus,
  type CleaningSummary,
  type Episode,
  type EpisodeQualityResult,
  type ExportResult,
  type FilterConfig,
  type FilterDetail,
  type FilterStageId,
  type FilterSummary,
  type FormatInfo,
  type Project,
  type VisualQualityIncident,
  type VisualQualityMetricSample,
  type VlmSettings,
} from "./api";
import { languageStorageKey, readStoredLanguage, translations, type Language, type Translation } from "./i18n";
import RerunViewer from "./RerunViewer";
import "./styles.css";

const defaultPath = "";
const datasetPathPlaceholder = "Enter a local dataset path";
const defaultVlmPrompt =
  "You are an automated robot episode evaluator. Return only JSON with success, score, and reason. Judge whether the task was successfully completed from the visual evidence.";

type ExportScope =
  | "selected"
  | "all"
  | "filtered"
  | "checked"
  | "status_passed"
  | "status_review"
  | "status_excluded";

const defaultQualityPanelWidth = 300;
const minQualityPanelWidth = 220;
const maxQualityPanelWidth = 520;

const filterStageIds: FilterStageId[] = [
  "visual_quality",
  "sudden_change",
  "state_action_alignment",
  "extreme_value",
  "kinematic_consistency",
  "orientation_alignment",
  "metadata_completeness",
];

const defaultEnabledFilterStages: FilterStageId[] = [
  "visual_quality",
  "sudden_change",
  "state_action_alignment",
  "extreme_value",
  "metadata_completeness",
];

const defaultSidebarFilterCodes = [
  "visual_quality",
  "sudden_change",
  "state_action_alignment",
  "extreme_value",
  "metadata_completeness",
];

function hasOnlyDefaultSidebarFilters(filters: string[]) {
  return (
    filters.length === defaultSidebarFilterCodes.length &&
    defaultSidebarFilterCodes.every((filter) => filters.includes(filter))
  );
}

const defaultQualityWeights: Record<string, number> = {
  visual_quality: 1.5,
  sudden_change: 1.5,
  state_action_alignment: 1.5,
  extreme_value: 2,
  kinematic_consistency: 2,
  orientation_alignment: 1,
  metadata_completeness: 1,
  task_success: 2,
};

function isFilterStageId(value: string): value is FilterStageId {
  return filterStageIds.includes(value as FilterStageId);
}

function clampPanelWidth(width: number) {
  return Math.min(maxQualityPanelWidth, Math.max(minQualityPanelWidth, width));
}

function findingFilters(_copy: Translation) {
  return [] as Array<{ code: string; label: string }>;
}

type QualityRuleId = FilterStageId | "task_success";

function dataFilters(copy: Translation): Array<{ id: QualityRuleId; stageId?: FilterStageId; label: string; code: string }> {
  return [
    { id: "visual_quality", stageId: "visual_quality", label: copy.filters.visualQuality, code: "visual_quality" },
    { id: "sudden_change", stageId: "sudden_change", label: copy.filters.suddenChange, code: "sudden_change" },
    { id: "state_action_alignment", stageId: "state_action_alignment", label: copy.filters.stateActionAlignment, code: "state_action_alignment" },
    { id: "extreme_value", stageId: "extreme_value", label: copy.filters.extremeValue, code: "extreme_value" },
    { id: "metadata_completeness", stageId: "metadata_completeness", label: copy.filters.metadataCompleteness, code: "metadata_completeness" },
    { id: "kinematic_consistency", stageId: "kinematic_consistency", label: copy.filters.kinematicConsistency, code: "kinematic_consistency" },
    { id: "task_success", label: copy.filters.taskVlmValidity, code: "vlm_failed" },
    { id: "orientation_alignment", stageId: "orientation_alignment", label: copy.filters.orientationAlignment, code: "orientation_alignment" },
  ];
}

function isKinematicsConfigured(config: FilterConfig["kinematics"] | null | undefined) {
  return Boolean(
    config?.urdf_path &&
    config.end_effector_link &&
    config.joint_names.length > 0 &&
    config.eef_position_indices.length > 0,
  );
}

function episodeLabel(index: number) {
  return `Episode ${index.toString().padStart(6, "0")}`;
}

function compactEpisodeLabel(index: number) {
  return `#${index.toString().padStart(6, "0")}`;
}

function scoreLabel(score: number | null, copy: Translation) {
  return score === null ? copy.status.unscored.toLowerCase() : `${Math.round(score * 100)} / 100`;
}

function subtaskTimeRange(startSeconds: number, endSeconds: number) {
  return `${startSeconds.toFixed(1)}-${endSeconds.toFixed(1)}s`;
}

function statusLabel(status: CleaningStatus, copy: Translation) {
  return {
    passed: copy.status.passed,
    review: copy.status.review,
    excluded: copy.status.excluded,
    unscored: copy.status.unscored,
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

function pickNextEpisodeInStatus(
  summary: CleaningSummary,
  currentEpisodeIndex: number,
  status: CleaningStatus,
) {
  const inStatus = summary.results
    .filter((result) => result.status === status)
    .sort((left, right) => left.episode_index - right.episode_index);
  return (
    inStatus.find((result) => result.episode_index > currentEpisodeIndex)?.episode_index ??
    inStatus[0]?.episode_index ??
    null
  );
}

function resolveExportEpisodeIndexes(
  scope: ExportScope,
  episodes: Episode[],
  selected: number | null,
  checkedEpisodeIndexes: Set<number>,
  visibleEpisodes: Episode[],
  cleaningSummary: CleaningSummary | null,
) {
  if (scope === "selected") return selected === null ? [] : [selected];
  if (scope === "all") return episodes.map((episode) => episode.episode_index);
  if (scope === "filtered") return visibleEpisodes.map((episode) => episode.episode_index);
  if (scope === "checked") {
    const validIndexes = new Set(episodes.map((episode) => episode.episode_index));
    return [...checkedEpisodeIndexes]
      .filter((episodeIndex) => validIndexes.has(episodeIndex))
      .sort((left, right) => left - right);
  }
  const status = {
    status_passed: "passed",
    status_review: "review",
    status_excluded: "excluded",
  }[scope];
  if (!cleaningSummary || !status) return [];
  return cleaningSummary.results
    .filter((result) => result.status === status)
    .map((result) => result.episode_index)
    .sort((left, right) => left - right);
}

export default function App() {
  const [language, setLanguageState] = useState<Language>(() => readStoredLanguage());
  const [path, setPath] = useState(defaultPath);
  const [importFormat, setImportFormat] = useState("auto");
  const [exportFormat, setExportFormat] = useState("act_hdf5");
  const [exportOutputDir, setExportOutputDir] = useState("");
  const [exportScope, setExportScope] = useState<ExportScope>("selected");
  const [qualityPanelWidth, setQualityPanelWidth] = useState(defaultQualityPanelWidth);
  const [project, setProject] = useState<Project | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [checkedEpisodeIndexes, setCheckedEpisodeIndexes] = useState<Set<number>>(() => new Set());
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [replayReady, setReplayReady] = useState(false);
  const [exportedResult, setExportedResult] = useState<ExportResult | null>(null);
  const [cleaningSummary, setCleaningSummary] = useState<CleaningSummary | null>(null);
  const [cleaningProgress, setCleaningProgress] = useState(0);
  const [filterSummary, setFilterSummary] = useState<FilterSummary | null>(null);
  const [activeFilterStage, setActiveFilterStage] = useState<FilterStageId | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFindingFilters, setActiveFindingFilters] = useState<string[]>([]);
  const [enabledFilterStages, setEnabledFilterStages] = useState<FilterStageId[]>(() => [...defaultEnabledFilterStages]);
  const [qualityWeights, setQualityWeights] = useState<Record<string, number>>(() => ({ ...defaultQualityWeights }));
  const [kinematicsConfigured, setKinematicsConfigured] = useState(false);
  const [showVlmSettings, setShowVlmSettings] = useState(false);
  const [vlmSettingsDirty, setVlmSettingsDirty] = useState(false);
  const [vlmSettings, setVlmSettings] = useState<VlmSettings>({
    enabled: false,
    provider: "OpenAI",
    model: "gpt-4o-mini",
    api_base_url: null,
    prompt: defaultVlmPrompt,
    sample_frames: 4,
  });
  const copy = translations[language];
  const setLanguage = (nextLanguage: Language) => {
    setLanguageState(nextLanguage);
    if (typeof window.localStorage?.setItem === "function") {
      window.localStorage.setItem(languageStorageKey, nextLanguage);
    }
  };

  const formatsQuery = useQuery({
    queryKey: ["formats"],
    queryFn: listFormats,
  });
  const importMutation = useMutation({
    mutationFn: () => importDataset(path, importFormat),
    onSuccess: (value) => {
      setProject(value);
      setSelected(0);
      setCheckedEpisodeIndexes(new Set());
      setRecordingUrl(null);
      setReplayReady(false);
      setExportedResult(null);
      setCleaningSummary(null);
      setFilterSummary(null);
      setActiveFilterStage(null);
      setActiveFindingFilters([
        ...defaultSidebarFilterCodes,
        ...(vlmSettings.enabled ? ["vlm_failed"] : []),
      ]);
      setEnabledFilterStages([...defaultEnabledFilterStages]);
      setQualityWeights({ ...defaultQualityWeights });
      setKinematicsConfigured(false);
    },
  });
  const episodesQuery = useQuery({
    queryKey: ["episodes", project?.id],
    queryFn: () => listEpisodes(project!.id),
    enabled: Boolean(project),
  });
  const cleaningMutation = useMutation({
    mutationFn: async (episodeIndexes?: number[]) => {
      const activeQualityWeights = Object.fromEntries(
        enabledFilterStages.map((stageId) => [stageId, qualityWeights[stageId] ?? 1]),
      );
      if (vlmSettings.enabled) {
        activeQualityWeights.task_success = qualityWeights.task_success ?? 2;
      }
      const ruleConfig: CleaningRuleConfig = {
        enabled_filter_stages: enabledFilterStages,
        quality_weights: activeQualityWeights,
      };
      return runPipelineStream(project!.id, vlmSettings, episodeIndexes, ruleConfig, (progress) => {
        const ratio = progress.total > 0 ? progress.completed / progress.total : 0;
        const pct = progress.phase === "cleaning" ? 50 * ratio : 50 + 50 * ratio;
        setCleaningProgress(Math.min(100, Math.max(0, Math.round(pct))));
      });
    },
    onMutate: () => setCleaningProgress(0),
    onSuccess: ({ cleaning, filters }) => {
      const nextSelected = pickEpisodeAfterCleaning(cleaning.summary);
      setCleaningSummary(cleaning.summary);
      setFilterSummary(filters.summary);
      setSelected(nextSelected);
      setRecordingUrl(null);
      setReplayReady(nextSelected !== null);
      setExportedResult(null);
      setCleaningProgress(0);
    },
    onError: () => setCleaningProgress(0),
  });
  const vlmSettingsMutation = useMutation({
    mutationFn: (settings: VlmSettings) => saveVlmSettings(project!.id, settings),
    onSuccess: (settings) => setVlmSettings(settings),
  });
  const vlmSettingsQuery = useQuery({
    queryKey: ["vlm-settings", project?.id],
    queryFn: () => getVlmSettings(project!.id),
    enabled: Boolean(project && showVlmSettings && !vlmSettingsDirty),
  });
  const recordingMutation = useMutation({
    mutationFn: (episodeIndex: number) => createRecording(project!.id, episodeIndex),
    onSuccess: ({ recording_url }) => setRecordingUrl(recording_url),
  });
  const urdfUploadMutation = useMutation({
    mutationFn: (file: File) => uploadKinematicsUrdf(project!.id, file),
    onSuccess: () => filterDetailQuery.refetch(),
  });
  const filterConfigMutation = useMutation({
    mutationFn: (config: Partial<FilterConfig>) => saveFilterConfig(project!.id, config),
    onSuccess: (config) => {
      const configured = isKinematicsConfigured(config.kinematics);
      setKinematicsConfigured(configured);
      setEnabledFilterStages((current) => {
        const withoutKinematics = current.filter((stageId) => stageId !== "kinematic_consistency");
        return configured ? [...withoutKinematics, "kinematic_consistency"] : withoutKinematics;
      });
      filterDetailQuery.refetch();
    },
  });
  const exportMutation = useMutation({
    mutationFn: () => {
      if (exportEpisodeIndexes.length === 0) {
        throw new Error("Select at least one episode to export");
      }
      return exportDataset(project!.id, exportEpisodeIndexes, exportFormat, exportOutputDir);
    },
    onSuccess: (result) => setExportedResult(result),
  });

  const workspaceStyle = {
    "--quality-panel-width": `${qualityPanelWidth}px`,
  } as CSSProperties & Record<"--quality-panel-width", string>;
  const resizeQualityPanel = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const startX = event.clientX;
    const startWidth = qualityPanelWidth;
    const onPointerMove = (moveEvent: PointerEvent) => {
      if (!Number.isFinite(moveEvent.clientX)) return;
      setQualityPanelWidth(clampPanelWidth(startWidth + startX - moveEvent.clientX));
    };
    const onPointerUp = () => {
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    };
    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp);
  };

  const qualityByEpisode = useMemo(
    () => new Map(cleaningSummary?.results.map((result) => [result.episode_index, result]) ?? []),
    [cleaningSummary],
  );
  const selectedEpisode = useMemo(
    () => episodesQuery.data?.find((episode) => episode.episode_index === selected) ?? null,
    [episodesQuery.data, selected],
  );
  const selectedQuality = selected === null ? null : qualityByEpisode.get(selected) ?? null;
  const filterDetailQuery = useQuery({
    queryKey: ["filter-detail", project?.id, activeFilterStage, selected],
    queryFn: () => getFilterDetail(project!.id, activeFilterStage!, selected!),
    enabled: Boolean(project && activeFilterStage && selected !== null),
  });
  const selectedFilterDetail = filterDetailQuery.data ?? null;
  const formats = formatsQuery.data ?? [];
  const importFormats = formats.filter((format) => format.can_import);
  const exportFormats = formats.filter((format) => format.can_export);
  const episodes = episodesQuery.data ?? [];
  const visibleEpisodes = useMemo(
    () =>
      episodes.filter((episode) =>
        episodeMatchesFilters(
          episode,
          qualityByEpisode.get(episode.episode_index),
          searchQuery,
          activeFindingFilters,
          filterSummary,
        ),
      ),
    [activeFindingFilters, episodes, filterSummary, qualityByEpisode, searchQuery],
  );
  const exportEpisodeIndexes = useMemo(
    () =>
      resolveExportEpisodeIndexes(
        exportScope,
        episodes,
        selected,
        checkedEpisodeIndexes,
        visibleEpisodes,
        cleaningSummary,
      ),
    [checkedEpisodeIndexes, cleaningSummary, episodes, exportScope, selected, visibleEpisodes],
  );
  useEffect(() => {
    if (vlmSettingsQuery.data) {
      setVlmSettings(vlmSettingsQuery.data);
    }
  }, [vlmSettingsQuery.data]);
  const updateRuleWeight = (key: string, value: number) => {
    setQualityWeights((current) => ({ ...current, [key]: value }));
  };
  const toggleCheckedEpisode = (episodeIndex: number) => {
    setCheckedEpisodeIndexes((current) => {
      const next = new Set(current);
      if (next.has(episodeIndex)) {
        next.delete(episodeIndex);
      } else {
        next.add(episodeIndex);
      }
      return next;
    });
    setExportedResult(null);
  };
  const toggleCheckedEpisodes = (episodeIndexes: number[]) => {
    if (episodeIndexes.length === 0) return;
    setCheckedEpisodeIndexes((current) => {
      const next = new Set(current);
      const allSelected = episodeIndexes.every((episodeIndex) => next.has(episodeIndex));
      for (const episodeIndex of episodeIndexes) {
        if (allSelected) {
          next.delete(episodeIndex);
        } else {
          next.add(episodeIndex);
        }
      }
      return next;
    });
    setExportedResult(null);
  };
  const updateVlmSettings = (nextSettings: VlmSettings) => {
    setVlmSettingsDirty(true);
    setVlmSettings(nextSettings);
    setActiveFindingFilters((current) => {
      const withoutVlm = current.filter((filter) => filter !== "vlm_failed");
      return nextSettings.enabled ? [...withoutVlm, "vlm_failed"] : withoutVlm;
    });
    if (project) {
      vlmSettingsMutation.mutate(nextSettings);
    }
  };
  const toggleFilterStage = (stageId: FilterStageId) => {
    setEnabledFilterStages((current) =>
      current.includes(stageId)
        ? current.filter((item) => item !== stageId)
        : [...current, stageId],
    );
    setActiveFindingFilters((current) =>
      current.includes(stageId)
        ? current.filter((item) => item !== stageId)
        : [...current, stageId],
    );
  };
  const selectEpisode = (episodeIndex: number) => {
    setSelected(episodeIndex);
    setActiveFilterStage(null);
    setRecordingUrl(null);
    setExportedResult(null);
    setReplayReady(true);
  };
  const buildReplay = (episodeIndex: number) => {
    setRecordingUrl(null);
    setReplayReady(true);
    recordingMutation.mutate(episodeIndex);
  };
  const replayEpisode = (episodeIndex: number) => {
    selectEpisode(episodeIndex);
    buildReplay(episodeIndex);
  };
  const decisionMutation = useMutation({
    mutationFn: ({ episodeIndex, status }: { episodeIndex: number; status: "passed" | "review" | "excluded" }) =>
      updateEpisodeDecision(project!.id, episodeIndex, status),
    onSuccess: (updated) => {
      const originalStatus =
        cleaningSummary?.results.find((result) => result.episode_index === updated.episode_index)?.status ?? null;
      const updatedSummary = cleaningSummary
        ? summarizeCleaning({
            ...cleaningSummary,
            results: cleaningSummary.results.map((result) =>
              result.episode_index === updated.episode_index ? updated : result,
            ),
          })
        : null;
      setCleaningSummary(updatedSummary);
      if (!updatedSummary || !originalStatus) return;
      const nextEpisode = pickNextEpisodeInStatus(updatedSummary, updated.episode_index, originalStatus);
      if (nextEpisode === null) return;
      replayEpisode(nextEpisode);
    },
  });

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">{copy.topbar.eyebrow}</span>
          <h1>Robot Data Studio</h1>
        </div>
        <div className="topbar-actions">
          <div className="language-toggle" aria-label="Language">
            <button
              type="button"
              className={language === "en" ? "active" : ""}
              onClick={() => setLanguage("en")}
            >
              {copy.language.english}
            </button>
            <button
              type="button"
              className={language === "zh" ? "active" : ""}
              onClick={() => setLanguage("zh")}
            >
              {copy.language.chinese}
            </button>
          </div>
          <button className="secondary" onClick={() => setShowVlmSettings((value) => !value)}>
            {copy.topbar.vlmSettings}
          </button>
          <span className="status-dot">{copy.topbar.local}</span>
          {showVlmSettings && (
            <VlmSettingsPanel
              copy={copy}
              settings={vlmSettings}
              onChange={updateVlmSettings}
            />
          )}
        </div>
      </header>

      <section className="import-bar">
        <label>
          <span>{copy.import.datasetPath}</span>
          <input
            aria-label={copy.import.datasetPath}
            placeholder={datasetPathPlaceholder}
            value={path}
            onChange={(event) => setPath(event.target.value)}
          />
        </label>
        <label>
          <span>{copy.import.importFormat}</span>
          <select
            aria-label={copy.import.importFormat}
            value={importFormat}
            onChange={(event) => setImportFormat(event.target.value)}
          >
            <option value="auto">{copy.import.autoDetect}</option>
            {importFormats.map((format) => (
              <option key={format.id} value={format.id}>
                {format.label}
              </option>
            ))}
          </select>
        </label>
        <button onClick={() => importMutation.mutate()} disabled={importMutation.isPending || !path.trim()}>
          {importMutation.isPending ? copy.import.scanning : copy.import.importDataset}
        </button>
        {importMutation.error && <p className="error">{importMutation.error.message}</p>}
      </section>

      {!project ? (
        <section className="empty-state">
          <div className="empty-icon">↳</div>
          <h2>{copy.import.emptyTitle}</h2>
          <p>{copy.import.emptyBody}</p>
        </section>
      ) : (
        <>
          <section className="dataset-strip">
            <div><span>{copy.dataset.format}</span><strong>{project.dataset.format} · {project.dataset.version}</strong></div>
            <div><span>{copy.dataset.dataset}</span><strong>{project.dataset.total_episodes} {copy.dataset.episodes}</strong></div>
            <div><span>{copy.dataset.frames}</span><strong>{project.dataset.total_frames.toLocaleString()}</strong></div>
            <div><span>{copy.dataset.rate}</span><strong>{project.dataset.fps} Hz</strong></div>
            <div><span>{copy.dataset.cleaning}</span><strong>{cleaningSummary ? `${cleaningSummary.review_count} ${copy.dataset.review}` : copy.dataset.notRun}</strong></div>
          </section>

          <section className="workspace" style={workspaceStyle}>
            <aside className="episode-panel">
              <div className="panel-heading">
                <span>Episodes</span>
                <small>{episodesQuery.data?.length ?? 0} {copy.dataset.indexed}</small>
              </div>
              <div className="episode-list">
                <EpisodeNavigation
                  copy={copy}
                  episodes={episodes}
                  visibleEpisodes={visibleEpisodes}
                  qualityByEpisode={qualityByEpisode}
                  selected={selected}
                  checkedEpisodeIndexes={checkedEpisodeIndexes}
                  searchQuery={searchQuery}
                  activeFindingFilters={activeFindingFilters}
                  filterSummary={filterSummary}
                  activeFilterStage={activeFilterStage}
                  enabledFilterStages={enabledFilterStages}
                  taskSuccessEnabled={vlmSettings.enabled}
                  kinematicsConfigured={kinematicsConfigured}
                  weights={qualityWeights}
                  onSearchChange={setSearchQuery}
                  onToggleFindingFilter={(code) =>
                    setActiveFindingFilters((filters) =>
                      filters.includes(code)
                        ? filters.filter((filter) => filter !== code)
                        : [...filters, code],
                    )
                  }
                  onToggleFilterStage={toggleFilterStage}
                  onOpenFilter={(stageId) => {
                    setActiveFilterStage(stageId);
                    setRecordingUrl(null);
                    setExportedResult(null);
                  }}
                  onWeightChange={updateRuleWeight}
                  onToggleChecked={toggleCheckedEpisode}
                  onToggleGroupChecked={toggleCheckedEpisodes}
                  onSelect={(episodeIndex) => selectEpisode(episodeIndex)}
                />
              </div>
            </aside>

            <section className="viewer-panel">
              <div className="viewer-toolbar">
                <div>
                  <span className="eyebrow">{copy.viewer.inspect}</span>
                  <h2>{selectedEpisode ? episodeLabel(selectedEpisode.episode_index) : copy.viewer.fallbackEpisode}</h2>
                </div>
                <div className="actions">
                  <button
                    className="secondary has-progress"
                    disabled={cleaningMutation.isPending}
                    onClick={() => cleaningMutation.mutate(undefined)}
                  >
                    {cleaningMutation.isPending ? copy.viewer.cleaning : copy.viewer.runCleaning}
                    {cleaningMutation.isPending && (
                      <i className="button-progress-line" style={{ left: `${cleaningProgress}%` }} />
                    )}
                  </button>
                  <button
                    className="secondary"
                    disabled={cleaningMutation.isPending || selected === null}
                    onClick={() => selected !== null && cleaningMutation.mutate([selected])}
                  >
                    {cleaningMutation.isPending ? copy.viewer.cleaning : copy.viewer.runSelected}
                  </button>
                  <button
                    className="secondary"
                    disabled={selected === null || recordingMutation.isPending}
                    onClick={() => selected !== null && buildReplay(selected)}
                  >
                    {recordingMutation.isPending ? copy.viewer.buildingReplay : copy.viewer.replay}
                  </button>
                  {recordingUrl && (
                    <button
                      className="secondary"
                      disabled={recordingMutation.isPending}
                      onClick={() => {
                        setRecordingUrl(null);
                        setExportedResult(null);
                        setReplayReady(false);
                      }}
                    >
                      {copy.viewer.report}
                    </button>
                  )}
                </div>
              </div>

              <ExportPanel
                copy={copy}
                formats={exportFormats}
                selectedFormat={exportFormat}
                outputDir={exportOutputDir}
                scope={exportScope}
                exportCount={exportEpisodeIndexes.length}
                checkedCount={checkedEpisodeIndexes.size}
                cleaningReady={Boolean(cleaningSummary)}
                pending={exportMutation.isPending}
                disabled={exportMutation.isPending || exportEpisodeIndexes.length === 0}
                onFormatChange={setExportFormat}
                onOutputDirChange={setExportOutputDir}
                onScopeChange={setExportScope}
                onExport={() => exportMutation.mutate()}
              />

              <div className="viewer-stage viewer-stage-scroll">
                {activeFilterStage ? (
                  <FilterDetailView
                    copy={copy}
                    projectId={project.id}
                    detail={selectedFilterDetail}
                    pending={filterDetailQuery.isLoading}
                    error={filterDetailQuery.error}
                    onUrdfUpload={(file) => urdfUploadMutation.mutate(file)}
                    onConfigSave={(config) => filterConfigMutation.mutate(config)}
                    configSaving={filterConfigMutation.isPending}
                  />
                ) : recordingUrl ? (
                  <RerunViewer recordingUrl={recordingUrl} />
                ) : replayReady && selected !== null ? (
                  <div className="viewer-placeholder">
                    <div className="signal-preview">
                      <i /><i /><i /><i /><i /><i /><i /><i />
                    </div>
                    <h3>{copy.viewer.placeholderTitle}</h3>
                    <p>{copy.viewer.placeholderBody}</p>
                  </div>
                ) : cleaningSummary ? (
                  <CleaningSummaryView
                    copy={copy}
                    summary={cleaningSummary}
                    onSelect={(episodeIndex) => selectEpisode(episodeIndex)}
                  />
                ) : (
                  <div className="viewer-placeholder">
                    <div className="signal-preview">
                      <i /><i /><i /><i /><i /><i /><i /><i />
                    </div>
                    <h3>{copy.viewer.placeholderTitle}</h3>
                    <p>{copy.viewer.placeholderBody}</p>
                  </div>
                )}
              </div>

              {selectedEpisode && (
                <footer className="episode-detail">
                  <div><span>{copy.viewer.length}</span><strong>{selectedEpisode.length} {copy.viewer.frames}</strong></div>
                  <div><span>{copy.viewer.duration}</span><strong>{selectedEpisode.duration_seconds.toFixed(1)} s</strong></div>
                  <div className="task">
                    <span>{copy.viewer.task}</span>
                    <strong>{selectedEpisode.tasks[0]}</strong>
                    {(selectedEpisode.subtasks?.length ?? 0) > 0 && (
                      <div className="subtask-list">
                        <span>{copy.viewer.subtasks}</span>
                        {selectedEpisode.subtasks?.map((subtask, index) => (
                          <div className="subtask-row" key={`${subtask.start_frame}-${subtask.end_frame}-${index}`}>
                            <small>{subtaskTimeRange(subtask.start_seconds, subtask.end_seconds)}</small>
                            {subtask.skill && <em>{subtask.skill}</em>}
                            <b>{subtask.prompt}</b>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </footer>
              )}
              {exportedResult && (
                <div className="success">
                  {copy.viewer.exportedPrefix} {exportedResult.episode_count} {copy.viewer.episodeCountSuffix} {exportedResult.output_path}
                  <br />
                  {copy.viewer.report}: {exportedResult.report_path}
                </div>
              )}
            </section>

            <div
              aria-label={copy.quality.resizePanel}
              aria-orientation="vertical"
              aria-valuemax={maxQualityPanelWidth}
              aria-valuemin={minQualityPanelWidth}
              aria-valuenow={qualityPanelWidth}
              className="quality-resizer"
              role="separator"
              tabIndex={0}
              onPointerDown={resizeQualityPanel}
            />

            <QualityReport
              copy={copy}
              quality={selectedQuality}
              filterDetail={activeFilterStage ? selectedFilterDetail : null}
              filterPending={filterDetailQuery.isLoading || urdfUploadMutation.isPending || filterConfigMutation.isPending}
              pending={decisionMutation.isPending}
              requiresRerun={Boolean(cleaningSummary?.requires_rerun)}
              onDecision={(status) => {
                if (selected !== null) decisionMutation.mutate({ episodeIndex: selected, status });
              }}
              onAddToPipeline={() => cleaningMutation.mutate(undefined)}
            />
          </section>
        </>
      )}
    </main>
  );
}

function ExportPanel({
  copy,
  formats,
  selectedFormat,
  outputDir,
  scope,
  exportCount,
  checkedCount,
  cleaningReady,
  pending,
  disabled,
  onFormatChange,
  onOutputDirChange,
  onScopeChange,
  onExport,
}: {
  copy: Translation;
  formats: FormatInfo[];
  selectedFormat: string;
  outputDir: string;
  scope: ExportScope;
  exportCount: number;
  checkedCount: number;
  cleaningReady: boolean;
  pending: boolean;
  disabled: boolean;
  onFormatChange: (value: string) => void;
  onOutputDirChange: (value: string) => void;
  onScopeChange: (value: ExportScope) => void;
  onExport: () => void;
}) {
  return (
    <div className="export-panel">
      <label>
        <span>{copy.exportPanel.format}</span>
        <select
          aria-label={copy.exportPanel.format}
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
      <label className="export-output">
        <span>{copy.exportPanel.outputFolder}</span>
        <input
          aria-label={copy.exportPanel.outputFolder}
          value={outputDir}
          placeholder={copy.exportPanel.outputPlaceholder}
          onChange={(event) => onOutputDirChange(event.target.value)}
        />
      </label>
      <label>
        <span>{copy.exportPanel.scope}</span>
        <select
          aria-label={copy.exportPanel.scope}
          value={scope}
          onChange={(event) => onScopeChange(event.target.value as ExportScope)}
        >
          <option value="selected">{copy.exportPanel.selectedEpisode}</option>
          <option value="all">{copy.exportPanel.allEpisodes}</option>
          <option value="filtered">{copy.exportPanel.filteredEpisodes}</option>
          <option value="checked">{copy.exportPanel.checkedEpisodes}</option>
          <option value="status_passed" disabled={!cleaningReady}>{copy.exportPanel.passedEpisodes}</option>
          <option value="status_review" disabled={!cleaningReady}>{copy.exportPanel.reviewEpisodes}</option>
          <option value="status_excluded" disabled={!cleaningReady}>{copy.exportPanel.excludedEpisodes}</option>
        </select>
      </label>
      <div className="export-count" aria-live="polite">
        <span>{copy.exportPanel.exportingCount}</span>
        <strong>{checkedCount}</strong>
        <small>{copy.exportPanel.checkedCount(checkedCount)}</small>
      </div>
      <button type="button" disabled={disabled} onClick={onExport}>
        {pending ? copy.viewer.exporting : copy.exportPanel.exportCount(exportCount)}
      </button>
    </div>
  );
}

function CleaningSummaryView({
  copy,
  summary,
  onSelect,
}: {
  copy: Translation;
  summary: CleaningSummary;
  onSelect: (episodeIndex: number) => void;
}) {
  const scored = [...summary.results]
    .filter((result) => result.score !== null)
    .sort((left, right) => (left.score ?? 1) - (right.score ?? 1));
  const chartResults = [...summary.results].sort((left, right) => {
    if (left.score === null) return right.score === null ? left.episode_index - right.episode_index : 1;
    if (right.score === null) return -1;
    return left.score - right.score || left.episode_index - right.episode_index;
  });
  return (
    <div className="cleaning-summary-view">
      <div className="cleaning-summary-header">
        <div>
          <span className="eyebrow">{copy.cleaningSummary.eyebrow}</span>
          <h3>{copy.cleaningSummary.title}</h3>
        </div>
        <div className="summary-counts">
          <span className="passed">{copy.status.passed} {summary.passed_count}</span>
          <span className="review">{copy.status.review} {summary.review_count}</span>
          <span className="excluded">{copy.status.excluded} {summary.excluded_count}</span>
        </div>
      </div>
      <div className="score-chart">
        <div aria-hidden="true" className="score-axis">
          <span>100</span>
          <span>75</span>
          <span>50</span>
          <span>25</span>
          <span>0</span>
        </div>
        <div className="score-bars" role="list" aria-label={copy.cleaningSummary.scoreChart}>
          {chartResults.map((result) => {
            const score = result.score ?? 0;
            return (
              <button
                aria-label={`${episodeLabel(result.episode_index)} score ${Math.round(score * 100)} status ${statusLabel(result.status, copy)}`}
                className={`score-bar ${result.status}`}
                key={result.episode_index}
                onClick={() => onSelect(result.episode_index)}
                style={{ "--bar-height": `${Math.max(5, Math.round(score * 100))}%` } as CSSProperties}
                type="button"
              >
                <i />
              </button>
            );
          })}
        </div>
      </div>
      <div className="lowest-episodes">
        <h4>{copy.cleaningSummary.lowest}</h4>
        {scored.slice(0, 6).map((result) => (
          <button key={result.episode_index} type="button" onClick={() => onSelect(result.episode_index)}>
            <span>{compactEpisodeLabel(result.episode_index)}</span>
            <strong>{scoreLabel(result.score, copy)}</strong>
            <small className={result.status}>{statusLabel(result.status, copy)}</small>
          </button>
        ))}
      </div>
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
  copy,
  episodes,
  visibleEpisodes,
  qualityByEpisode,
  selected,
  checkedEpisodeIndexes,
  searchQuery,
  activeFindingFilters,
  filterSummary,
  activeFilterStage,
  enabledFilterStages,
  taskSuccessEnabled,
  kinematicsConfigured,
  weights,
  onSearchChange,
  onToggleFindingFilter,
  onToggleFilterStage,
  onOpenFilter,
  onWeightChange,
  onToggleChecked,
  onToggleGroupChecked,
  onSelect,
}: {
  copy: Translation;
  episodes: Episode[];
  visibleEpisodes: Episode[];
  qualityByEpisode: Map<number, EpisodeQualityResult>;
  selected: number | null;
  checkedEpisodeIndexes: Set<number>;
  searchQuery: string;
  activeFindingFilters: string[];
  filterSummary: FilterSummary | null;
  activeFilterStage: FilterStageId | null;
  enabledFilterStages: FilterStageId[];
  taskSuccessEnabled: boolean;
  kinematicsConfigured: boolean;
  weights: Record<string, number>;
  onSearchChange: (value: string) => void;
  onToggleFindingFilter: (code: string) => void;
  onToggleFilterStage: (stageId: FilterStageId) => void;
  onOpenFilter: (stageId: FilterStageId) => void;
  onWeightChange: (key: string, value: number) => void;
  onToggleChecked: (episodeIndex: number) => void;
  onToggleGroupChecked: (episodeIndexes: number[]) => void;
  onSelect: (episodeIndex: number) => void;
}) {
  const [collapsedStatusFolders, setCollapsedStatusFolders] = useState<CleaningStatus[]>([]);
  if (qualityByEpisode.size === 0) {
    return (
      <>
        <SidebarFilters
          copy={copy}
          searchQuery={searchQuery}
          activeFindingFilters={activeFindingFilters}
          filterSummary={filterSummary}
          activeFilterStage={activeFilterStage}
          enabledFilterStages={enabledFilterStages}
          taskSuccessEnabled={taskSuccessEnabled}
          kinematicsConfigured={kinematicsConfigured}
          weights={weights}
          onSearchChange={onSearchChange}
          onToggleFindingFilter={onToggleFindingFilter}
          onToggleFilterStage={onToggleFilterStage}
          onOpenFilter={onOpenFilter}
          onWeightChange={onWeightChange}
        />
        {visibleEpisodes.map((episode) => (
          <EpisodeRow
            key={episode.episode_index}
            episode={episode}
            copy={copy}
            quality={null}
            active={episode.episode_index === selected}
            checked={checkedEpisodeIndexes.has(episode.episode_index)}
            onToggleChecked={() => onToggleChecked(episode.episode_index)}
            onSelect={() => onSelect(episode.episode_index)}
          />
        ))}
      </>
    );
  }
  const groups: Array<{ status: CleaningStatus; label: string; episodes: Episode[] }> = [
    { status: "review", label: copy.status.review, episodes: [] },
    { status: "excluded", label: copy.status.excluded, episodes: [] },
    { status: "passed", label: copy.status.passed, episodes: [] },
  ];
  const folderEpisodes = hasOnlyDefaultSidebarFilters(activeFindingFilters)
    ? episodes.filter((episode) => episodeMatchesSearchQuery(episode, searchQuery))
    : visibleEpisodes;
  for (const episode of folderEpisodes) {
    const quality = qualityByEpisode.get(episode.episode_index);
    const group = groups.find((item) => item.status === quality?.status);
    if (group) group.episodes.push(episode);
  }
  return (
    <>
      <SidebarFilters
        copy={copy}
        searchQuery={searchQuery}
        activeFindingFilters={activeFindingFilters}
        filterSummary={filterSummary}
        activeFilterStage={activeFilterStage}
        enabledFilterStages={enabledFilterStages}
        taskSuccessEnabled={taskSuccessEnabled}
        kinematicsConfigured={kinematicsConfigured}
        weights={weights}
        onSearchChange={onSearchChange}
        onToggleFindingFilter={onToggleFindingFilter}
        onToggleFilterStage={onToggleFilterStage}
        onOpenFilter={onOpenFilter}
        onWeightChange={onWeightChange}
      />
      {groups.map((group) => (
        <StatusEpisodeFolder
          key={group.status}
          group={group}
          copy={copy}
          qualityByEpisode={qualityByEpisode}
          selected={selected}
          checkedEpisodeIndexes={checkedEpisodeIndexes}
          collapsed={collapsedStatusFolders.includes(group.status)}
          onToggleCollapsed={() =>
            setCollapsedStatusFolders((current) =>
              current.includes(group.status)
                ? current.filter((status) => status !== group.status)
                : [...current, group.status],
            )
          }
          onToggleChecked={onToggleChecked}
          onToggleGroupChecked={onToggleGroupChecked}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

function StatusEpisodeFolder({
  group,
  copy,
  qualityByEpisode,
  selected,
  checkedEpisodeIndexes,
  collapsed,
  onToggleCollapsed,
  onToggleChecked,
  onToggleGroupChecked,
  onSelect,
}: {
  group: { status: CleaningStatus; label: string; episodes: Episode[] };
  copy: Translation;
  qualityByEpisode: Map<number, EpisodeQualityResult>;
  selected: number | null;
  checkedEpisodeIndexes: Set<number>;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onToggleChecked: (episodeIndex: number) => void;
  onToggleGroupChecked: (episodeIndexes: number[]) => void;
  onSelect: (episodeIndex: number) => void;
}) {
  const episodeIndexes = group.episodes.map((episode) => episode.episode_index);
  const checkedCount = episodeIndexes.filter((episodeIndex) => checkedEpisodeIndexes.has(episodeIndex)).length;
  const allChecked = episodeIndexes.length > 0 && checkedCount === episodeIndexes.length;
  const partiallyChecked = checkedCount > 0 && !allChecked;
  return (
    <section className="episode-folder">
      <div className="folder-heading">
        <button
          type="button"
          className={`folder-toggle ${collapsed ? "" : "expanded"}`}
          aria-expanded={!collapsed}
          aria-label={`${collapsed ? "Expand" : "Collapse"} ${group.label} folder`}
          onClick={onToggleCollapsed}
        >
          <span aria-hidden="true">▸</span>
          <strong>{group.label}</strong>
        </button>
        <div className="folder-actions">
          <FolderSelectionCheckbox
            label={group.label}
            checked={allChecked}
            indeterminate={partiallyChecked}
            disabled={episodeIndexes.length === 0}
            onChange={() => onToggleGroupChecked(episodeIndexes)}
          />
          <small>{group.episodes.length}</small>
        </div>
      </div>
      {collapsed
        ? null
        : group.episodes.map((episode) => (
            <EpisodeRow
              key={episode.episode_index}
              episode={episode}
              copy={copy}
              quality={qualityByEpisode.get(episode.episode_index) ?? null}
              active={episode.episode_index === selected}
              checked={checkedEpisodeIndexes.has(episode.episode_index)}
              onToggleChecked={() => onToggleChecked(episode.episode_index)}
              onSelect={() => onSelect(episode.episode_index)}
            />
          ))}
    </section>
  );
}

function FolderSelectionCheckbox({
  label,
  checked,
  indeterminate,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  indeterminate: boolean;
  disabled: boolean;
  onChange: () => void;
}) {
  const checkboxRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (checkboxRef.current) {
      checkboxRef.current.indeterminate = indeterminate;
    }
  }, [indeterminate]);
  return (
    <label className="folder-check">
      <input
        ref={checkboxRef}
        type="checkbox"
        aria-label={`${checked ? "Clear all" : "Select all"} visible ${label} episodes for export`}
        checked={checked}
        disabled={disabled}
        onChange={onChange}
      />
    </label>
  );
}

function episodeMatchesFilters(
  episode: Episode,
  quality: EpisodeQualityResult | undefined,
  query: string,
  activeFilters: string[],
  filterSummary: FilterSummary | null,
) {
  if (!episodeMatchesSearchQuery(episode, query)) return false;
  if (hasOnlyDefaultSidebarFilters(activeFilters)) return true;
  if (activeFilters.length === 0 || (!quality && !filterSummary)) return true;
  return activeFilters.every((filter) => {
    if (isFilterStageId(filter)) {
      const stage = filterSummary?.episodes
        .find((item) => item.episode_index === episode.episode_index)
        ?.stage_status[filter];
      return Boolean(stage && !stage.skipped_reason && stage.count > 0);
    }
    return Boolean(quality?.findings.some((finding) => finding.code === filter));
  });
}

function episodeMatchesSearchQuery(episode: Episode, query: string) {
  const normalizedQuery = query.trim().toLowerCase();
  const text = [
    compactEpisodeLabel(episode.episode_index),
    episodeLabel(episode.episode_index),
    episode.tasks.join(" "),
    ...(episode.subtasks ?? []).flatMap((subtask) => [subtask.prompt, subtask.skill ?? ""]),
  ].join(" ").toLowerCase();
  if (normalizedQuery && !text.includes(normalizedQuery)) return false;
  return true;
}

function SidebarFilters({
  copy,
  searchQuery,
  activeFindingFilters,
  filterSummary,
  activeFilterStage,
  enabledFilterStages,
  taskSuccessEnabled,
  kinematicsConfigured,
  weights,
  onSearchChange,
  onToggleFindingFilter,
  onToggleFilterStage,
  onOpenFilter,
  onWeightChange,
}: {
  copy: Translation;
  searchQuery: string;
  activeFindingFilters: string[];
  filterSummary: FilterSummary | null;
  activeFilterStage: FilterStageId | null;
  enabledFilterStages: FilterStageId[];
  taskSuccessEnabled: boolean;
  kinematicsConfigured: boolean;
  weights: Record<string, number>;
  onSearchChange: (value: string) => void;
  onToggleFindingFilter: (code: string) => void;
  onToggleFilterStage: (stageId: FilterStageId) => void;
  onOpenFilter: (stageId: FilterStageId) => void;
  onWeightChange: (key: string, value: number) => void;
}) {
  const counts = new Map((filterSummary?.stages ?? []).map((stage) => [stage.id, stage]));
  const [expandedWeightStage, setExpandedWeightStage] = useState<QualityRuleId | null>(null);
  return (
    <div className="sidebar-tools">
      <label className="search-box">
        <span>{copy.filters.search}</span>
        <input
          aria-label={copy.filters.search}
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="#000184"
        />
      </label>
      <div className="filter-box">
        <span>{copy.filters.filters}</span>
        {findingFilters(copy).map((filter) => (
          <label key={filter.code}>
            <input
              type="checkbox"
              checked={activeFindingFilters.includes(filter.code)}
              onChange={() => onToggleFindingFilter(filter.code)}
            />
            {filter.label}
          </label>
        ))}
        {dataFilters(copy).map((filter) => {
          const stage = filter.stageId ? counts.get(filter.stageId) : undefined;
          const configured =
            filter.id === "kinematic_consistency"
              ? kinematicsConfigured
              : filter.id === "orientation_alignment"
              ? false
              : filter.id === "task_success"
              ? taskSuccessEnabled
              : true;
          const checked =
            filter.id === "task_success"
              ? taskSuccessEnabled
              : filter.stageId
              ? enabledFilterStages.includes(filter.stageId)
              : false;
          const badge =
            !configured && (filter.id === "task_success" || filter.id === "kinematic_consistency" || filter.id === "orientation_alignment")
              ? copy.filters.notConfigured
              : stage?.skipped_reason
              ? copy.filters.notConfigured
              : String(stage?.count ?? 0);
          const weight = weights[filter.id] ?? 1;
          const weightEnabled = checked && configured;
          const expanded = expandedWeightStage === filter.id;
          return (
            <div className={`filter-row ${activeFilterStage === filter.id ? "active" : ""}`} key={filter.id}>
              <div className="filter-row-main">
                <label>
                  <input
                    type="checkbox"
                    aria-label={filter.label}
                    checked={checked}
                    disabled={!configured}
                    onChange={() => {
                      if (filter.stageId) onToggleFilterStage(filter.stageId);
                    }}
                  />
                  <button
                    type="button"
                    className="filter-link"
                    disabled={!filter.stageId}
                    onClick={() => {
                      if (filter.stageId) onOpenFilter(filter.stageId);
                    }}
                  >
                    {filter.label}
                  </button>
                </label>
                <button
                  type="button"
                  className={`filter-expand-toggle ${expanded ? "expanded" : ""}`}
                  aria-label={`${expanded ? "Collapse" : "Expand"} ${filter.label} weight`}
                  aria-expanded={expanded}
                  onClick={() => setExpandedWeightStage((current) => (current === filter.id ? null : filter.id))}
                >
                  <span aria-hidden="true">▸</span>
                </button>
                <small>{badge}</small>
              </div>
              {expanded ? (
                <div className="filter-weight-panel">
                  <input
                    aria-label={`${filter.label} weight`}
                    type="range"
                    min="0.25"
                    max="3"
                    step="0.25"
                    value={weight}
                    disabled={!weightEnabled}
                    onChange={(event) => onWeightChange(filter.id, Number(event.target.value))}
                  />
                  <strong>{weight.toFixed(2)}x</strong>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EpisodeRow({
  copy,
  episode,
  quality,
  active,
  checked,
  onToggleChecked,
  onSelect,
}: {
  copy: Translation;
  episode: Episode;
  quality: EpisodeQualityResult | null;
  active: boolean;
  checked: boolean;
  onToggleChecked: () => void;
  onSelect: () => void;
}) {
  const label = episodeLabel(episode.episode_index);
  return (
    <div className={`episode-row-shell ${active ? "active" : ""}`}>
      <label className="episode-check">
        <input
          type="checkbox"
          aria-label={copy.exportPanel.toggleEpisode(label)}
          checked={checked}
          onChange={onToggleChecked}
        />
      </label>
      <button className="episode-row" onClick={onSelect}>
        <span>
          <strong>{quality ? compactEpisodeLabel(episode.episode_index) : label}</strong>
          <small>
            {quality
              ? `${scoreLabel(quality.score, copy)} · ${statusLabel(quality.status, copy)} · ${quality.findings.length} ${copy.quality.issues}`
              : `${episode.length} ${copy.viewer.frames} · ${episode.duration_seconds.toFixed(1)} s`}
          </small>
        </span>
        <b>›</b>
      </button>
    </div>
  );
}

function FilterDetailView({
  copy,
  projectId,
  detail,
  pending,
  error,
  onUrdfUpload,
  onConfigSave,
  configSaving,
}: {
  copy: Translation;
  projectId: string;
  detail: FilterDetail | null;
  pending: boolean;
  error: Error | null;
  onUrdfUpload: (file: File) => void;
  onConfigSave: (config: Partial<FilterConfig>) => void;
  configSaving: boolean;
}) {
  if (pending) {
    return <div className="viewer-placeholder"><h3>{copy.filterDetail.loading}</h3></div>;
  }
  if (error) {
    return <div className="viewer-placeholder"><h3>{copy.filterDetail.unavailable}</h3><p>{error.message}</p></div>;
  }
  if (!detail) {
    return <div className="viewer-placeholder"><h3>{copy.filterDetail.select}</h3></div>;
  }
  return (
    <div className="filter-detail-page">
      <div className="filter-detail-header">
        <div>
          <span className="eyebrow">{copy.filterDetail.eyebrow}</span>
          <h3>{detail.title}</h3>
        </div>
        <span className={`filter-status ${detail.status}`}>
          {detail.status === "passed" ? copy.status.passed : detail.status === "review" ? copy.status.review : copy.status.skipped}
        </span>
      </div>
      {detail.stage_id === "kinematic_consistency" && (
        <KinematicConfigStrip
          copy={copy}
          detail={detail}
          onUrdfUpload={onUrdfUpload}
          onConfigSave={onConfigSave}
          saving={configSaving}
        />
      )}
      {detail.stage_id === "visual_quality" && detail.visual_quality ? (
        <VisualQualityPanel copy={copy} projectId={projectId} detail={detail} />
      ) : detail.stage_id === "orientation_alignment" ? (
        <OrientationPanel detail={detail} />
      ) : detail.stage_id === "metadata_completeness" ? (
        <MetadataCompletenessPanel copy={copy} detail={detail} />
      ) : (
        <>
          <SignalChart detail={detail} />
          <FilterRows copy={copy} detail={detail} />
        </>
      )}
    </div>
  );
}

function VisualQualityPanel({
  copy,
  projectId,
  detail,
}: {
  copy: Translation;
  projectId: string;
  detail: FilterDetail;
}) {
  const evidence = detail.visual_quality!;
  const [camera, setCamera] = useState("all");
  const [issue, setIssue] = useState("all");
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [preview, setPreview] = useState<{ url: string; alt: string; frame: number } | null>(null);
  const [failedImages, setFailedImages] = useState<Set<string>>(() => new Set());
  const cameras = [...new Set(evidence.incidents.map((incident) => incident.camera))].sort();
  const issues = [...new Set(evidence.incidents.map((incident) => incident.issue))].sort();
  const incidents = evidence.incidents.filter(
    (incident) =>
      (camera === "all" || incident.camera === camera)
      && (issue === "all" || incident.issue === issue),
  );
  const issueRate = evidence.sampled_frame_count
    ? ((evidence.issue_sample_count / evidence.sampled_frame_count) * 100).toFixed(1)
    : "0.0";
  const summary = detail.status === "skipped"
    ? copy.filterDetail.visualSystemIssue
    : detail.status === "passed" && evidence.issue_sample_count
      ? copy.filterDetail.visualPassWithIssues(evidence.issue_sample_count)
      : detail.status === "review"
        ? copy.filterDetail.visualReviewSummary(evidence.issue_sample_count)
        : copy.filterDetail.visualClean;
  const systemIssues = detail.table_rows.filter((row) =>
    ["video_missing", "decode_failed"].includes(String(row.issue ?? "")),
  );

  return (
    <div className="visual-quality-panel">
      <section className={`visual-quality-summary ${detail.status}`}>
        <div>
          <span className="eyebrow">{copy.filterDetail.visualEvidence}</span>
          <strong>{summary}</strong>
        </div>
        <div className="visual-quality-metrics">
          <div><b>{copy.filterDetail.visualIssueIntervals(evidence.incidents.length)}</b><small>{copy.filterDetail.visualTimeline}</small></div>
          <div><b>{copy.filterDetail.visualAffectedCameras(evidence.affected_camera_count, evidence.camera_count)}</b><small>{copy.filterDetail.visualCamera}</small></div>
          <div><b>{copy.filterDetail.visualSampledFrames(evidence.sampled_frame_count)}</b><small>{evidence.episode_frame_count} total</small></div>
          <div><b>{copy.filterDetail.visualFlaggedRate(issueRate)}</b><small>{evidence.issue_sample_count} flagged</small></div>
        </div>
      </section>

      {evidence.incidents.length > 0 && (
        <section className="visual-issue-timeline" aria-label={copy.filterDetail.visualTimeline}>
          <div className="visual-section-heading">
            <strong>{copy.filterDetail.visualTimeline}</strong>
            <small>0–{formatSeconds(evidence.episode_duration_seconds)}</small>
          </div>
          <div className="visual-timeline-track">
            {evidence.incidents.map((incident) => {
              const duration = Math.max(evidence.episode_duration_seconds, 0.001);
              const left = Math.min(100, Math.max(0, incident.start_timestamp / duration * 100));
              const width = Math.max(
                0.8,
                (incident.end_timestamp - incident.start_timestamp) / duration * 100,
              );
              return (
                <button
                  key={incident.id}
                  type="button"
                  className={`visual-timeline-marker issue-${incident.issue}`}
                  style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
                  title={`${visualIssueLabel(copy, incident.issue)} · ${formatSeconds(incident.start_timestamp)}`}
                  aria-label={`${visualIssueLabel(copy, incident.issue)} ${formatSeconds(incident.start_timestamp)}`}
                  onClick={() => document
                    .getElementById(`visual-incident-${safeDomId(incident.id)}`)
                    ?.scrollIntoView({ behavior: "smooth", block: "nearest" })}
                />
              );
            })}
          </div>
        </section>
      )}

      {evidence.incidents.length > 0 && (
        <div className="visual-evidence-toolbar">
          <label>
            <span>{copy.filterDetail.visualCamera}</span>
            <select value={camera} onChange={(event) => setCamera(event.target.value)}>
              <option value="all">{copy.filterDetail.visualAllCameras}</option>
              {cameras.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>
            <span>{copy.filterDetail.visualIssueType}</span>
            <select value={issue} onChange={(event) => setIssue(event.target.value)}>
              <option value="all">{copy.filterDetail.visualAllIssues}</option>
              {issues.map((name) => (
                <option key={name} value={name}>{visualIssueLabel(copy, name)}</option>
              ))}
            </select>
          </label>
        </div>
      )}

      {systemIssues.map((row, index) => (
        <div className="visual-system-issue" key={`${String(row.camera)}-${index}`}>
          <strong>{copy.filterDetail.visualSystemIssue}</strong>
          <span>{String(row.camera ?? "-")} · {visualIssueLabel(copy, String(row.issue ?? ""))}</span>
        </div>
      ))}

      {evidence.incidents.length === 0 && systemIssues.length === 0 ? (
        <div className="visual-clean-state">
          <strong>{copy.filterDetail.visualClean}</strong>
          <span>{copy.filterDetail.visualSampledFrames(evidence.sampled_frame_count)} · {evidence.camera_count} cameras</span>
        </div>
      ) : evidence.incidents.length > 0 ? (
        <div className="visual-incident-grid">
          {incidents.map((incident) => (
            <VisualIncidentCard
              key={incident.id}
              copy={copy}
              projectId={projectId}
              episodeIndex={detail.episode_index}
              incident={incident}
              failedImages={failedImages}
              onImageError={(url) => setFailedImages((current) => new Set(current).add(url))}
              onPreview={setPreview}
            />
          ))}
        </div>
      ) : null}

      {Object.keys(evidence.metrics).length > 0 && <section className="visual-metrics-disclosure">
        <button
          type="button"
          aria-label={copy.filterDetail.visualMetrics}
          aria-expanded={metricsOpen}
          onClick={() => setMetricsOpen((open) => !open)}
        >
          <span>{copy.filterDetail.visualMetrics}</span>
          <span>{metricsOpen ? "−" : "+"}</span>
        </button>
        {metricsOpen && <VisualMetricsCharts detail={detail} />}
      </section>}

      {preview && (
        <div className="visual-lightbox" role="dialog" aria-modal="true" aria-label={preview.alt}>
          <button type="button" onClick={() => setPreview(null)}>{copy.filterDetail.visualClosePreview}</button>
          <img src={preview.url} alt={preview.alt} />
          <strong>{copy.filterDetail.visualFrame} {preview.frame}</strong>
        </div>
      )}
    </div>
  );
}

function VisualIncidentCard({
  copy,
  projectId,
  episodeIndex,
  incident,
  failedImages,
  onImageError,
  onPreview,
}: {
  copy: Translation;
  projectId: string;
  episodeIndex: number;
  incident: VisualQualityIncident;
  failedImages: Set<string>;
  onImageError: (url: string) => void;
  onPreview: (preview: { url: string; alt: string; frame: number }) => void;
}) {
  return (
    <article
      className={`visual-incident-card issue-${incident.issue}`}
      id={`visual-incident-${safeDomId(incident.id)}`}
    >
      <header>
        <div>
          <span className="visual-issue-pill">{visualIssueLabel(copy, incident.issue)}</span>
          <strong>{incident.camera}</strong>
        </div>
        <small>{incident.sample_count} {copy.filterDetail.visualSamples}</small>
      </header>
      <div className="visual-incident-range">
        <b>
          {incident.start_frame === incident.end_frame
            ? `${copy.filterDetail.visualFrame} ${incident.start_frame}`
            : `${copy.filterDetail.visualFrame} ${incident.start_frame}–${incident.end_frame}`}
        </b>
        <span>
          {formatSeconds(incident.start_timestamp)}
          {incident.end_timestamp > incident.start_timestamp
            ? `–${formatSeconds(incident.end_timestamp)}`
            : ""}
        </span>
      </div>
      <div className="visual-evidence-strip">
        {incident.representative_frames.map((frame) => {
          const url = visualEvidenceUrl(projectId, episodeIndex, incident.camera, frame.frame, 640);
          const alt = `${visualIssueLabel(copy, incident.issue)} evidence at frame ${frame.frame} from ${incident.camera}`;
          return (
            <button
              type="button"
              className="visual-evidence-image"
              key={`${frame.frame}-${frame.timestamp}`}
              onClick={() => onPreview({
                url: visualEvidenceUrl(projectId, episodeIndex, incident.camera, frame.frame, 1600),
                alt,
                frame: frame.frame,
              })}
            >
              {failedImages.has(url) ? (
                <span>{copy.filterDetail.visualImageUnavailable}</span>
              ) : (
                <img src={url} alt={alt} onError={() => onImageError(url)} />
              )}
              <small>{copy.filterDetail.visualFrame} {frame.frame}</small>
            </button>
          );
        })}
      </div>
      <footer>
        <span>{copy.filterDetail.visualObserved}: <b>{String(incident.worst_value)}</b></span>
        <span>{copy.filterDetail.visualThreshold}: <b>{String(incident.threshold)}</b></span>
      </footer>
    </article>
  );
}

function VisualMetricsCharts({ detail }: { detail: FilterDetail }) {
  const evidence = detail.visual_quality!;
  const thresholds = detail.thresholds.visual_quality ?? {};
  const definitions: Array<{
    key: "sharpness" | "brightness" | "contrast";
    thresholds: number[];
  }> = [
    { key: "sharpness", thresholds: [thresholds.blur_laplacian].filter(Number.isFinite) },
    { key: "brightness", thresholds: [thresholds.dark_mean, thresholds.bright_mean].filter(Number.isFinite) },
    { key: "contrast", thresholds: [thresholds.low_contrast_std].filter(Number.isFinite) },
  ];
  return (
    <div className="visual-camera-charts">
      {Object.entries(evidence.metrics).map(([camera, samples]) => (
        <section key={camera}>
          <strong>{camera}</strong>
          <div className="visual-metric-grid">
            {definitions.map((definition) => (
              <VisualMetricChart
                key={definition.key}
                label={definition.key}
                metric={definition.key}
                samples={samples}
                duration={evidence.episode_duration_seconds}
                thresholds={definition.thresholds}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function VisualMetricChart({
  label,
  metric,
  samples,
  duration,
  thresholds,
}: {
  label: string;
  metric: "sharpness" | "brightness" | "contrast";
  samples: VisualQualityMetricSample[];
  duration: number;
  thresholds: number[];
}) {
  const values = samples
    .map((sample) => sample[metric])
    .filter((value): value is number => value !== null && Number.isFinite(value));
  const domain = [...values, ...thresholds];
  const min = Math.min(...domain, 0);
  const max = Math.max(...domain, 1);
  const span = Math.max(max - min, 1e-9);
  const y = (value: number) => 108 - ((value - min) / span) * 92;
  const points = samples
    .filter((sample) => sample[metric] !== null)
    .map((sample) => {
      const x = 8 + sample.timestamp / Math.max(duration, 0.001) * 704;
      return `${x.toFixed(1)},${y(sample[metric] as number).toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="visual-metric-chart">
      <span>{label}</span>
      <svg viewBox="0 0 720 120" role="img" aria-label={`${label} over time`}>
        <rect x="0" y="0" width="720" height="120" rx="8" />
        {thresholds.map((threshold) => (
          <line
            className="visual-threshold-line"
            key={threshold}
            x1="0"
            x2="720"
            y1={y(threshold)}
            y2={y(threshold)}
          />
        ))}
        <polyline className="visual-metric-line" points={points} />
      </svg>
    </div>
  );
}

function visualEvidenceUrl(
  projectId: string,
  episodeIndex: number,
  camera: string,
  frame: number,
  width: number,
) {
  return `/api/projects/${projectId}/episodes/${episodeIndex}/visual-quality/frame?camera=${encodeURIComponent(camera)}&frame=${frame}&width=${width}`;
}

function visualIssueLabel(copy: Translation, issue: string) {
  return copy.filterDetail.visualIssueLabels[issue] ?? issue;
}

function formatSeconds(value: number) {
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`;
}

function safeDomId(value: string) {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "-");
}

function KinematicConfigStrip({
  copy,
  detail,
  onUrdfUpload,
  onConfigSave,
  saving,
}: {
  copy: Translation;
  detail: FilterDetail;
  onUrdfUpload: (file: File) => void;
  onConfigSave: (config: Partial<FilterConfig>) => void;
  saving: boolean;
}) {
  const parameters = detail.parameters as Record<string, unknown>;
  const [endEffectorLink, setEndEffectorLink] = useState("");
  const [jointNames, setJointNames] = useState("");
  const [jointStateIndices, setJointStateIndices] = useState("");
  const [eefPositionIndices, setEefPositionIndices] = useState("");
  const [positionTolerance, setPositionTolerance] = useState("0.05");
  const [resolveTcpOffset, setResolveTcpOffset] = useState(true);

  useEffect(() => {
    setEndEffectorLink(String(parameters.end_effector_link ?? ""));
    setJointNames((parameters.joint_names as string[] | undefined)?.join(", ") ?? "");
    setJointStateIndices((parameters.joint_state_indices as number[] | undefined)?.join(", ") ?? "");
    setEefPositionIndices((parameters.eef_position_indices as number[] | undefined)?.join(", ") ?? "");
    setPositionTolerance(String(parameters.position_tolerance ?? 0.05));
    setResolveTcpOffset(Boolean(parameters.resolve_tcp_offset ?? true));
  }, [parameters]);

  const save = () => {
    onConfigSave({
      kinematics: {
        urdf_path: (parameters.urdf_path as string | null | undefined) ?? null,
        end_effector_link: endEffectorLink.trim() || null,
        joint_names: parseStringList(jointNames),
        joint_state_indices: parseNumberList(jointStateIndices),
        eef_position_indices: parseNumberList(eefPositionIndices),
        position_tolerance: Number(positionTolerance) || 0.05,
        resolve_tcp_offset: resolveTcpOffset,
      },
    });
  };

  return (
    <div className="kinematic-config">
      <label className="file-button">
        <span>{copy.kinematics.importUrdf}</span>
        <input
          aria-label={copy.kinematics.importUrdf}
          type="file"
          accept=".urdf"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onUrdfUpload(file);
          }}
        />
      </label>
      <label>
        <span>{copy.kinematics.endEffectorLink}</span>
        <input
          aria-label={copy.kinematics.endEffectorLink}
          value={endEffectorLink}
          onChange={(event) => setEndEffectorLink(event.target.value)}
        />
      </label>
      <label>
        <span>{copy.kinematics.jointNames}</span>
        <input
          aria-label={copy.kinematics.jointNames}
          value={jointNames}
          onChange={(event) => setJointNames(event.target.value)}
        />
      </label>
      <label>
        <span>{copy.kinematics.jointStateIndices}</span>
        <input
          aria-label={copy.kinematics.jointStateIndices}
          value={jointStateIndices}
          onChange={(event) => setJointStateIndices(event.target.value)}
        />
      </label>
      <label>
        <span>{copy.kinematics.eefPositionIndices}</span>
        <input
          aria-label={copy.kinematics.eefPositionIndices}
          value={eefPositionIndices}
          onChange={(event) => setEefPositionIndices(event.target.value)}
        />
      </label>
      <label>
        <span>{copy.kinematics.tolerance}</span>
        <input
          aria-label={copy.kinematics.positionTolerance}
          type="number"
          step="0.01"
          value={positionTolerance}
          onChange={(event) => setPositionTolerance(event.target.value)}
        />
      </label>
      <label className="inline-toggle">
        <input
          aria-label={copy.kinematics.tcpOffset}
          type="checkbox"
          checked={resolveTcpOffset}
          onChange={(event) => setResolveTcpOffset(event.target.checked)}
        />
        <span>{copy.kinematics.tcpOffset}</span>
      </label>
      <button type="button" className="secondary compact" onClick={save} disabled={saving}>
        {saving ? copy.kinematics.saving : copy.kinematics.saveConfig}
      </button>
    </div>
  );
}

function parseStringList(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value: string) {
  return value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function SignalChart({ detail }: { detail: FilterDetail }) {
  const entries = Object.entries(detail.series).slice(0, 3);
  const threshold = Object.entries(detail.thresholds)[0]?.[1] ?? {};
  return (
    <div className="filter-chart">
      <div className="chart-labels">
        {entries.map(([name]) => <span key={name}>{name}</span>)}
        {Object.keys(threshold).map((name) => <b key={name}>{name}</b>)}
      </div>
      <svg viewBox="0 0 720 260" role="img" aria-label={`${detail.title} visualization`}>
        <rect x="0" y="0" width="720" height="260" rx="8" />
        <g className="grid">
          {Array.from({ length: 6 }, (_, index) => <line key={`h-${index}`} x1="0" x2="720" y1={index * 52} y2={index * 52} />)}
          {Array.from({ length: 9 }, (_, index) => <line key={`v-${index}`} y1="0" y2="260" x1={index * 90} x2={index * 90} />)}
        </g>
        {entries.map(([name, values], index) => (
          <polyline key={name} className={`series series-${index}`} points={polylinePoints(values, 720, 220, 20)} />
        ))}
        {detail.table_rows.slice(0, 12).map((row, index) => (
          <line
            className="flag-line"
            key={index}
            x1={Math.min(700, Number(row.frame ?? 0) * 4)}
            x2={Math.min(700, Number(row.frame ?? 0) * 4)}
            y1="20"
            y2="240"
          />
        ))}
      </svg>
    </div>
  );
}

function OrientationPanel({ detail }: { detail: FilterDetail }) {
  return (
    <div className="orientation-panel">
      <div className="axis-scene">
        <span className="axis x">x</span>
        <span className="axis y">y</span>
        <span className="axis z">z</span>
      </div>
      <SignalChart detail={detail} />
    </div>
  );
}

type MetadataInspectionRow = {
  kind?: string;
  check?: string;
  scope?: string;
  status?: string;
  label?: string;
  expected?: string;
  value?: string;
  detail?: string | null;
};

function metadataInspectionRows(detail: FilterDetail): MetadataInspectionRow[] {
  return detail.table_rows.filter((row) => row.kind === "inspection") as MetadataInspectionRow[];
}

function metadataInspectionLabel(copy: Translation, row: MetadataInspectionRow) {
  const check = String(row.check ?? "");
  return copy.filterDetail.metadataChecks[check] ?? row.label ?? check;
}

function metadataInspectionStatus(copy: Translation, status: string | undefined) {
  if (status === "warning") return copy.filterDetail.metadataStatusWarning;
  if (status === "info") return copy.filterDetail.metadataStatusInfo;
  return copy.filterDetail.metadataStatusPassed;
}

function MetadataCompletenessPanel({
  copy,
  detail,
}: {
  copy: Translation;
  detail: FilterDetail;
}) {
  const rows = metadataInspectionRows(detail);
  const summary = (detail.parameters.summary ?? {}) as {
    passed?: number;
    total_checks?: number;
    warnings?: number;
    infos?: number;
  };
  const passed = summary.passed ?? rows.filter((row) => row.status === "passed").length;
  const total = summary.total_checks ?? rows.length;
  const allPassed = detail.findings.length === 0;

  return (
    <div className="metadata-inspection-panel">
      <div className="metadata-inspection-summary">
        <strong>{copy.filterDetail.metadataSummary(passed, total)}</strong>
        <small>{allPassed ? copy.filterDetail.metadataAllPassed : copy.quality.issuesFound(detail.findings.length)}</small>
      </div>
      <div className="metadata-inspection-table">
        <div className="metadata-inspection-header">
          <span>{copy.filterDetail.metadataCheck}</span>
          <span>{copy.filterDetail.metadataScope}</span>
          <span>{copy.filterDetail.metadataExpected}</span>
          <span>{copy.filterDetail.metadataActual}</span>
          <span>{copy.status.passed}</span>
        </div>
        {rows.map((row) => (
          <div className="metadata-inspection-row" key={String(row.check)}>
            <strong>{metadataInspectionLabel(copy, row)}</strong>
            <span>
              {row.scope === "dataset"
                ? copy.filterDetail.metadataScopeDataset
                : copy.filterDetail.metadataScopeEpisode}
            </span>
            <span>{String(row.expected ?? "-")}</span>
            <span>
              {String(row.value ?? "-")}
              {row.detail ? <small>{`${copy.filterDetail.metadataDetail}: ${row.detail}`}</small> : null}
            </span>
            <span className={`inspection-status ${row.status ?? "passed"}`}>
              {metadataInspectionStatus(copy, row.status)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FilterRows({ copy, detail }: { copy: Translation; detail: FilterDetail }) {
  const rows = detail.table_rows.slice(0, 8);
  return (
    <div className="filter-table">
      {rows.length === 0 ? (
        <p>{detail.skipped_reason ? copy.filterDetail.needsConfig : copy.filterDetail.noFlaggedFrames}</p>
      ) : (
        rows.map((row, index) => (
          <div key={index}>
            <span>frame {String(row.frame ?? "-")}</span>
            <strong>{String(row.dimension ?? row.source ?? row.issue ?? row.status ?? row.camera ?? "-")}</strong>
            <small>
              {Object.entries(row)
                .filter(([key]) => key !== "frame")
                .slice(0, 4)
                .map(([key, value]) => `${key}: ${String(value)}`)
                .join(" · ")}
            </small>
          </div>
        ))
      )}
    </div>
  );
}

function polylinePoints(values: number[], width: number, height: number, top: number) {
  if (values.length === 0) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
      const y = top + height - ((value - min) / span) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function QualityReport({
  copy,
  quality,
  filterDetail,
  filterPending,
  pending,
  requiresRerun,
  onDecision,
  onAddToPipeline,
}: {
  copy: Translation;
  quality: EpisodeQualityResult | null;
  filterDetail: FilterDetail | null;
  filterPending: boolean;
  pending: boolean;
  requiresRerun: boolean;
  onDecision: (status: "passed" | "excluded") => void;
  onAddToPipeline: () => void;
}) {
  if (filterDetail || filterPending) {
    return <FilterReport copy={copy} detail={filterDetail} pending={filterPending} onAddToPipeline={onAddToPipeline} />;
  }
  return (
    <aside className="quality-panel">
      <div className="panel-heading">
        <span>{copy.quality.report}</span>
      </div>
      {!quality ? (
        <div className="quality-empty">
          <strong>{copy.quality.notScored}</strong>
          <p>{copy.quality.runPipeline}</p>
        </div>
      ) : (
        <div className="quality-body">
          {requiresRerun && (
            <div className="finding">
              <strong>{copy.quality.rerunRequired}</strong>
              <p>{copy.quality.rerunExplanation}</p>
            </div>
          )}
          <div className="quality-score">
            <span>{scoreLabel(quality.score, copy)}</span>
            <small>{statusLabel(quality.status, copy)}{copy.quality.sourceSeparator}{quality.source}</small>
          </div>
          <div className="score-list">
            <div>
              <span>{copy.quality.dataQuality}</span>
              <strong>
                {quality.data_quality_score === null
                  ? copy.quality.notEvaluated
                  : Math.round(quality.data_quality_score * 100)}
              </strong>
            </div>
            <div>
              <span>{copy.quality.taskSuccess}</span>
              <strong>
                {quality.task_success_score === null
                  ? copy.quality.notEvaluated
                  : Math.round(quality.task_success_score * 100)}
              </strong>
            </div>
            {Object.entries(quality.per_attribute_scores).map(([name, value]) => (
              <div key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <strong>{Math.round(value * 100)}</strong>
              </div>
            ))}
          </div>
          <div className="finding-list">
            <h3>{copy.quality.issuesFound(quality.findings.length)}</h3>
            {quality.findings.length === 0 ? (
              <p>{copy.quality.noFindings}</p>
            ) : (
              quality.findings.map((finding) => (
                <div className="finding" key={`${finding.code}-${finding.message}`}>
                  {finding.message}
                </div>
              ))
            )}
          </div>
          <div className="review-actions">
            {quality.status !== "passed" && (
              <button disabled={pending} onClick={() => onDecision("passed")}>{copy.status.passed}</button>
            )}
            {quality.status !== "excluded" && (
              <button className="secondary danger" disabled={pending} onClick={() => onDecision("excluded")}>{copy.status.excluded}</button>
            )}
            <button className="secondary span-all" disabled={pending} onClick={onAddToPipeline}>
              {copy.quality.addToPipeline}
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

function FilterReport({
  copy,
  detail,
  pending,
  onAddToPipeline,
}: {
  copy: Translation;
  detail: FilterDetail | null;
  pending: boolean;
  onAddToPipeline: () => void;
}) {
  return (
    <aside className="quality-panel">
      <div className="panel-heading">
        <span>{copy.filterDetail.detail}</span>
      </div>
      {pending || !detail ? (
        <div className="quality-empty"><strong>{copy.filterDetail.loadingShort}</strong></div>
      ) : (
        <div className="quality-body">
          <div className="quality-score">
            <span>{detail.status === "review" ? copy.status.review : detail.status === "passed" ? copy.status.passed : copy.status.skipped}</span>
            <small>{detail.title}</small>
          </div>
          {detail.stage_id === "metadata_completeness" ? (
            <>
              <div className="score-list">
                {Object.entries((detail.parameters.summary ?? {}) as Record<string, number>).map(([name, value]) => (
                  <div key={name}>
                    <span>{name.replaceAll("_", " ")}</span>
                    <strong>{String(value)}</strong>
                  </div>
                ))}
              </div>
              <div className="finding-list">
                <h3>{copy.quality.issuesFound(detail.findings.length)}</h3>
                {detail.findings.length === 0 ? (
                  <p>{copy.filterDetail.metadataAllPassed}</p>
                ) : (
                  detail.findings.map((finding) => (
                    <div className="finding" key={`${finding.code}-${finding.message}`}>{finding.message}</div>
                  ))
                )}
              </div>
              <details className="metadata-config-details">
                <summary>{copy.filterDetail.metadataConfig}</summary>
                <div className="score-list">
                  {Object.entries(detail.parameters)
                    .filter(([name]) => name !== "summary")
                    .map(([name, value]) => (
                      <div key={name}>
                        <span>{name.replaceAll("_", " ")}</span>
                        <strong>{Array.isArray(value) ? value.join(", ") || "-" : String(value ?? "-")}</strong>
                      </div>
                    ))}
                </div>
              </details>
            </>
          ) : (
            <>
              <div className="score-list">
                {Object.entries(detail.parameters).map(([name, value]) => (
                  <div key={name}>
                    <span>{name.replaceAll("_", " ")}</span>
                    <strong>{Array.isArray(value) ? value.join(", ") || "-" : String(value ?? "-")}</strong>
                  </div>
                ))}
              </div>
              <div className="finding-list">
                <h3>{copy.quality.issuesFound(detail.findings.length)}</h3>
                {detail.findings.map((finding) => (
                  <div className="finding" key={`${finding.code}-${finding.message}`}>{finding.message}</div>
                ))}
              </div>
            </>
          )}
          <div className="review-actions">
            <button disabled={pending}>{copy.status.passed}</button>
            <button className="secondary danger" disabled={pending}>{copy.status.excluded}</button>
            <button className="secondary span-all" disabled={pending} onClick={onAddToPipeline}>{copy.quality.addToPipeline}</button>
          </div>
        </div>
      )}
    </aside>
  );
}

function VlmSettingsPanel({
  copy,
  settings,
  onChange,
}: {
  copy: Translation;
  settings: VlmSettings;
  onChange: (settings: VlmSettings) => void;
}) {
  const update = (changes: Partial<VlmSettings>) => onChange({ ...settings, ...changes });
  return (
    <div className="vlm-popover">
      <label className="checkbox-row">
        <input
          type="checkbox"
          aria-label={copy.vlm.enable}
          checked={settings.enabled}
          onChange={(event) => update({ enabled: event.target.checked })}
        />
        {copy.vlm.enable}
      </label>
      <label>
        <span>{copy.vlm.provider}</span>
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
        <span>{copy.vlm.model}</span>
        <input
          aria-label={copy.vlm.modelAria}
          value={settings.model}
          onChange={(event) => update({ model: event.target.value })}
        />
      </label>
      <label>
        <span>{copy.vlm.apiBaseUrl}</span>
        <input
          aria-label="VLM API Base URL"
          placeholder="https://api.openai.com/v1"
          value={settings.api_base_url ?? ""}
          onChange={(event) => update({ api_base_url: event.target.value || null })}
        />
      </label>
      <label>
        <span>{copy.vlm.apiKey}</span>
        <input
          aria-label="VLM API Key"
          type="password"
          placeholder={settings.api_key_configured ? copy.vlm.apiKeyConfiguredPlaceholder : copy.vlm.apiKeyPlaceholder}
          value={settings.api_key ?? ""}
          onChange={(event) => update({ api_key: event.target.value || null })}
        />
        {settings.api_key_configured && !settings.api_key ? (
          <small>{copy.vlm.apiKeyConfigured}</small>
        ) : null}
      </label>
      <label>
        <span>{copy.vlm.sampleFrames}</span>
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
        <span>{copy.vlm.prompt}</span>
        <textarea
          aria-label="VLM Prompt"
          value={settings.prompt}
          onChange={(event) => update({ prompt: event.target.value })}
        />
      </label>
    </div>
  );
}
