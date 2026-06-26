export type DatasetMetadata = {
  path: string;
  format: string;
  version: string;
  robot_type: string;
  total_episodes: number;
  total_frames: number;
  fps: number;
  video_keys: string[];
  scalar_keys: string[];
  features: Record<string, unknown>;
};

export type Project = {
  id: string;
  dataset: DatasetMetadata;
};

export type FormatInfo = {
  id: string;
  label: string;
  profile: string;
  can_import: boolean;
  can_export: boolean;
  requires_extra?: boolean;
};

export type Episode = {
  episode_index: number;
  length: number;
  duration_seconds: number;
  tasks: string[];
  subtasks?: EpisodeSubtask[];
  data_file: string;
  video_files: Record<string, string>;
  video_start_seconds: Record<string, number>;
  video_end_seconds: Record<string, number>;
};

export type EpisodeSubtask = {
  start_frame: number;
  end_frame: number;
  start_seconds: number;
  end_seconds: number;
  prompt: string;
  skill: string | null;
  track: string | null;
  is_mistake: boolean;
};

export type CleaningStatus = "passed" | "review" | "excluded" | "unscored";

export type QualityFinding = {
  code: string;
  severity: "info" | "warn" | "error";
  message: string;
};

export type EpisodeQualityResult = {
  episode_index: number;
  score: number | null;
  status: CleaningStatus;
  source: "auto" | "manual";
  per_attribute_scores: Record<string, number>;
  findings: QualityFinding[];
  review_note: string | null;
  updated_at: string;
};

export type CleaningSummary = {
  total: number;
  passed_count: number;
  review_count: number;
  excluded_count: number;
  unscored_count: number;
  results: EpisodeQualityResult[];
  config: {
    pass_threshold: number;
    review_threshold: number;
    overwrite_manual: boolean;
    enabled_filter_stages: FilterStageId[];
    quality_weights: Record<string, number>;
  };
  scorer_version: string;
};

export type CleaningRun = {
  run_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  summary: CleaningSummary;
};

export type ExportResult = {
  output_path: string;
  report_path: string;
  format: string;
  episode_count: number;
};

export type VlmSettings = {
  enabled: boolean;
  provider: string;
  model: string;
  api_base_url: string | null;
  api_key?: string | null;
  api_key_configured?: boolean;
  prompt: string;
  sample_frames: number;
};

export type FilterStageId =
  | "sudden_change"
  | "state_action_alignment"
  | "extreme_value"
  | "kinematic_consistency"
  | "orientation_alignment";

export type FilterStatus = "passed" | "review" | "skipped";

export type FilterFinding = {
  code: string;
  severity: "info" | "warn" | "error";
  message: string;
};

export type FilterDetail = {
  stage_id: FilterStageId;
  episode_index: number;
  title: string;
  status: FilterStatus;
  series: Record<string, number[]>;
  thresholds: Record<string, Record<string, number>>;
  table_rows: Array<Record<string, unknown>>;
  parameters: Record<string, unknown>;
  findings: FilterFinding[];
  skipped_reason: string | null;
};

export type FilterSummary = {
  dataset_path: string;
  total_episodes: number;
  total_frames: number;
  stages: Array<{
    id: FilterStageId;
    label: string;
    count: number;
    status: FilterStatus;
    skipped_reason: string | null;
  }>;
  episodes: Array<{
    episode_index: number;
    stage_status: Record<FilterStageId, { count: number; status: FilterStatus; skipped_reason: string | null }>;
  }>;
};

export type FilterRun = {
  run_id: string;
  status: "succeeded" | "failed";
  summary: FilterSummary;
};

export type FilterConfig = {
  gripper_indices: number[];
  kinematics: {
    urdf_path: string | null;
    end_effector_link: string | null;
    joint_names: string[];
    joint_state_indices: number[];
    eef_position_indices: number[];
    position_tolerance: number;
    resolve_tcp_offset: boolean;
  };
};

export type CleaningRuleConfig = {
  enabled_filter_stages: FilterStageId[];
  quality_weights: Record<string, number>;
};

function normalizeFilesystemPath(path: string) {
  return path.trim().replace(/^(['"])(.*)\1$/, "$2").trim();
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json();
}

export function listFormats() {
  return request<FormatInfo[]>("/api/formats");
}

export function importDataset(path: string, formatHint?: string) {
  const normalizedPath = normalizeFilesystemPath(path);
  const body = formatHint && formatHint !== "auto" ? { path: normalizedPath, format_hint: formatHint } : { path: normalizedPath };
  return request<Project>("/api/projects", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function listEpisodes(projectId: string) {
  return request<Episode[]>(`/api/projects/${projectId}/episodes?limit=500`);
}

export function createRecording(projectId: string, episodeIndex: number) {
  return request<{ recording_url: string }>(
    `/api/projects/${projectId}/episodes/${episodeIndex}/recording`,
    { method: "POST" },
  );
}

export function exportDataset(projectId: string, episodeIndexes: number[], format: string, outputDir = "") {
  const normalizedOutputDir = normalizeFilesystemPath(outputDir);
  return request<ExportResult>(`/api/projects/${projectId}/exports`, {
    method: "POST",
    body: JSON.stringify({
      episode_indexes: episodeIndexes,
      format,
      options: normalizedOutputDir ? { output_dir: normalizedOutputDir } : {},
    }),
  });
}

export function runCleaning(
  projectId: string,
  vlm?: VlmSettings,
  episodeIndexes?: number[],
  ruleConfig?: CleaningRuleConfig,
) {
  return request<CleaningRun>(`/api/projects/${projectId}/cleaning/runs`, {
    method: "POST",
    body: JSON.stringify({
      pass_threshold: 0.8,
      review_threshold: 0.6,
      ...(ruleConfig ?? {}),
      ...(vlm ? { vlm: writableVlmSettings(vlm) } : {}),
      ...(episodeIndexes ? { episode_indexes: episodeIndexes } : {}),
    }),
  });
}

export function saveVlmSettings(projectId: string, settings: VlmSettings) {
  return request<VlmSettings>(`/api/projects/${projectId}/vlm-settings`, {
    method: "PATCH",
    body: JSON.stringify(writableVlmSettings(settings)),
  });
}

export function getVlmSettings(projectId: string) {
  return request<VlmSettings>(`/api/projects/${projectId}/vlm-settings`);
}

function writableVlmSettings(settings: VlmSettings) {
  const { api_key_configured: _apiKeyConfigured, ...writable } = settings;
  return writable;
}

export function updateEpisodeDecision(
  projectId: string,
  episodeIndex: number,
  status: Exclude<CleaningStatus, "unscored">,
) {
  return request<EpisodeQualityResult>(
    `/api/projects/${projectId}/episodes/${episodeIndex}/decision`,
    {
      method: "PATCH",
      body: JSON.stringify({ status }),
    },
  );
}

export function runFilters(projectId: string) {
  return request<FilterRun>(`/api/projects/${projectId}/filters/runs`, { method: "POST" });
}

export function getFilterDetail(projectId: string, stageId: FilterStageId, episodeIndex: number) {
  return request<FilterDetail>(`/api/projects/${projectId}/filters/${stageId}/episodes/${episodeIndex}`);
}

export function saveFilterConfig(projectId: string, config: Partial<FilterConfig>) {
  return request<FilterConfig>(`/api/projects/${projectId}/filters/config`, {
    method: "PATCH",
    body: JSON.stringify(config),
  });
}

export async function uploadKinematicsUrdf(projectId: string, file: File) {
  const response = await fetch(
    `/api/projects/${projectId}/filters/kinematics/urdf?filename=${encodeURIComponent(file.name)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: await file.arrayBuffer(),
    },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<{ filename: string; path: string }>;
}
