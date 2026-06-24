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

export type Episode = {
  episode_index: number;
  length: number;
  duration_seconds: number;
  tasks: string[];
  data_file: string;
  video_files: Record<string, string>;
  video_start_seconds: Record<string, number>;
  video_end_seconds: Record<string, number>;
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
  };
  scorer_version: string;
};

export type CleaningRun = {
  run_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  summary: CleaningSummary;
};

export type VlmSettings = {
  enabled: boolean;
  provider: string;
  model: string;
};

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

export function importDataset(path: string) {
  return request<Project>("/api/projects", {
    method: "POST",
    body: JSON.stringify({ path }),
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

export function exportEpisode(projectId: string, episodeIndex: number) {
  return request<{ output_path: string }>(`/api/projects/${projectId}/exports`, {
    method: "POST",
    body: JSON.stringify({ episode_indexes: [episodeIndex], format: "act_hdf5" }),
  });
}

export function runCleaning(projectId: string, vlm?: VlmSettings) {
  return request<CleaningRun>(`/api/projects/${projectId}/cleaning/runs`, {
    method: "POST",
    body: JSON.stringify({ pass_threshold: 0.8, review_threshold: 0.6, ...(vlm ? { vlm } : {}) }),
  });
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
