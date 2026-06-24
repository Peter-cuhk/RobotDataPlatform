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

