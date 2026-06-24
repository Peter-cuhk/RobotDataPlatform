from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from robot_data_studio.projects.models import CleaningRunRequest, CreateProjectRequest, ExportRequest, Project
from robot_data_studio.projects.service import ProjectService
from robot_data_studio.quality import CleaningRun, CleaningSummary, EpisodeDecisionRequest, EpisodeQualityResult
from robot_data_studio.viewer import create_episode_recording


def create_app(artifact_root: str | Path = ".rds-artifacts") -> FastAPI:
    app = FastAPI(title="Robot Data Studio API")
    service = ProjectService(Path(artifact_root))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/projects", status_code=201, response_model=Project)
    def create_project(request: CreateProjectRequest) -> Project:
        try:
            return service.create(request.path)
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/projects/{project_id}", response_model=Project)
    def get_project(project_id: str) -> Project:
        try:
            return service.project(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/projects/{project_id}/episodes")
    def episodes(project_id: str, limit: int | None = Query(default=100, ge=1, le=1000)):
        try:
            return service.reader(project_id).list_episodes(limit=limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/projects/{project_id}/episodes/{episode_index}/frames")
    def frames(project_id: str, episode_index: int):
        try:
            return service.reader(project_id).read_episode_frames(episode_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post(
        "/api/projects/{project_id}/episodes/{episode_index}/recording",
        status_code=201,
    )
    def recording(project_id: str, episode_index: int) -> dict[str, str]:
        try:
            filename = f"{project_id}-episode-{episode_index:06d}.rrd"
            output = service.artifact_root / filename
            create_episode_recording(service.reader(project_id), episode_index, output)
            return {"recording_url": f"/api/artifacts/{filename}"}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/projects/{project_id}/exports", status_code=201)
    def export(project_id: str, request: ExportRequest) -> dict[str, str]:
        if request.format != "act_hdf5":
            raise HTTPException(status_code=400, detail="Only act_hdf5 is supported")
        if len(request.episode_indexes) != 1:
            raise HTTPException(status_code=400, detail="Export exactly one episode")
        try:
            output = service.export_act(project_id, request.episode_indexes[0])
            return {"output_path": str(output)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post(
        "/api/projects/{project_id}/cleaning/runs",
        status_code=201,
        response_model=CleaningRun,
    )
    def run_cleaning(project_id: str, request: CleaningRunRequest) -> CleaningRun:
        try:
            return service.run_cleaning(project_id, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/projects/{project_id}/cleaning", response_model=CleaningSummary)
    def cleaning(project_id: str) -> CleaningSummary:
        try:
            return service.cleaning_summary(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.patch(
        "/api/projects/{project_id}/episodes/{episode_index}/decision",
        response_model=EpisodeQualityResult,
    )
    def decision(
        project_id: str,
        episode_index: int,
        request: EpisodeDecisionRequest,
    ) -> EpisodeQualityResult:
        try:
            return service.update_episode_decision(project_id, episode_index, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/artifacts/{filename}")
    def artifact(filename: str) -> FileResponse:
        path = service.artifact_root / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path)

    return app


app = create_app()
