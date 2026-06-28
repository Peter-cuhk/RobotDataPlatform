import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from robot_data_studio.formats import UnsupportedDatasetFormat
from robot_data_studio.projects.models import CleaningRunRequest, CreateProjectRequest, ExportRequest, Project
from robot_data_studio.projects.service import ProjectService
from robot_data_studio.quality import (
    CleaningRun,
    CleaningSummary,
    EpisodeDecisionRequest,
    EpisodeQualityResult,
    FilterConfig,
    FilterConfigPatch,
    FilterDetail,
    FilterRun,
    FilterSummary,
    VlmSettings,
)
from robot_data_studio.viewer import create_episode_recording


def create_app(artifact_root: str | Path = ".rds-artifacts") -> FastAPI:
    app = FastAPI(title="Robot Data Studio API")
    service = ProjectService(Path(artifact_root))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/formats")
    def formats():
        return service.formats()

    @app.post("/api/projects", status_code=201, response_model=Project)
    def create_project(request: CreateProjectRequest) -> Project:
        try:
            return service.create(request.path, request.format_hint)
        except (UnsupportedDatasetFormat, ValueError, OSError) as error:
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

    @app.get(
        "/api/projects/{project_id}/episodes/{episode_index}/visual-quality/frame",
        response_class=Response,
    )
    def visual_quality_frame(
        project_id: str,
        episode_index: int,
        camera: str = Query(min_length=1),
        frame: int = Query(ge=0),
        width: int = Query(default=640, ge=160, le=1600),
    ) -> Response:
        try:
            content = service.visual_quality_frame(
                project_id,
                episode_index,
                camera,
                frame,
                width,
            )
            return Response(
                content=content,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=3600"},
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

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
    def export(project_id: str, request: ExportRequest) -> dict[str, str | int]:
        try:
            output = service.export_dataset(
                project_id,
                request.format,
                request.episode_indexes,
                request.options.output_dir,
            )
            return {
                "output_path": str(output.output_path),
                "report_path": str(output.report_path),
                "format": output.format,
                "episode_count": output.episode_count,
            }
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (UnsupportedDatasetFormat, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

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

    @app.post(
        "/api/projects/{project_id}/filters/runs",
        status_code=201,
        response_model=FilterRun,
    )
    def run_filters(project_id: str) -> FilterRun:
        try:
            return service.run_filters(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/projects/{project_id}/pipeline/runs/stream")
    async def stream_pipeline(project_id: str, request: CleaningRunRequest) -> StreamingResponse:
        try:
            service.project(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def event_generator():
            queue: asyncio.Queue[dict] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def emit(payload: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            def cleaning_progress(completed: int, total: int) -> None:
                emit({"event": "progress", "phase": "cleaning", "completed": completed, "total": total})

            def filters_progress(completed: int, total: int) -> None:
                emit({"event": "progress", "phase": "filters", "completed": completed, "total": total})

            async def run_work() -> None:
                try:
                    cleaning, filters = await asyncio.to_thread(
                        service.run_pipeline,
                        project_id,
                        request,
                        cleaning_progress,
                        filters_progress,
                    )
                    emit(
                        {
                            "event": "done",
                            "cleaning": cleaning.model_dump(mode="json"),
                            "filters": filters.model_dump(mode="json"),
                        }
                    )
                except Exception as error:  # noqa: BLE001 - surfaced to client via SSE
                    emit({"event": "error", "message": str(error)})

            task = asyncio.create_task(run_work())
            try:
                while True:
                    payload = await queue.get()
                    event_name = payload.get("event", "message")
                    data = json.dumps(payload, ensure_ascii=False)
                    yield f"event: {event_name}\ndata: {data}\n\n"
                    if event_name in ("done", "error"):
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/api/projects/{project_id}/filters", response_model=FilterSummary)
    def filters(project_id: str) -> FilterSummary:
        try:
            return service.filter_summary(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get(
        "/api/projects/{project_id}/filters/{stage_id}/episodes/{episode_index}",
        response_model=FilterDetail,
    )
    def filter_detail(project_id: str, stage_id: str, episode_index: int) -> FilterDetail:
        try:
            return service.filter_detail(project_id, stage_id, episode_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.patch("/api/projects/{project_id}/filters/config", response_model=FilterConfig)
    def update_filter_config(project_id: str, request: FilterConfigPatch) -> FilterConfig:
        try:
            return service.update_filter_config(project_id, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/projects/{project_id}/filters/kinematics/urdf", status_code=201)
    async def upload_filter_urdf(
        project_id: str,
        request: Request,
        filename: str = Query(default="robot.urdf"),
    ) -> dict[str, str]:
        try:
            return service.save_filter_urdf(project_id, filename, await request.body())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/projects/{project_id}/vlm-settings", response_model=VlmSettings)
    def vlm_settings(project_id: str) -> VlmSettings:
        try:
            return service.vlm_settings(project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.patch("/api/projects/{project_id}/vlm-settings", response_model=VlmSettings)
    def update_vlm_settings(project_id: str, request: VlmSettings) -> VlmSettings:
        try:
            return service.update_vlm_settings(project_id, request)
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
