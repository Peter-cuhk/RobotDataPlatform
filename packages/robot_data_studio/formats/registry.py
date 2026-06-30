from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from robot_data_studio.formats.hdf5 import (
    ActHDF5DatasetAdapter,
    RobomimicHDF5DatasetAdapter,
    write_act_hdf5,
    write_robomimic_hdf5,
)
from robot_data_studio.formats.models import DatasetAdapter, ExportResult, FormatInfo, conversion_report
from robot_data_studio.formats.umi_zarr import UMIZarrDatasetAdapter, write_umi_zarr
from robot_data_studio.lerobot.reader import LeRobotDatasetReader
from robot_data_studio.quality.metadata_completeness import build_metadata_inspection_report


class UnsupportedDatasetFormat(ValueError):
    pass


class FormatRegistry:
    def __init__(self) -> None:
        self._readers: dict[str, type] = {
            "lerobot_v2_1": LeRobotDatasetReader,
            "lerobot_v3": LeRobotDatasetReader,
            "lerobot": LeRobotDatasetReader,
            "act_hdf5": ActHDF5DatasetAdapter,
            "robomimic_hdf5": RobomimicHDF5DatasetAdapter,
            "umi_zarr": UMIZarrDatasetAdapter,
        }
        self._probe_order = [
            ("lerobot_v3", LeRobotDatasetReader),
            ("act_hdf5", ActHDF5DatasetAdapter),
            ("robomimic_hdf5", RobomimicHDF5DatasetAdapter),
            ("umi_zarr", UMIZarrDatasetAdapter),
        ]

    @classmethod
    def default(cls) -> "FormatRegistry":
        return cls()

    def list_formats(self) -> list[FormatInfo]:
        return [
            FormatInfo(id="lerobot_v2_1", label="LeRobot v2.1", profile="lerobot", can_import=True, can_export=True),
            FormatInfo(id="lerobot_v3", label="LeRobot v3", profile="lerobot", can_import=True, can_export=True),
            FormatInfo(id="act_hdf5", label="ACT HDF5", profile="hdf5", can_import=True, can_export=True),
            FormatInfo(id="robomimic_hdf5", label="robomimic HDF5", profile="hdf5", can_import=True, can_export=True),
            FormatInfo(id="umi_zarr", label="UMI Zarr", profile="zarr", can_import=True, can_export=True),
        ]

    def open_dataset(self, path: str | Path, format_hint: str | None = None) -> DatasetAdapter:
        if format_hint and format_hint != "auto":
            reader = self._readers.get(format_hint)
            if reader is None:
                raise UnsupportedDatasetFormat(f"Unsupported dataset format: {format_hint}")
            return reader(path)
        errors = []
        for _, reader in self._probe_order:
            try:
                if reader.probe(path):
                    return reader(path)
            except Exception as error:
                errors.append(str(error))
        detail = "; ".join(errors[-3:])
        raise UnsupportedDatasetFormat(f"Could not detect dataset format for {path}. {detail}".strip())

    def export_dataset(
        self,
        *,
        adapter: DatasetAdapter,
        target_format: str,
        episode_indexes: list[int],
        output_root: Path,
    ) -> ExportResult:
        if not episode_indexes:
            raise UnsupportedDatasetFormat("Export at least one episode")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if target_format == "act_hdf5":
            output_path = write_act_hdf5(adapter, episode_indexes, output_root)
            mapping = {"observation.state": "observations/qpos", "action": "action"}
        elif target_format == "robomimic_hdf5":
            output_path = write_robomimic_hdf5(adapter, episode_indexes, output_root)
            mapping = {"observation.state": "data/demo_*/obs/qpos", "action": "data/demo_*/actions"}
        elif target_format == "umi_zarr":
            output_path = write_umi_zarr(adapter, episode_indexes, output_root)
            mapping = {"observation.state": "data/robot0_eef_pos", "action": "data/action"}
        elif target_format in {"lerobot_v3", "lerobot_v2_1"}:
            output_path = self._write_lerobot_placeholder(adapter, episode_indexes, output_root, target_format)
            mapping = {"observation.state": "observation.state", "action": "action"}
        else:
            raise UnsupportedDatasetFormat(f"Unsupported export format: {target_format}")
        report_path = output_path if output_path.is_dir() else output_path.parent
        report_path = report_path / "conversion_report.json"
        backend = {
            "lerobot_available": importlib.util.find_spec("lerobot") is not None,
            "forge_available": importlib.util.find_spec("forge") is not None,
            "any4lerobot_available": importlib.util.find_spec("any4lerobot") is not None,
        }
        if target_format in {"lerobot_v3", "lerobot_v2_1"}:
            backend["writer_backend"] = (
                "jsonl_fallback"
                if output_path.is_dir() and (output_path / "frames.jsonl").is_file()
                else "lerobot"
            )
        report = conversion_report(
            source=adapter.metadata(),
            target_format=target_format,
            episode_indexes=episode_indexes,
            output_path=output_path,
            field_mapping=mapping,
            backend=backend,
            metadata_completeness=build_metadata_inspection_report(adapter),
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return ExportResult(
            output_path=output_path,
            report_path=report_path,
            format=target_format,
            episode_count=len(episode_indexes),
        )

    def _write_lerobot_placeholder(
        self,
        adapter: DatasetAdapter,
        episode_indexes: list[int],
        output_root: Path,
        target_format: str,
    ) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        # Prefer the official LeRobot writer when installed. The lightweight JSONL fallback keeps
        # local tests usable in environments where torch/lerobot are intentionally absent.
        if importlib.util.find_spec("lerobot") is not None:
            try:
                from robot_data_studio.formats.lerobot_writer import write_lerobot_dataset

                return write_lerobot_dataset(adapter, episode_indexes, output_root, target_format)
            except Exception:
                pass
        data_path = output_root / "frames.jsonl"
        with data_path.open("w", encoding="utf-8") as file:
            for episode_index in episode_indexes:
                for frame in adapter.read_episode_frames(episode_index):
                    file.write(
                        json.dumps(
                            {
                                "episode_index": episode_index,
                                "frame_index": frame.frame_index,
                                "timestamp": frame.timestamp,
                                "observation.state": frame.observation_state,
                                "action": frame.action,
                            }
                        )
                        + "\n"
                    )
        meta = output_root / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0" if target_format == "lerobot_v3" else "v2.1",
                    "source_format": adapter.metadata().format,
                    "total_episodes": len(episode_indexes),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_root
