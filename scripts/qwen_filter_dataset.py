from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from robot_data_studio.quality.qwen_filters import (
    ExtremeValueConfig,
    KinematicsConfig,
    QwenFilterConfig,
    QwenManipDataFilter,
    StateActionAlignmentConfig,
    SuddenChangeConfig,
)


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_config(
    *,
    gripper_indices: str | None = None,
    urdf: Path | None = None,
    end_effector_link: str | None = None,
    joint_names: str | None = None,
    joint_state_indices: str | None = None,
    eef_position_indices: str | None = None,
    max_lag: int = 8,
    sudden_min_threshold: float = 0.02,
) -> QwenFilterConfig:
    grippers = parse_int_list(gripper_indices)
    return QwenFilterConfig(
        gripper_indices=grippers,
        sudden_change=SuddenChangeConfig(
            residual_min_threshold=sudden_min_threshold,
            acceleration_min_threshold=sudden_min_threshold,
            jerk_min_threshold=sudden_min_threshold,
        ),
        state_action_alignment=StateActionAlignmentConfig(
            max_lag=max_lag,
            ignored_indices=grippers,
        ),
        extreme_values=ExtremeValueConfig(gripper_indices=grippers),
        kinematics=KinematicsConfig(
            urdf_path=urdf,
            end_effector_link=end_effector_link,
            joint_names=parse_str_list(joint_names),
            joint_state_indices=parse_int_list(joint_state_indices),
            eef_position_indices=parse_int_list(eef_position_indices),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Qwen-RobotManip S1-S5 state/action data filters on a LeRobot dataset."
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for a quick run.")
    parser.add_argument("--gripper-indices", default="", help="Comma-separated gripper dimensions to exempt.")
    parser.add_argument("--max-lag", type=int, default=8, help="Maximum lag for state-action alignment.")
    parser.add_argument(
        "--sudden-min-threshold",
        type=float,
        default=0.02,
        help="Minimum residual/acceleration/jerk threshold for S1.",
    )
    parser.add_argument("--urdf", type=Path, default=None, help="Optional URDF path for S4 FK checks.")
    parser.add_argument("--end-effector-link", default=None, help="URDF link name used as the EEF frame.")
    parser.add_argument("--joint-names", default="", help="Comma-separated URDF joint names in state order.")
    parser.add_argument(
        "--joint-state-indices",
        default="",
        help="Comma-separated observation.state indices matching --joint-names.",
    )
    parser.add_argument(
        "--eef-position-indices",
        default="",
        help="Comma-separated observation.state indices containing logged EEF xyz.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(
        gripper_indices=args.gripper_indices,
        urdf=args.urdf,
        end_effector_link=args.end_effector_link,
        joint_names=args.joint_names,
        joint_state_indices=args.joint_state_indices,
        eef_position_indices=args.eef_position_indices,
        max_lag=args.max_lag,
        sudden_min_threshold=args.sudden_min_threshold,
    )
    summary = QwenManipDataFilter(config).evaluate_lerobot_dataset(
        args.dataset,
        max_episodes=args.max_episodes,
    )
    payload = json.dumps(asdict(summary), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
