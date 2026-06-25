from __future__ import annotations

from pathlib import Path

from scripts.qwen_filter_dataset import build_config, parse_int_list, parse_str_list


def test_parse_cli_lists() -> None:
    assert parse_int_list("1, 3,5") == [1, 3, 5]
    assert parse_int_list("") == []
    assert parse_str_list("joint1, joint2") == ["joint1", "joint2"]


def test_build_config_accepts_urdf_options(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='empty'/>", encoding="utf-8")

    config = build_config(
        gripper_indices="6,13",
        urdf=urdf,
        end_effector_link="tool0",
        joint_names="waist,shoulder",
        joint_state_indices="0,1",
        eef_position_indices="7,8,9",
        max_lag=4,
    )

    assert config.gripper_indices == [6, 13]
    assert config.extreme_values.gripper_indices == [6, 13]
    assert config.sudden_change.residual_min_threshold == 0.02
    assert config.sudden_change.acceleration_min_threshold == 0.02
    assert config.sudden_change.jerk_min_threshold == 0.02
    assert config.state_action_alignment.max_lag == 4
    assert config.kinematics.urdf_path == urdf
    assert config.kinematics.end_effector_link == "tool0"
    assert config.kinematics.joint_names == ["waist", "shoulder"]
    assert config.kinematics.joint_state_indices == [0, 1]
    assert config.kinematics.eef_position_indices == [7, 8, 9]
