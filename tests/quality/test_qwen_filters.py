from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_data_studio.quality.qwen_filters import (
    ExtremeValueConfig,
    KinematicsConfig,
    OrientationAlignmentConfig,
    QwenFilterConfig,
    QwenManipDataFilter,
    StateActionAlignmentConfig,
    SuddenChangeConfig,
    apply_orientation_alignment,
    detect_extreme_values,
    detect_state_action_trend_alignment,
    detect_sudden_changes,
    evaluate_kinematic_consistency,
    rotation_matrix_from_rpy,
)


def test_s1_sudden_change_requires_residual_and_high_order_motion() -> None:
    trajectory = np.zeros((21, 1), dtype=np.float64)
    trajectory[10, 0] = 10.0

    result = detect_sudden_changes(
        trajectory,
        SuddenChangeConfig(residual_scale=3.0, acceleration_scale=3.0, jerk_scale=3.0),
    )

    assert 10 in result.flagged_frames
    assert result.frame_mask[10]
    assert result.dimension_scores[0].max_residual > 5


def test_s2_state_action_alignment_accepts_action_leading_state_and_rejects_state_leading() -> None:
    action = np.asarray([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [4.0]], dtype=np.float64)
    state = np.asarray([[0.0], [0.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)

    aligned = detect_state_action_trend_alignment(
        state,
        action,
        StateActionAlignmentConfig(max_lag=2, directional_agreement_threshold=0.6),
    )
    reversed_streams = detect_state_action_trend_alignment(
        action,
        state,
        StateActionAlignmentConfig(max_lag=2, directional_agreement_threshold=0.6),
    )

    assert aligned.dimension_results[0].lag == 1
    assert aligned.dimension_results[0].directional_agreement >= 0.6
    assert not aligned.flagged_dimensions
    assert reversed_streams.dimension_results[0].lag < 0
    assert reversed_streams.flagged_dimensions == [0]


def test_s2_state_action_alignment_integrates_delta_actions_before_comparison() -> None:
    state = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    delta_action = np.asarray([[0.0], [1.0], [1.0], [1.0]], dtype=np.float64)

    result = detect_state_action_trend_alignment(
        state,
        delta_action,
        StateActionAlignmentConfig(max_lag=1, actions_are_delta=True),
    )

    assert result.dimension_results[0].lag == 0
    assert result.dimension_results[0].directional_agreement == 1.0
    assert not result.flagged_dimensions


def test_s3_extreme_value_detection_uses_expanded_quantile_band_and_exempts_gripper_dims() -> None:
    values = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 100.0],
            [200.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )

    result = detect_extreme_values(
        {"state": values},
        ExtremeValueConfig(alpha=0.0, lower_quantile=0.05, upper_quantile=0.95, gripper_indices=[1]),
    )

    assert 3 in result.flagged_frames
    assert 2 not in result.flagged_frames
    assert result.dimension_results["state"][0].low < result.dimension_results["state"][0].high


def test_s4_kinematic_consistency_loads_urdf_and_resolves_constant_tcp_offset(tmp_path: Path) -> None:
    urdf = tmp_path / "two_link.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base"/>
  <link name="link1"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint2" type="fixed">
    <parent link="link1"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    joint_states = np.asarray([[0.0], [np.pi / 2]], dtype=np.float64)
    logged_eef = np.asarray([[1.1, 0.2, 0.0], [0.1, 1.2, 0.0]], dtype=np.float64)

    result = evaluate_kinematic_consistency(
        joint_states,
        logged_eef,
        KinematicsConfig(
            urdf_path=urdf,
            end_effector_link="tool",
            joint_names=["joint1"],
            position_tolerance=1e-6,
            resolve_tcp_offset=True,
        ),
    )

    assert result.loaded_urdf == urdf
    assert np.allclose(result.tcp_offset, [0.1, 0.2, 0.0])
    assert result.max_position_error <= 1e-6
    assert not result.flagged_frames


def test_s5_orientation_alignment_applies_dataset_rotation_to_positions_and_6d_orientation() -> None:
    correction = rotation_matrix_from_rpy(0.0, 0.0, np.pi / 2)
    values = np.asarray([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)

    aligned = apply_orientation_alignment(
        values,
        OrientationAlignmentConfig(
            rotation_correction=correction,
            position_slice=slice(0, 3),
            orientation_slice=slice(3, 9),
            orientation_representation="rotation_6d",
        ),
    )

    assert np.allclose(aligned[0, :3], [0.0, 1.0, 0.0], atol=1e-7)
    assert np.allclose(aligned[0, 3:6], [0.0, 1.0, 0.0], atol=1e-7)
    assert np.allclose(aligned[0, 6:9], [-1.0, 0.0, 0.0], atol=1e-7)


def test_qwen_filter_can_process_aloha_static_coffee_sample() -> None:
    sample = Path("/Users/peterxie/Desktop/data platform /data/samples/aloha_static_coffee")
    if not sample.exists():
        return

    summary = QwenManipDataFilter(
        QwenFilterConfig(
            gripper_indices=[6, 13],
            sudden_change=SuddenChangeConfig(residual_scale=12.0, acceleration_scale=12.0, jerk_scale=12.0),
            state_action_alignment=StateActionAlignmentConfig(max_lag=8),
            extreme_values=ExtremeValueConfig(alpha=0.5, gripper_indices=[6, 13]),
        )
    ).evaluate_lerobot_dataset(sample, max_episodes=2)

    assert summary.total_episodes == 2
    assert summary.total_frames > 0
    assert len(summary.episode_results) == 2
    assert {"s1_sudden_change", "s2_state_action_alignment", "s3_extreme_value"} <= set(
        summary.stage_counts
    )
