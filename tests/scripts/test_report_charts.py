from __future__ import annotations

import pytest


def test_finds_named_gripper_dimensions() -> None:
    from scripts.report_charts import find_named_dimensions

    features = {
        "action": {
            "names": {
                "motors": [
                    "left_waist",
                    "left_gripper",
                    "right_waist",
                    "right_gripper",
                ]
            }
        }
    }

    dimensions = find_named_dimensions(features, "action", "gripper")

    assert dimensions == [(1, "left_gripper"), (3, "right_gripper")]


def test_episode_duration_rows_are_ordered() -> None:
    from scripts.report_charts import episode_duration_rows

    class Episode:
        def __init__(self, episode_index: int, duration_seconds: float) -> None:
            self.episode_index = episode_index
            self.duration_seconds = duration_seconds

    rows = episode_duration_rows([Episode(2, 1.25), Episode(0, 0.5), Episode(1, 1.0)])

    assert rows == [(0, pytest.approx(0.5)), (1, pytest.approx(1.0)), (2, pytest.approx(1.25))]
