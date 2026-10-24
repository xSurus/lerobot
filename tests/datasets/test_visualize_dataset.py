#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import (
    build_blueprint_from_dataset,
    get_extra_scalar_keys,
    visualize_dataset,
)

EXTRA_KEYS = [
    "observation.environment_state",
    "complementary_info.is_intervention",
    "complementary_info.q_value",
]


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()


def make_scalar_dataset(root):
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": None},
        "complementary_info.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.q_value": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(repo_id="test/extra", root=root, fps=10, features=features)
    for i in range(3):
        dataset.add_frame(
            {
                "observation.state": np.full(2, i, dtype=np.float32),
                "observation.environment_state": np.full(3, i, dtype=np.float32),
                "complementary_info.is_intervention": np.array([i % 2 == 1]),
                "complementary_info.q_value": np.array([i], dtype=np.float32),
                "action": np.full(2, i, dtype=np.float32),
                "task": "t",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("test/extra", root=root, episodes=[0])


def test_get_extra_scalar_keys(tmp_path):
    dataset = make_scalar_dataset(tmp_path / "ds")
    assert get_extra_scalar_keys(dataset) == EXTRA_KEYS


def test_visualize_logs_extra_scalars(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    dataset = make_scalar_dataset(tmp_path / "ds")

    blueprint = build_blueprint_from_dataset(dataset)
    origins = {str(view.origin) for view in blueprint.root_container.contents[1].contents}
    assert set(EXTRA_KEYS) <= origins

    logged = set()
    monkeypatch.setattr(rr, "log", lambda path, *args, **kwargs: logged.add(path))
    visualize_dataset(dataset, episode_index=0, save=True, output_dir=tmp_path / "out", num_workers=0)
    assert set(EXTRA_KEYS) <= logged
