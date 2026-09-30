#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.common import simhub_utils
from lerobot.common.simhub_utils import SimHubLauncher, resolve_checkpoint_uri
from lerobot.configs import SimHubConfig


def test_resolve_checkpoint_uri_with_prefix():
    uri = resolve_checkpoint_uri(
        Path("/tmp/run/checkpoints/037500"), Path("/tmp/run"), "gs://bucket/lerobot_outputs/run/"
    )
    assert uri == "gs://bucket/lerobot_outputs/run/checkpoints/037500"


def test_resolve_checkpoint_uri_from_gcsfuse_mount():
    uri = resolve_checkpoint_uri(
        Path("/gcs/bucket/lerobot_outputs/run/checkpoints/037500"),
        Path("/gcs/bucket/lerobot_outputs/run"),
        None,
    )
    assert uri == "gs://bucket/lerobot_outputs/run/checkpoints/037500"


def test_resolve_checkpoint_uri_unresolvable():
    with pytest.raises(ValueError, match="checkpoint_uri_prefix"):
        resolve_checkpoint_uri(Path("/tmp/run/checkpoints/037500"), Path("/tmp/run"), None)


def test_config_requires_evaluation():
    with pytest.raises(ValueError, match="simhub.evaluation"):
        SimHubConfig(enable=True)


@pytest.fixture
def fake_simhub(monkeypatch):
    client = MagicMock()
    client.launch.side_effect = lambda *args, seed, **kwargs: SimpleNamespace(
        id=f"batch-{seed}", launched={"job_ids": [f"job-{seed}"]}
    )
    module = MagicMock()
    module.connect.return_value.__enter__.return_value = client
    monkeypatch.setitem(sys.modules, "simhub", module)
    monkeypatch.setattr(simhub_utils, "_simhub_available", True)
    monkeypatch.setenv("SIMHUB_API_URL", "https://simhub.example.com")
    monkeypatch.setenv("SIMHUB_API_KEY", "secret")
    return module, client


def test_launcher_launches_every_seed(fake_simhub):
    module, client = fake_simhub
    cfg = SimHubConfig(
        enable=True,
        evaluation="puppet-golf-evaluation",
        seeds=[0, 1],
        checkpoint_uri_prefix="gs://bucket/run",
        run_id="myrun",
    )
    launcher = SimHubLauncher(cfg, Path("/tmp/run"))
    launcher.launch(Path("/tmp/run/checkpoints/037500"))
    launcher.close()

    module.connect.assert_called_once_with(url="https://simhub.example.com", key="secret")
    assert [call.kwargs for call in client.launch.call_args_list] == [
        {
            "checkpoint": "gs://bucket/run/checkpoints/037500",
            "seed": seed,
            "request_id": f"myrun-puppet-golf-evaluation-037500-seed{seed}",
        }
        for seed in (0, 1)
    ]
    assert all(call.args == ("puppet-golf-evaluation",) for call in client.launch.call_args_list)


def test_launcher_swallows_api_errors(fake_simhub):
    _, client = fake_simhub
    client.launch.side_effect = RuntimeError("SimHub is down")
    cfg = SimHubConfig(enable=True, evaluation="eval", checkpoint_uri_prefix="gs://bucket/run")
    launcher = SimHubLauncher(cfg, Path("/tmp/run"))
    launcher.launch(Path("/tmp/run/checkpoints/000010"))
    launcher.close()  # Does not raise.


@pytest.mark.parametrize("env_var", ["SIMHUB_API_URL", "SIMHUB_API_KEY"])
def test_launcher_requires_api_env_vars(fake_simhub, monkeypatch, env_var):
    monkeypatch.delenv(env_var)
    cfg = SimHubConfig(enable=True, evaluation="eval", checkpoint_uri_prefix="gs://bucket/run")
    with pytest.raises(ValueError, match=env_var):
        SimHubLauncher(cfg, Path("/tmp/run"))
