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
"""Prompt loading and ``prompts_dir`` variant selection."""

from __future__ import annotations

from pathlib import Path

import pytest

# ``lerobot.annotations.steerable_pipeline`` imports its writer, which pulls in
# ``lerobot.datasets`` (-> the HF ``datasets`` library); skip without it.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline import prompts  # noqa: E402
from lerobot.annotations.steerable_pipeline.config import PlanConfig  # noqa: E402

# Placeholders each plan-module call site formats in; a variant must keep them.
_FORMAT_ARGS = {
    "plan_subtasks": {
        "episode_task": "Play golf",
        "observation_block": "",
        "min_subtask_seconds": 0.5,
        "max_steps": 8,
        "episode_duration": "20.000",
    },
    "plan_subtask_describe": {"episode_task": "Play golf"},
}


def test_load_without_prompts_dir_reads_packaged_default() -> None:
    expected = (Path(prompts.__file__).parent / "plan_subtasks.txt").read_text(encoding="utf-8")
    assert prompts.load("plan_subtasks") == expected
    assert prompts.load("plan_subtasks", None) == expected
    assert prompts.load("plan_subtasks", "") == expected


def test_prompts_dir_overrides_present_files_and_falls_back_for_others(tmp_path: Path) -> None:
    (tmp_path / "plan_subtasks.txt").write_text("CUSTOM {episode_task}", encoding="utf-8")

    assert prompts.load("plan_subtasks", tmp_path) == "CUSTOM {episode_task}"
    assert prompts.load("plan_memory", tmp_path) == prompts.load("plan_memory")
    assert prompts.overridden_prompts(tmp_path) == ["plan_subtasks"]


def test_packaged_variant_resolves_by_name() -> None:
    resolved = prompts.resolve_prompts_dir("golf")
    assert resolved == Path(prompts.__file__).parent / "golf"
    assert "golf" in prompts.builtin_variants()
    assert prompts.overridden_prompts("golf") == ["plan_subtask_describe", "plan_subtasks"]


@pytest.mark.parametrize("name", sorted(_FORMAT_ARGS))
def test_golf_variant_differs_from_default_and_formats(name: str) -> None:
    golf = prompts.load(name, "golf")
    assert golf != prompts.load(name)
    assert "GOLF" in golf
    golf.format(**_FORMAT_ARGS[name])  # raises on a broken placeholder or stray brace


def test_unknown_prompts_dir_raises() -> None:
    with pytest.raises(FileNotFoundError, match="packaged variants"):
        prompts.resolve_prompts_dir("no-such-variant")
    with pytest.raises(FileNotFoundError):
        PlanConfig(prompts_dir="no-such-variant")


def test_misspelled_template_name_raises(tmp_path: Path) -> None:
    (tmp_path / "plan_subtask.txt").write_text("typo", encoding="utf-8")
    with pytest.raises(ValueError, match="plan_subtask.txt"):
        prompts.resolve_prompts_dir(tmp_path)


def test_env_override_wins_over_prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "plan_subtasks.txt").write_text("FROM DIR", encoding="utf-8")
    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", "FROM ENV")
    assert prompts.load("plan_subtasks", tmp_path) == "FROM ENV"
