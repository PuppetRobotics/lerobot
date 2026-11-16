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
"""Prompt templates loaded as plain text.

One file per use site. Templates use ``str.format(**vars)`` substitution; we
intentionally avoid jinja2 here so the templates remain inspectable in
plain editors and roundtrip cleanly through ``ruff format``.

Subdirectories (e.g. ``golf/``) hold prompt variants. A variant directory only
needs the files it changes; :func:`load` falls back to the default template
for every other prompt. Select one with ``prompts_dir`` (``--plan.prompts_dir``
on the CLI), either by the subdirectory name or by a path to any directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_DIR = Path(__file__).parent


def default_prompt_names() -> list[str]:
    """Names of the packaged default templates (``<name>.txt`` in ``prompts/``)."""
    return sorted(p.stem for p in _DIR.glob("*.txt"))


def builtin_variants() -> list[str]:
    """Subdirectories of ``prompts/`` that contain at least one template."""
    return sorted(p.name for p in _DIR.iterdir() if p.is_dir() and any(p.glob("*.txt")))


def resolve_prompts_dir(prompts_dir: str | Path | None) -> Path | None:
    """Resolve and validate a prompt variant directory.

    ``prompts_dir`` is either a path to a directory or the name of a packaged
    variant (a subdirectory of ``prompts/``, e.g. ``"golf"``). An existing path
    wins over a packaged variant of the same name. Returns ``None`` when
    ``prompts_dir`` is unset or empty.

    Raises ``FileNotFoundError`` when the directory does not exist and
    ``ValueError`` when it contains a ``.txt`` file that matches no default
    template, so a misspelled file name cannot silently fall back to the
    default prompt.
    """
    if prompts_dir is None or not str(prompts_dir).strip():
        return None
    path = Path(prompts_dir).expanduser()
    if not path.is_dir():
        builtin = _DIR / path
        if path.is_absolute() or not builtin.is_dir():
            raise FileNotFoundError(
                f"prompts_dir {str(prompts_dir)!r} is neither a directory nor a packaged variant; "
                f"packaged variants: {builtin_variants()}"
            )
        path = builtin
    known = set(default_prompt_names())
    unknown = sorted(p.name for p in path.glob("*.txt") if p.stem not in known)
    if unknown:
        raise ValueError(
            f"prompts_dir {str(path)!r} contains templates that match no default prompt: {unknown}; "
            f"expected names: {sorted(known)}"
        )
    return path


def overridden_prompts(prompts_dir: str | Path | None) -> list[str]:
    """Names of the templates that ``prompts_dir`` replaces (empty when unset)."""
    path = resolve_prompts_dir(prompts_dir)
    if path is None:
        return []
    return sorted(p.stem for p in path.glob("*.txt"))


def load(name: str, prompts_dir: str | Path | None = None) -> str:
    """Read prompt template ``name.txt``.

    Lookup order:

    1. A ``LEROBOT_PROMPT_OVERRIDE_<name>`` environment variable, when set to
       a non-empty value. This lets prompt search (e.g. GEPA) inject candidate
       templates into a remote job without rebuilding the package.
    2. ``<prompts_dir>/<name>.txt``, when ``prompts_dir`` is set and the file
       exists (see :func:`resolve_prompts_dir`).
    3. The packaged default ``prompts/<name>.txt``.

    Every override must keep the same ``{placeholder}`` fields the call site
    formats in.
    """
    override = os.environ.get(f"LEROBOT_PROMPT_OVERRIDE_{name}")
    if override and override.strip():
        return override
    variant_dir = resolve_prompts_dir(prompts_dir)
    if variant_dir is not None:
        candidate = variant_dir / f"{name}.txt"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    path = _DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
