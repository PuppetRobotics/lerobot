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
"""Launch SimHub simulation evaluations on freshly written training checkpoints."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lerobot.configs import SimHubConfig
from lerobot.utils.import_utils import _simhub_available

logger = logging.getLogger(__name__)

# gcsfuse mount root used by Vertex AI / GKE (`/gcs/<bucket>/<object path>`).
GCSFUSE_MOUNT_ROOT = Path("/gcs")


def resolve_checkpoint_uri(checkpoint_dir: Path, output_dir: Path, uri_prefix: str | None) -> str:
    """Map a local checkpoint directory to the remote URI SimHub will pull it from."""
    checkpoint_dir = checkpoint_dir.absolute()
    if uri_prefix:
        relative = checkpoint_dir.relative_to(output_dir.absolute())
        return f"{uri_prefix.rstrip('/')}/{relative.as_posix()}"
    if checkpoint_dir.is_relative_to(GCSFUSE_MOUNT_ROOT):
        return f"gs://{checkpoint_dir.relative_to(GCSFUSE_MOUNT_ROOT).as_posix()}"
    raise ValueError(
        f"Cannot derive a remote URI for checkpoint {checkpoint_dir}: output_dir is not on a gcsfuse "
        f"mount ({GCSFUSE_MOUNT_ROOT}/<bucket>/...). Set --simhub.checkpoint_uri_prefix to the remote "
        "location mirroring output_dir."
    )


class SimHubLauncher:
    """Fire-and-forget SimHub launches, run off the training thread.

    Launches go through a single background worker so a slow SimHub API never stalls rank 0 while the
    other ranks wait at the post-checkpoint barrier. Failures are logged, never raised: a broken
    evaluation service must not kill a training run.
    """

    def __init__(self, cfg: SimHubConfig, output_dir: Path):
        if not _simhub_available:
            raise ImportError("simhub.enable=true requires the 'simhub' package to be installed.")
        for env_var in (cfg.api_url_env, cfg.api_key_env):
            if not os.environ.get(env_var):
                raise ValueError(f"simhub.enable=true requires the {env_var} environment variable.")
        # Fail fast at startup rather than at the first checkpoint.
        resolve_checkpoint_uri(output_dir / "checkpoints", output_dir, cfg.checkpoint_uri_prefix)

        self.cfg = cfg
        self.output_dir = output_dir
        self.run_id = cfg.run_id or output_dir.absolute().name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simhub")

    def launch(self, checkpoint_dir: Path) -> None:
        """Queue one SimHub launch per configured seed for `checkpoint_dir`."""
        checkpoint_uri = resolve_checkpoint_uri(
            checkpoint_dir, self.output_dir, self.cfg.checkpoint_uri_prefix
        )
        self._executor.submit(self._launch, checkpoint_uri, checkpoint_dir.name)

    def _launch(self, checkpoint_uri: str, step_id: str) -> None:
        import simhub

        try:
            with simhub.connect(
                url=os.environ[self.cfg.api_url_env], key=os.environ[self.cfg.api_key_env]
            ) as sh:
                for seed in self.cfg.seeds:
                    batch = sh.launch(
                        self.cfg.evaluation,
                        checkpoint=checkpoint_uri,
                        seed=seed,
                        # Idempotency key: re-launching the same checkpoint/seed (e.g. on resume) is a no-op.
                        request_id=f"{self.run_id}-{self.cfg.evaluation}-{step_id}-seed{seed}",
                    )
                    logger.info(
                        "SimHub '%s' launched for %s (seed=%d): batch=%s jobs=%s",
                        self.cfg.evaluation,
                        checkpoint_uri,
                        seed,
                        batch.id,
                        batch.launched["job_ids"],
                    )
        except Exception:
            logger.warning("SimHub launch failed for %s", checkpoint_uri, exc_info=True)

    def close(self) -> None:
        """Wait for queued launches (so the final checkpoint's launch is not dropped on exit)."""
        self._executor.shutdown(wait=True)
