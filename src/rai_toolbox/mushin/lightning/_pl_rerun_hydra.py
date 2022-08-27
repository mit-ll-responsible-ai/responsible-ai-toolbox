# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging

import cloudpickle
import hydra

log = logging.getLogger(__name__)


@hydra.main(config_path=None, version_base="1.1")
def main(cfg):
    import os

    with open(str(cfg.config), "rb") as input:
        config = cloudpickle.load(input)  # nosec

    with open(str(cfg.task_fn), "rb") as input:
        task_fn = cloudpickle.load(input)  # nosec

    pl_local_rank = os.getenv("LOCAL_RANK", None)
    log.info(f"Rank {pl_local_rank}: Launched subprocess for PyTorch Lightning")
    task_fn(config)


if __name__ == "__main__":  # pragma: no cover
    main()
