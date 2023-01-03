# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT


def test_configs():
    from configs import Config  # runs config validation for all configs

    assert Config  # touch for no unused imports


def test_solver():
    from solver import AdversarialTrainer  # tests imports

    assert AdversarialTrainer  # touch for no unused imports


def test_train():
    from train import task_fn  # tests imports

    assert task_fn  # touch for no unused imports
