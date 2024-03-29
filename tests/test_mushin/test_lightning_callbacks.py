# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer

from rai_toolbox.mushin.lightning import MetricsCallback
from rai_toolbox.mushin.testing.lightning import SimpleLightningModule


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("testing", [True, False])
def test_metrics_callback(testing):
    trainer = Trainer(max_epochs=1, callbacks=[MetricsCallback()])
    module = SimpleLightningModule()

    if testing:
        trainer.test(module)
        metric_files = list(Path(".").glob("**/test_metrics.pt"))
    else:
        trainer.fit(module)
        metric_files = list(Path(".").glob("**/fit_metrics.pt"))

    assert len(metric_files) == 1
    metrics = torch.load(metric_files[0])
    assert isinstance(metrics, dict)

    if testing:
        assert "test_tensor_metric" in metrics
    else:
        assert "fit_tensor_metric" in metrics
        assert "val_tensor_metric" in metrics

    for k, v in metrics.items():
        assert not isinstance(v, torch.Tensor)
