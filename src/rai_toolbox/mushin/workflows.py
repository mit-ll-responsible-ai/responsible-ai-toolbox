# Copyright 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from typing import Any, Optional, Sequence, Union

from hydra_zen import make_config

from .hydra import launch, zen


def plot_robustness_curve(
    *,
    metrics: Sequence[Any],
    epsilons=Sequence[float],
    ax=None,
    label=None,
    save_fig: bool = True,
) -> None:
    from matplotlib import pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    multiplot = False

    if metrics and isinstance(metrics[0], Sequence):
        multiplot = True

    if multiplot:
        nplots = len(metrics[0])
        for i in range(nplots):
            ax.plot(epsilons, [y[i] for y in metrics], label=label)
            ax.set_xlabel("Perturbation Size")
            ax.set_ylabel(f"Metric {i}")
            plt.savefig(f"robustness_curve_{i}.pdf")
    else:
        ax.plot(epsilons, metrics, label=label)
        ax.set_xlabel("Perturbation Size")
        ax.set_ylabel("Test Accuracy")
        if save_fig:
            plt.savefig("robustness_curve.pdf")
    return ax


# A special "Task Container" class
class RobustnessCurve:
    EPSILON_NAME: str = "epsilon"

    def __init__(self, eval_task_cfg=None) -> None:
        # we can do validation checks here
        self.eval_task_cfg = (
            eval_task_cfg if eval_task_cfg is not None else make_config()
        )
        self.validate()

    @staticmethod
    def evaluation_task(eval_task_cfg) -> Any:
        raise NotImplementedError()

    def validate(self):
        zen(self.evaluation_task).validate(
            self.eval_task_cfg, excluded_params=(self.EPSILON_NAME,)
        )

    def to_xarray(self):
        import xarray as xr

        return xr.DataArray(
            self.metrics,
            dims=("epsilon",),
            coords={"epsilon": self.epsilons},
            name="accuracy",
        )

    def plot(self, **kwargs) -> None:
        # TODO: sort metrics by epsilon
        f = zen(plot_robustness_curve)
        return f(
            self.eval_task_cfg, metrics=self.metrics, epsilons=self.epsilons, **kwargs
        )

    def run(
        self,
        *,
        job_epsilons: Union[str, Sequence[float], range],
        sweepdir: str = ".",
        launcher: Optional[str] = None,
        additional_overrides: Optional[Sequence[str]] = None,
    ):
        if not isinstance(job_epsilons, str):
            eps_arg = ",".join(str(i) for i in job_epsilons)
        elif isinstance(job_epsilons, range):
            eps_arg = repr(job_epsilons)
        else:
            eps_arg = job_epsilons

        prefix = "+" if not hasattr(self.eval_task_cfg, self.EPSILON_NAME) else ""

        overrides = [
            f"hydra.sweep.dir={sweepdir}",
            prefix + f"{self.EPSILON_NAME}={eps_arg}",
        ]

        if launcher:
            overrides.append(f"hydra/launcher={launcher}")

        if additional_overrides:
            overrides.extend(additional_overrides)

        # Run a Multirun over epsilons
        (jobs,) = launch(
            self.eval_task_cfg,
            zen(self.evaluation_task),
            overrides=overrides,
            multirun=True,
            config_name="robustness_task",
            job_name="robustness_task",
        )
        self.jobs = jobs
        self.jobs_post_process()

    def jobs_post_process(self):
        # save and unpack jobs
        self.epsilons = []
        self.metrics = []

        for j in self.jobs:
            self.epsilons.append(getattr(j.cfg, self.EPSILON_NAME))
            self.metrics.append(j.return_value)
