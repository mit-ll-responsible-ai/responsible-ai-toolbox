"""
Script to reproduce the table from the Madry paper.

Experiment Configurations:
    - standard: standard training (full dataset)
    - robust: adversarial training (full dataset)
    - standard_fast: standard training with only 100 examples
    - robust_fast: adversarial training with only 100 examples

1. CPU (single process)
$ python madry.py +experiment=standard_fast
standard [tensor(0.9570), tensor(0.1063), tensor(0.0216), tensor(0.), tensor(0.)]

$ python madry.py +experiment=robust_fast
robust [tensor(0.8251), tensor(0.7855), tensor(0.6789), tensor(0.4522), tensor(0.1719)]

2. Torch Distributed (2 GPU)
$ torchrun --nproc_per_node=2 madry.py +experiment=standard_fast
standard [tensor(0.9570), tensor(0.1063), tensor(0.0216), tensor(0.), tensor(0.)]

$ torchrun --nproc_per_node=2 madry.py +experiment=standard_fast
robust [tensor(0.8251), tensor(0.7855), tensor(0.6789), tensor(0.4522), tensor(0.1719)]

3. Run multiple experiments (2 GPU)
torchrun --nproc_per_node=2 madry.py +experiment=robust,standard --multirun

4. Run multiple experiments (2 GPU) with different seeds
torchrun --nproc_per_node=2 madry.py +experiment=robust,standard --multirun --seed=1,2
"""

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    cast,
)

import torch as tr
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from typing_extensions import Protocol, Self, TypedDict

from rai_experiments.models.pretrained import _MODEL_NAMES, load_model
from rai_toolbox import evaluating
from rai_toolbox.optim import L2ProjectedOptim
from rai_toolbox.perturbations.init import uniform_like_l2_n_ball_
from rai_toolbox.perturbations.models import AdditivePerturbation
from rai_toolbox.perturbations.solvers import gradient_ascent


class ImageClassificationData(TypedDict):
    """Type for image classification data."""

    image: Tensor
    label: Tensor


class ImageClassifier(Protocol):
    """Protocol for image classifiers."""

    def to(self, device: Optional[tr.device] = ...) -> Self:
        """Move the model to a device."""
        ...

    def __call__(self, image: tr.Tensor) -> tr.Tensor:
        """Classify an image and output the logits."""
        ...


class Perturbation(Protocol):
    def __call__(
        self, model: ImageClassifier, batch: ImageClassificationData, **kwds: Any
    ) -> ImageClassificationData:
        ...


def collator(examples: List[Tuple[Tensor, int]]) -> ImageClassificationData:
    """
    Collator for image classification datasets.

    Parameters
    ----------
    examples : list
    """
    images, labels = default_collate(examples)
    return ImageClassificationData(image=images, label=labels)


def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = "~/data",
    transforms_test: Callable = transforms.Compose([transforms.ToTensor()]),
    batch_size: int = 10,
) -> Iterable[ImageClassificationData]:
    """
    Load CIFAR10 dataset.

    Parameters
    ----------
    n_examples : int, optional
        Number of examples to load. Default: None.
    data_dir : str, optional
        Directory where to store the data. Default: `~/data`.
    transforms_test : callable, optional
        Transformations to apply to the data. Default: `transforms.ToTensor()`.
    batch_size : int, optional
        Batch size. Default: 10.

    Returns
    -------
    dataset : Iterable[ImageClassificationData]

    Examples
    --------
    >>> dataset = load_cifar10()
    >>> for batch in dataset:
    ...     print(batch)
    """
    dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=transforms_test, download=True
    )

    if n_examples is not None:
        dataset = Subset(dataset, list(range(n_examples)))

    sampler = None
    if dist.is_initialized():
        print("*** Using DistributedSampler ***")
        sampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        sampler=sampler,
    )


def adversarial_attack_l2(
    model: ImageClassifier,
    batch: ImageClassificationData,
    steps: int = 10,
    epsilon: float = 0.0,
    lr: float = 1.0,  # pylint: disable=invalid-name
    **kwargs: Any,
) -> ImageClassificationData:
    """
    Generate adversarial examples using PGD.

    Parameters
    ----------
    model : ImageClassifier
        Model to attack.
    batch : dict
        Batch of data.
    steps : int, optional
        Number of steps to run the attack. Default: 10.
    epsilon : float, optional
        Maximum perturbation norm. Default: 0.0.
    lr : float, optional
        Learning rate. Default: 1.0.
    **kwargs : Any
        Additional keyword arguments for `gradient_ascent`.

    Returns
    -------
    batch : dict
        Batch of adversarial examples.

    Examples
    --------
    """
    image = batch["image"]
    label = batch["label"]
    pert_model = AdditivePerturbation(image, init_fn=uniform_like_l2_n_ball_)
    image_pert, _ = gradient_ascent(
        model=model,
        data=image,
        target=label,
        optimizer=L2ProjectedOptim,
        perturbation_model=pert_model,
        steps=steps,
        epsilon=epsilon,
        lr=lr,
        **kwargs,
    )
    batch["image"] = image_pert
    return batch


def evaluate(
    model: ImageClassifier,
    data: Iterable[ImageClassificationData],
    perturbation: Perturbation,
    device: tr.device = tr.device("cpu"),
    **kwargs,
):
    """
    Evaluate a model on a dataset against a perturbation.

    Parameters
    ----------
    model : ImageClassifier
        Model to evaluate.
    data : Iterable[ImageClassificationData]
        Dataset to evaluate on.
    perturbation : Perturbation
        Perturbation to apply to the data.
    device : int | str
        Device to use. Default: `cpu`.
    **kwargs : Any
        Keyword arguments to pass to the perturbation.

    Returns
    -------
    accuracy : float

    Examples
    --------
    >>> model = load_model("resnet18")
    >>> dataset = load_cifar10()
    >>> evaluate(model, dataset, adversarial_attack_l2)
    """
    model.to(device)

    if dist.is_initialized():
        model = tr.nn.parallel.DistributedDataParallel(model)

    accuracy = MulticlassAccuracy(num_classes=10)
    accuracy.to(device)
    accuracy.reset()

    for batch in data:
        batch = {
            k: v.to(device=device) for k, v in batch.items() if isinstance(v, Tensor)
        }

        if TYPE_CHECKING:
            assert isinstance(model, tr.nn.Module)
            batch = cast(ImageClassificationData, batch)

        batch = perturbation(model, batch, **kwargs)

        with evaluating(model), tr.inference_mode():
            logits = model(batch["image"])
            accuracy.update(logits, batch["label"])

    return accuracy.compute().item()


def main(
    model_info: Dict[str, _MODEL_NAMES],
    num_examples: int = 100,
    epsilons: List[float] = [0.001, 0.25, 0.5, 1.0, 2.0],
    steps: int = 20,
    batch_size: int = 32,
):
    """
    Calculate the accuracy of a model on CIFAR10 against a perturbation.

    Parameters
    ----------
    model : Dict[str, str]
        Model to evaluate.
    num_examples : int
        Number of examples to evaluate on. Default: 100.
    epsilons : List[float]
        List of perturbation norms to evaluate. Default: [0.001, 0.25, 0.5, 1.0, 2.0].
    steps : int
        Number of steps to run the attack. Default: 20.
    batch_size : int
        Batch size. Default: 32.
    """
    local_rank = os.environ.get("LOCAL_RANK", None)
    device = tr.device("cpu")

    if local_rank is not None:
        local_rank = int(local_rank)

        if tr.cuda.is_available():
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
                device = tr.device(f"cuda:{local_rank}")
                world_size = dist.get_world_size()
                print(
                    f"**** Intializing Torch Distributed: {local_rank} / {world_size} (Local Rank / World Size) ****"
                )
            else:
                world_size = dist.get_world_size()
                device = tr.device(f"cuda:{local_rank}")
                print(
                    f"**** Using Existing Torch Distributed: {local_rank} / {world_size} (Local Rank / World Size) ****"
                )

    elif tr.cuda.is_available():
        device = tr.device("cuda")

    model = load_model(model_info["ckpt"])
    dataset = load_cifar10(num_examples, batch_size=batch_size)

    iterator = tqdm(epsilons) if local_rank in (0, None) else epsilons
    result = [
        evaluate(
            model,
            dataset,
            adversarial_attack_l2,
            device=device,
            epsilon=e,
            steps=steps,
            lr=2.5 * e / steps,
        )
        for e in iterator
    ]

    if local_rank in (0, None):
        print(model_info["name"], result)


if __name__ == "__main__":
    import hydra_zen
    from hydra.conf import HydraConf, JobConf, RunDir, SweepDir

    # Define hydra directories
    # --- We define a seperate directory for each rank to demonstrate both processes running.
    # --- You do not need to do this in your own code, the logs will show both processes running.
    local_rank = os.environ.get("LOCAL_RANK", None)
    hydra_zen.store(
        HydraConf(
            job=JobConf(chdir=True),
            run=RunDir("outputs/${now:%Y-%m-%d-%H-%M-%S}" + f"/rank_{local_rank}"),
            sweep=SweepDir(
                dir="multirun/${now:%Y-%m-%d-%H-%M-%S}",
                subdir="${hydra.job.id}" + f"/rank_{local_rank}",
            ),
        )
    )

    # Configure experiments
    hydra_zen.store(
        hydra_zen.builds(dict, ckpt="mitll_cifar_nat.pt", name="standard"),
        group="model_info",
        name="standard",
    )

    hydra_zen.store(
        hydra_zen.builds(dict, ckpt="mitll_cifar_l2_1_0.pt", name="robust"),
        group="model_info",
        name="robust",
    )

    Config = hydra_zen.make_config(
        hydra_defaults=["_self_", {"model_info": "standard"}],
        model_info=hydra_zen.MISSING,
        num_examples=100,
        batch_size=64,
    )
    hydra_zen.store(Config, name="config")

    # Define experiments
    # --- We define a seperate experiment store both standard and robust models
    # --- while also providing fast versions of each for testing.
    experiment_store = hydra_zen.store(group="experiment", package="_global_")
    experiment_store(
        hydra_zen.make_config(
            hydra_defaults=["_self_", {"override /model_info": "standard"}],
            num_examples=100,
            bases=(Config,),
        ),
        name="standard_fast",
    )
    experiment_store(
        hydra_zen.make_config(
            hydra_defaults=["_self_", {"override /model_info": "robust"}],
            num_examples=100,
            bases=(Config,),
        ),
        name="robust_fast",
    )
    experiment_store(
        hydra_zen.make_config(
            hydra_defaults=["_self_", {"override /model_info": "standard"}],
            num_examples=None,
            bases=(Config,),
        ),
        name="standard",
    )
    experiment_store(
        hydra_zen.make_config(
            hydra_defaults=["_self_", {"override /model_info": "robust"}],
            num_examples=None,
            bases=(Config,),
        ),
        name="robust",
    )

    hydra_zen.store.add_to_hydra_store()
    hydra_zen.zen(main).hydra_main(
        config_path=None, config_name="config", version_base="1.3"
    )
