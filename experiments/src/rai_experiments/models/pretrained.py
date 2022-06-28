from functools import partial

import pooch
from typing_extensions import Literal, TypeAlias

try:
    import tqdm
except ImportError:
    tqdm = None

__all__ = ["load_model", "get_path_to_checkpoint"]

_MODEL_NAMES: TypeAlias = Literal[
    "mitll_cifar_l2_1_0.pt",
    "mitll_cifar_nat.pt",
    "mitll_imagenet_l2_3_0.pt",
    "mitll_restricted_imagenet_l2_3_0.pt",
    "imagenet_nat.pt",
]


_pre_trained_manager = pooch.create(
    path=pooch.os_cache("rai-experiments"),
    base_url="https://github.com/mit-ll-responsible-ai/data_hosting/releases/download/madrylab_models/",
    registry={
        "mitll_cifar_l2_1_0.pt": "md5:738f109685dfc55f7227dc6dc473379e",
        "mitll_cifar_nat.pt": "md5:34e15ff183735933e13f41ad27f755b4",
        "mitll_restricted_imagenet_l2_3_0.pt": "md5:db37145acfd8bdf7ef7c05f4d261c0d9",
        "mitll_imagenet_l2_3_0.pt": "md5:37728e5c9c47684c0edd373e9d80ac9b",
        "imagenet_nat.pt": "md5:0e98d33b24eafc63a1f9e4ae65ad1695",
    },
)


def get_path_to_checkpoint(model_name: _MODEL_NAMES) -> str:
    r"""
    Returns path to pre-trained model weights. This function takes care of downloading
    and caching weights.

    All model weights were provided by Madry Lab [1]_. We converted these weights to a
    format that does not require extraneous dependencies, such as `dill`, to load.

    Parameters
    ----------
    model_name: str
        Supported models
           - mitll_cifar_l2_1_0.pt
           - mitll_cifar_nat.pt
           - mitll_imagenet_l2_3_0.pt
           - mitll_restricted_imagenet_l2_3_0.pt
           - imagenet_nat.p

    Returns
    -------
    str

    Notes
    -----
    Descriptions of models:

    - `mitll_cifar_nat.pt`: This is a ResNet-50 model trained on CIFAR 10 using standard training with no adversarial perturbations in the loop (i.e., :math:`\epsilon=0`)
    - `mitll_cifar_l2_1_0.pt`: A ResNet-50 model trained on CIFAR 10 with perturbations generated via PGD using perturbations constrained to :math:`L^2`-ball of radius :math:`\epsilon=1.0`
    -  `mitll_imagenet_l2_3_0`: This is a ResNet-50 model trained on ImageNet with PGD using :math:`\epsilon=3.0`
    -  `mitll_restricted_imagenet_l2_3_0`: This is a ResNet-50 model trained on restricted ImageNet [2]_ with PGD using :math:`\epsilon=3.0`

    References
    ----------
    .. [1] https://github.com/MadryLab/robustness
    .. [2] https://github.com/MadryLab/robust_representations
    """

    return _pre_trained_manager.fetch(model_name, progressbar=(tqdm is not None))


def load_model(model_name: _MODEL_NAMES):
    r"""
    Loads pre-trained model weights. This function takes care of downloading and caching
    weights.

    All model weights were provided by Madry Lab [1]_. We converted these weights to a
    format that does not require extraneous dependencies, such as `dill`, to load.

    Parameters
    ----------
    model_name: str
        Supported models
           - mitll_cifar_l2_1_0.pt
           - mitll_cifar_nat.pt
           - mitll_imagenet_l2_3_0.pt
           - mitll_restricted_imagenet_l2_3_0.pt
           - imagenet_nat.p

    Returns
    -------
    loaded_model : torch.nn.Module
        The loaded model. (Note that the model is **not** placed in eval-mode by default).

    Notes
    -----
    Descriptions of models:

    - `mitll_cifar_nat.pt`: This is a ResNet-50 model trained on CIFAR 10 using standard training with no adversarial perturbations in the loop (i.e., :math:`\epsilon=0`)
    - `mitll_cifar_l2_1_0.pt`: A ResNet-50 model trained on CIFAR 10 with perturbations generated via PGD using perturbations constrained to :math:`L^2`-ball of radius :math:`\epsilon=1.0`
    -  `mitll_imagenet_l2_3_0`: This is a ResNet-50 model trained on ImageNet with PGD using :math:`\epsilon=3.0`
    -  `mitll_restricted_imagenet_l2_3_0`: This is a ResNet-50 model trained on restricted ImageNet [2]_ with PGD using :math:`\epsilon=3.0`

    References
    ----------
    .. [1] https://github.com/MadryLab/robustness
    .. [2] https://github.com/MadryLab/robust_representations
    """

    import torch
    from torchvision import transforms

    from rai_toolbox.mushin._utils import load_from_checkpoint

    from ..models.resnet import resnet50
    from ..models.small_resnet import resnet50 as small_resnet50

    if model_name in {"mitll_cifar_l2_1_0.pt", "mitll_cifar_nat.pt"}:
        model = small_resnet50
        norm = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    elif model_name in {
        "mitll_imagenet_l2_3_0.pt",
        "mitll_restricted_imagenet_l2_3_0.pt",
        "imagenet_nat.pt",
    }:

        model = partial(resnet50, num_classes=9 if "restricted" in model_name else 1000)
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name}\nAvailable models: {', '.join(_pre_trained_manager.registry_files + ['imagenet_nat.pt'])}"
        )

    base_model = load_from_checkpoint(
        model=model(),
        ckpt=get_path_to_checkpoint(model_name),
        weights_key="state_dict" if model_name != "imagenet_nat.pt" else "model",
    )

    model = torch.nn.Sequential(norm, base_model)
    return model
