import pooch
from typing_extensions import Literal

__all__ = ["load_model"]


_pre_trained_manager = pooch.create(
    path=pooch.os_cache("rai-experiments"),
    base_url="https://github.com/mit-ll-responsible-ai/data_hosting/releases/download/madrylab_models/",
    registry={
        "mitll_cifar_l2_1_0.pt": "md5:738f109685dfc55f7227dc6dc473379e",
        "mitll_cifar_nat.pt": "md5:34e15ff183735933e13f41ad27f755b4",
        "mitll_restricted_imagenet_l2_3_0.pt": "md5:db37145acfd8bdf7ef7c05f4d261c0d9",
        "mitll_imagenet_l2_3_0.pt": "md5:37728e5c9c47684c0edd373e9d80ac9b",
    },
)


def load_model(
    name: Literal[
        "mitll_cifar_l2_1_0.pt",
        "mitll_cifar_nat.pt",
        "mitll_imagenet_l2_3_0.pt",
    ]
):
    # TODO: which architecture is correct for mitll_restricted_imagenet_l2_3_0?

    import torch
    import torchvision

    from rai_toolbox.mushin._utils import load_from_checkpoint

    from ..models.resnet import resnet50
    from ..models.small_resnet import resnet50 as small_resnet50

    if name in {"mitll_cifar_l2_1_0.pt", "mitll_cifar_nat.pt"}:
        model = small_resnet50
    elif name in {"mitll_imagenet_l2_3_0.pt"}:
        model = resnet50
    else:
        raise ValueError(
            f"Unknown model name: {name}\nAvailable models: {', '.join(_pre_trained_manager.registry_files)}"
        )

    base_model = load_from_checkpoint(
        model=model(),
        ckpt=_pre_trained_manager.fetch(name),
        weights_key="state_dict",
    )
    # Transform to normalize the data samples by the mean and standard deviation
    normalizer = torchvision.transforms.transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    model = torch.nn.Sequential(normalizer, base_model)
    return model
