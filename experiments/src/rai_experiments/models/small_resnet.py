# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import Tensor, nn
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

from ._resnet_blocks import VizBasicBlock, VizBottleneck


class ResNet(resnet.ResNet):
    """TorchVision ResNet Adapted for smaller images (i.e., CIFAR-10).

    Parameters
    ----------
    block: Type[BasicBlock] | Type[Bottleneck]

    layers: List[int]

    num_classes: int (default: 10)

    zero_init_residual: bool (default: False)

    groups: int (default: 1)

    width_per_group: int (default: 64)

    replace_stride_with_dilation: List[bool] | None (default: None)

    norm_layer: Callable[..., nn.Module] | None (default: None)

    input_channels: int (default: 3)
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_channels: int = 3,
    ):
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        del self.maxpool

        self.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.fc = nn.Linear(self.inplanes, 10)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    block = BasicBlock
    if viz_relu:
        block = VizBasicBlock
    return ResNet(block, [2, 2, 2, 2], **kwargs)


def resnet34(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    block = BasicBlock
    if viz_relu:
        block = VizBasicBlock
    return ResNet(block, [3, 4, 6, 3], **kwargs)


def resnet50(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    block = Bottleneck
    if viz_relu:
        block = VizBottleneck
    return ResNet(block, [3, 4, 6, 3], **kwargs)


def wide_resnet18_2(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """Wide ResNet-18 model"""
    kwargs["width_per_group"] = 64 * 2
    block = BasicBlock
    if viz_relu:
        block = VizBasicBlock
    return ResNet(block, [2, 2, 2, 2], **kwargs)


def wide_resnet50_2(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """Wide ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    kwargs["width_per_group"] = 64 * 2
    block = Bottleneck
    if viz_relu:
        block = VizBottleneck
    return ResNet(block, [3, 4, 6, 3], **kwargs)
