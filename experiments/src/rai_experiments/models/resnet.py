# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch import Tensor
from torchvision.models import resnet

from ._resnet_blocks import VizBasicBlock, VizBottleneck


class ResNet(resnet.ResNet):
    def __init__(self, *args, with_latent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_latent = with_latent

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.with_latent:
            return x

        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    block = resnet.BasicBlock
    if viz_relu:
        block = VizBasicBlock
    return ResNet(block, [2, 2, 2, 2], **kwargs)


def resnet50(viz_relu: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    block = resnet.Bottleneck
    if viz_relu:
        block = VizBottleneck
    return ResNet(block, [3, 4, 6, 3], **kwargs)
