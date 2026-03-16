from typing import List, Optional, override

import torch
import torch.nn as nn
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCIFAR(FeatureExtractor):
    """
    ResNet-32 (CIFAR-style) implementation in PyTorch.

    Depth: 6n + 2 -> for ResNet-32, n = 5
    Architecture follows "Deep Residual Learning for Image Recognition" but uses the CIFAR variant
    (3x3 convs throughout, no initial 7x7 conv / maxpool).

    Usage:
        model = resnet32(num_classes=10)

    """

    def __init__(
        self,
        block: type[BasicBlock] = BasicBlock,
        layers: List[int] = [5, 5, 5],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
    ):
        """
        block: block class (BasicBlock)
        layers: list with number of blocks in each of the 3 stages, e.g. [n, n, n]
        """
        super(ResNetCIFAR, self).__init__(no_features=64 * block.expansion)
        self.inplanes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

        self.freeze_weights = freeze_weights

        if freeze_weights:
            for p in self.parameters():
                p.requires_grad = False
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_weights:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _make_layer(
        self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[BasicBlock] = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @override
    def forward(self, case: Tensor, use_classification_head: bool = False) -> Tensor:
        if case.ndim == 4 and case.shape[-1] in [1, 3]:
            case = case.permute(0, 3, 1, 2).contiguous()

        x = case
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if use_classification_head:
            x = self.fc(x)

        return x.squeeze()

    @override
    def get_output_features(self) -> int:
        return self.no_features

    @override
    def plot_parameters(self):
        print("CNN PARAMETERS ARE NOT PRINTED")
        pass


def Resnet32(
    num_classes: int = 10,
    weights_path: Optional[str] = None,
    freeze_weights: bool = False,
):
    return ResNetCIFAR(
        BasicBlock,
        [5, 5, 5],
        num_classes=num_classes,
        zero_init_residual=False,
        weights_path=weights_path,
        freeze_weights=freeze_weights,
    )
