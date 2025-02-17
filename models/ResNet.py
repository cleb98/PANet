from typing import Any, Callable, List, Optional, Type, Union


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
from torchsummary import summary
from .base import BaseModel
# from IPython.core.display import display_latex


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    # type: (int, int, int, int, int) -> nn.Conv2d
    """
    3x3 convolution with 3x3 kernel.
    Provide also groups and dilation parameters
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding = dilation,
        groups = groups,
        bias = False,
        dilation = dilation,
    )

def conv1x1(in_channels, out_channels, stride=1):
    # type: (int, int, int) -> nn.Conv2d
    """
    1x1 convolution with 1x1 kernel
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

# class BasicBlock(nn.Module):
#     expansion: int = 1
#     #copy of the BasicBlock from torchvision.models.resnet
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # The output is Y = F(x) + x, where F(x) is the residual mapping to be learned.
    # The block that learn F(x) is composed by 3 convolutions (plus batchnorm and relu after each convolution):
    # 1x1 conv -> 3x3 conv -> 1x1 conv
    # 1. The first 1x1 convolution is used to reduce the number of channels:
    # 2. The 3x3 convolution is used to extract features
    # 3. The last 1x1 convolution is used to expand the number of channels
    # conv1x1, batchnorm -> no relu
    # 4. The block is completed by the residual connection:
    # -> Y = F(x) + x
    # -> if x and F(x) have different number of channels, x is downsampled to match the number of channels of F(x)
    # 5. The block is completed by the activation function if arg use_relu is True
    # -> Output = relu(Y)

    expansion = 4 #expand the number of channels

    def __init__(
            self,
            in_channels: int, #i dont like to call planes the number of channels
            channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module]=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            use_relu: bool = True
    )-> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # the number of channels in the convolutions if groups > 1
        out_channels = int(channels * (base_width / 64.)) * groups

        #half the number of channels in the first 1x1 convolution
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        #continue with 3x3 convolution to extract features
        self.conv2 = conv3x3(out_channels, out_channels, stride, groups, dilation)
        self.bn2 = norm_layer(out_channels)
        #expand the number of channels in the last 1x1 convolution
        self.conv3 = conv1x1(out_channels, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)

        #activation function
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_relu = use_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if the number of channels of the input is different from the output need to be downsampled
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity #residual connection
        if self.use_relu:
            out = self.relu(out)
        return out

class ResNetV1_5(BaseModel):
    """
    ResNetV1.5 model
    """
    def __init__(
            self,
            block: Type[Bottleneck],
            layers: List[int],
            num_classes: int = None,
            pretrained_path: Optional[str] = None,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetV1_5, self).__init__()
        self.pretrained_path = pretrained_path
        self.zero_init_residual = zero_init_residual
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, 2, 2] #it uses dilation instead of stride to increase the receptive field
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        #input_layer: conv7x7, BN, ReLU, MaxPool
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Bottleneck blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if num_classes is not None:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_relu=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        else: #if num_classes is None, resnet is used as encoder for feature extraction
            print("ResNet used as encoder")
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_relu=False)


        self._init_weights()


    def _make_layer(self, block, channels, blocks, stride=1, dilate=False, last_relu=True):
        # type: (type[Bottleneck], int, int, int, bool, bool) -> nn.Sequential

        """
        Make a layer of blocks, each block is a Bottleneck block.
        The function:
        1. creates the downsampling layer (if stride > 1 or dilate is True) to match the number of channels of the input and the output
        2. if is necessary to increase the receptive field and maintain the spatial resolution, use dilation instead of stride
        3. create the layer: Stack the first block (which can modify the dimensions) and then the other blocks that maintain the configuration.
        Args:
            block:
                block to be used (e.g., Bottleneck block)
            channels:
                number of channels of the block (e.g., 64)
            blocks:
                number of blocks (to define the depth of the network)
            stride:
                stride of the first convolution
            dilate:
                if True, apply dilation to the blocks to increase the receptive field
            last_relu:
                last_relu, if True, apply relu to the output (default is True).
                Typically, the last layer of a resnet used as a feature extractor does not have the last relu to extract embeddings.

        Returns:
            nn.Sequential


        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        #downsampling layer to match the number of channels of the input and the output
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )

        layers = []
        #first block
        layers.append(
            block(self.in_channels, channels, stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        #after the first Bottleneck, the number of channels for the next blocks is expanded by the block.expansion factor
        self.in_channels = channels * block.expansion #update the number of channels for the next blocks
        for i in range(1, blocks):
            if i < blocks - 1:
                use_relu = True
            else:
                use_relu = last_relu
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_relu=use_relu)
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        #initialize the weights of the model, following the initialization of the weights of the original ResNet.
        #In addition, load the weights of the model if a pretrained model is provided.

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

                # uncomment if you want to zero-initialize the weights of the BasicBlock
                # actually BasicBlock not used in this model
                # elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                #     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        if self.pretrained_path is not None:
            pretrained_dict = torch.load(self.pretrained_path, map_location='cpu')
            #get the model state dictionary, aka the weights of the model
            model_dict = self.state_dict()
            #filter out unnecessary keys
            #when the resnet is used in encoder mode, it doesn't have the 'fc' layer so need to filter it out
            compatible_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            #update the model state dictionary and load it
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict)

    def _encoder_forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        features = self._encoder_forward(x)
        if self.num_classes is not None:
            x = self.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        return features


def resnet50(pretrained_path=None, num_classes=None):
    # type: (Optional[str], Optional[int]) -> ResNetV1_5
    """
    ResNet-50 model:
    number of block for each of the 4 layers: [3, 4, 6, 3]
    """
    return ResNetV1_5(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained_path=pretrained_path)

def resnet101(pretrained_path=None, num_classes=None):
    # type: (Optional[str], Optional[int]) -> ResNetV1_5
    """
    ResNet-101 model:
    number of block for each of the 4 layers: [3, 4, 23, 3]
    """
    return ResNetV1_5(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, pretrained_path=pretrained_path)


if __name__ == '__main__':

    pretrained_path = "../pretrained_model/resnet50-19c8e357.pth"
    model50 = resnet50(num_classes=None, pretrained_path=pretrained_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model50.to(device)
    x = torch.randn(1, 3, 417, 417).to(device)
    out = model50(x)
    print("Output ResNet50:", out.shape, "\n")
    # Assicurati che l'input_size corrisponda alle dimensioni attese dal tuo modello
    summary(model50, (3, 417, 417), device=str(device))

    # pretrained_path = "../pretrained_model/resnet101-63fe2227.pth"
    # model101 = resnet101(num_classes=1000, pretrained_path=pretrained_path)
    # out101 = model101(x)
    # print("Output ResNet101:", out101.shape)
            
