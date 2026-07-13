from torch import nn
import torchvision.models as models

class ResNetBlock(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        backbone = models.resnet18(weights='DEFAULT')
        self.resnet = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        layer4 = self.resnet[7]
        layer4[0].conv1.stride = (1, 1)
        if layer4[0].downsample is not None:
            layer4[0].downsample[0].stride = (1, 1)
        self.out_channels = 512

    def forward(self, x):
        return self.resnet(x)
