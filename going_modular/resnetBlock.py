import torch
from torch import nn
from torchvision import models


class ResNetBlock(nn.Module):
    """
    ResNet18 backbone с multi-scale признаками.

    Изменения по сравнению с прошлой версией:
    - Возвращает КОНКАТЕНАЦИЮ признаков layer3 + layer4, а не только layer4.
      layer3 сохраняет более мелкое семантическое разрешение (лучше видит мелкие
      объекты вроде bottle/chair/pottedplant), а layer4 — более абстрактные,
      крупные признаки (лучше для целых объектов вроде train/cat/dog).
    - stride в layer4 по-прежнему модифицирован на (1,1), чтобы его выход
      остался 14x14 и совпадал по пространственному размеру с layer3 —
      это необходимо для конкатенации по каналам.
    - Заморозка: conv1..layer3 заморожены (freeze_until_layer=3 по умолчанию),
      layer4 остаётся обучаемым — как и в предыдущей рабочей конфигурации.
    """
    def __init__(self, input_size=224, freeze_until_layer=3, use_high_res=False):
        super().__init__()
        backbone = models.resnet18(weights='DEFAULT')

        # ВАЖНО: сохраняем ту же структуру self.resnet (Sequential с индексами 0-7),
        # что была в исходной модели — иначе веса из чекпоинта не подхватятся по именам.
        # 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, 5=layer2, 6=layer3, 7=layer4
        self.resnet = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        # Модификация stride в layer4 (индекс 7) — как и раньше, сохраняем 14x14
        layer4 = self.resnet[7]
        layer4[0].conv1.stride = (1, 1)
        if layer4[0].downsample is not None:
            layer4[0].downsample[0].stride = (1, 1)

        # Замораживаем conv1..layer3 (индексы 0 до 4+freeze_until_layer-1),
        # layer4 остаётся обучаемым — как в предыдущей рабочей конфигурации
        layers_to_freeze = list(self.resnet.children())[:4 + freeze_until_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        self.use_high_res = use_high_res

        if use_high_res:
            # layer2 (128 каналов, 28x28) + layer4 upsampled (512 каналов, 28x28) = 640
            self.out_channels = 128 + 512
        else:
            # старый вариант: layer3 (256) + layer4 (512) = 768, оба на 14x14
            self.out_channels = 256 + 512


    def forward(self, x):
      if self.use_high_res:
          # прогоняем до layer2 включительно (индексы 0-5) — реальное высокое разрешение 28x28
          layer2_out = self.resnet[:6](x)            # [B, 128, 28, 28]
          layer3_out = self.resnet[6](layer2_out)     # [B, 256, 14, 14]
          layer4_out = self.resnet[7](layer3_out)     # [B, 512, 14, 14]

          # апсемплим layer4 до разрешения layer2 (28x28), чтобы можно было объединить по каналам
          layer4_upsampled = nn.functional.interpolate(
              layer4_out, size=layer2_out.shape[-2:], mode='bilinear', align_corners=False
          )
          combined = torch.cat([layer2_out, layer4_upsampled], dim=1)  # [B, 640, 28, 28]
      else:
          layer3_out = self.resnet[:7](x)
          layer4_out = self.resnet[7](layer3_out)
          combined = torch.cat([layer3_out, layer4_out], dim=1)  # [B, 768, 14, 14]

      return combined
