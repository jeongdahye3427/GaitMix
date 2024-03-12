import torch
import torch.nn as nn

START_AGE = 2
END_AGE = 87

# Pre-Activation Residual Unit
class PreActResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        bottleneck_channels = int(out_channels / 4)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, bottleneck_channels, 1, stride=stride, padding=0),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, out_channels, 1, stride=1, padding=0)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


# AttentionModule 1
class AttentionModule1(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r) # 56x56 -> 28x28
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r) # 28x28 -> 14x14
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r) # 14x14 -> 7x7
        self.soft_resdown4 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 14x14
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r) # 14x14 -> 28x28
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r) # 28x28 -> 56x56

        self.shortcut_short = PreActResidual(in_channels, out_channels, 1)
        self.shortcut_long = PreActResidual(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        # first downsample out 28
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        # 28 shortcut
        shape1 = ((x_s.size(2), x_s.size(3)))
        shortcut_long = self.shortcut_long(x_s)

        # second downsample out 14
        x_s = F.max_pool2d(x_s, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        # 14 shortcut
        shape2 = ((x_s.size(2), x_s.size(3)))
        shortcut_short = self.shortcut_short(x_s)

        # third downsample out 7
        x_s = F.max_pool2d(x_s, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown3(x_s)

        # mid
        x_s = self.soft_resdown4(x_s)
        x_s = self.soft_resup1(x_s)

        # first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        # second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        # third upsample out 54
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1+x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidual(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


# AttentionModule 2
class AttentionModule2(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r) # 28x28 -> 14x14
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r) # 14x14 -> 7x7
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 14x14
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r) # 14x14 -> 28x28

        self.shortcut = PreActResidual(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        # first downsample out 14
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        # 14 shortcut
        shape1 = (x_s.size(2), x_s.size(3))
        shortcut = self.shortcut(x_s)

        # second downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        # mid
        x_s = self.soft_resdown3(x_s)
        x_s = self.soft_resup1(x_s)

        # first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = nn.functional.interpolate(x_s, size=shape1)
        x_s += shortcut

        # second upsample out 28
        x_s = self.soft_resup3(x_s)
        x_s = nn.functional.interpolate(x_s, size=input_size)
        
        x_s = self.sigmoid(x_s)
        x = (1+x_s) * x_t
        x = self.last(x)
        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidual(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


# AttentionModule 3
class AttentionModule3(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r) # 14x14 -> 7x7
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup1 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 7x7
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r) # 7x7 -> 14x14

        self.shortcut = PreActResidual(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        # first downsample out 7
        x_s = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        # mid
        x_s = self.soft_resdown2(x_s)
        x_s = self.soft_resup1(x_s)

        # first upsample out 14
        x_s = self.soft_resup2(x_s)
        x_s = nn.functional.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1+x_s) * x_t
        x = self.last(x)
        return x

    def _make_residual(self, in_channels, out_channels, p):
        layers = []
        for _ in range(p):
            layers.append(PreActResidual(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


# Residual Attention Network
class AttentionNet(nn.Module):
    def __init__(self, nblocks, num_classes=10, init_weights=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.stage1 = self._make_stage(64, 256, nblocks[0], AttentionModule1, 1)
        self.stage2 = self._make_stage(256, 512, nblocks[1], AttentionModule2, 2)
        self.stage3 = self._make_stage(512, 1024, nblocks[2], AttentionModule3, 2)
        self.stage4 = nn.Sequential(
            PreActResidual(1024, 2048, 1),
            PreActResidual(2048, 2048, 1),
            PreActResidual(2048, 2048, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_stage(self, in_channels, out_channels, nblock, block, stride):
        stage = []
        stage.append(PreActResidual(in_channels, out_channels, stride))

        for i in range(nblock):
            stage.append(block(out_channels, out_channels))

        return nn.Sequential(*stage)

def Attention50():
    return AttentionNet([1,1,1])
