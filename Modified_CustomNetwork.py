import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KANLayer

# CBAM Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)) # average across H and W
        max_out = self.fc(self.max_pool(x)) # max across H and W
        out = avg_out + max_out # sum both
        return self.sigmoid(out) # suqash between 0 and 1

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # (N, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (N, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1) # (N, 2, H, W)
        out = self.conv(x_cat) # (N, 1, H, W)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Basic Conv + BN + ReLU Block; remain same size if stride=1
def convbnrelu(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

# Residual +
class ResBlock(nn.Module):
    expansion = 4
    inception_layer = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.branch_channel = out_channel // self.inception_layer
        self.branch1 = nn.Sequential(
            convbnrelu(in_channel, self.branch_channel, kernel_size=1, stride=1),
        )
        self.branch2 = nn.Sequential(
            convbnrelu(in_channel, self.branch_channel, kernel_size=1, stride=1),
            convbnrelu(self.branch_channel, self.branch_channel, kernel_size=3, stride=1),
        )
        self.branch3 = nn.Sequential(
            convbnrelu(in_channel, self.branch_channel, kernel_size=1, stride=1),
            convbnrelu(self.branch_channel, self.branch_channel, kernel_size=5, stride=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convbnrelu(in_channel, self.branch_channel, kernel_size=1, stride=1),
        )
        self.proj = nn.Sequential(
            convbnrelu(out_channel, out_channel, kernel_size=3, stride=stride),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel * self.expansion),
        )
        self.cbam = CBAM(out_channel * self.expansion)
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if in_channel != out_channel * self.expansion or stride >1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * self.expansion),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out1 = torch.cat([x1, x2, x3, x4], dim=1)
        out1 = self.proj(out1)
        out2 = self.shortcut(x)
        out1 = self.cbam(out1)
        out = out1 + out2
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            convbnrelu(in_channel=3, out_channel=64, kernel_size=7, stride=2), # 224 - 112
            convbnrelu(in_channel=64, out_channel=64, kernel_size=3, stride=2),
            CBAM(64)
        )
        self.layer1 = self.make_layer(ResBlock, 64, 1, 3) # 56 -56
        self.layer2 = self.make_layer(ResBlock, 128, 2, 4) # 56 - 28
        self.layer3 = self.make_layer(ResBlock, 256, 2, 6) # 28 -> 14
        self.layer4 = self.make_layer(ResBlock, 512, 2, 3)  # 14 -> 7

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, 3)

    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel * ResBlock.expansion

        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def custom_resnet():
    return ResNet(ResBlock)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            convbnrelu(3, 32, 3, 1),
            convbnrelu(32, 32, 3, 2),
            CBAM(32)
        )
        self.conv2 = nn.Sequential(
            convbnrelu(32, 64, 3, 1),
            convbnrelu(64, 64, 3, 2),
            CBAM(64)
        )
        self.conv3 = nn.Sequential(
            convbnrelu(64, 128, 3, 1),
            convbnrelu(128, 128, 3, 2),
            CBAM(128)
        )
        self.conv4 = nn.Sequential(
            convbnrelu(128, 256, 3, 1),
            convbnrelu(256, 256, 3, 2),
            CBAM(256)
        )
        self.conv5 = nn.Sequential(
            convbnrelu(256, 512, 3, 1),
            convbnrelu(512, 512, 3, 2),
            CBAM(512)
        )
        self.conv6 = nn.Sequential(
            convbnrelu(512, 512, 3, 1),
            convbnrelu(512, 512, 3, 2),
            CBAM(512),
            nn.Dropout(0.25)
        )
        self.conv7 = nn.Sequential(
            convbnrelu(512, 512, 3, 1),
            convbnrelu(512, 512, 3, 2),
            CBAM(512),
            nn.Dropout(0.5)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 3)


    def forward(self, x):
        batch_size = x.size(0)

        out = self.conv1(x) # (N, 3, 224, 224) -> (N, 32, 112, 112)
        out = self.conv2(out) # (N, 32, 112, 112) -> (N, 64, 56, 56)
        out = self.conv3(out) # (N, 64, 56, 56) -> (N, 128, 28, 28)
        out = self.conv4(out) # (N, 128, 28, 28) -> (N, 256, 14, 14)
        out = self.conv5(out) # (N, 256, 14, 14) -> (N, 512, 7, 7)
        out = self.conv6(out) # (N, 512, 7, 7) -> (N, 512, 4, 4)
        out = self.conv7(out) # (N, 512, 4, 4) -> (N, 512, 2, 2)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def custom_vgg():
    return VGGNet()


class BaseInception(nn.Module):
    def __init__(self, in_channel, out_channel_list, reduced_channel_list):
        super(BaseInception, self).__init__()
        self.branch1 = nn.Sequential(
            convbnrelu(in_channel, out_channel_list[0], 1, 1),
        )
        self.branch2 = nn.Sequential(
            convbnrelu(in_channel, reduced_channel_list[0], 1, 1),
            convbnrelu(reduced_channel_list[0], out_channel_list[1], 3, 1),
        )
        self.branch3 = nn.Sequential(
            convbnrelu(in_channel, reduced_channel_list[1], 1, 1),
            convbnrelu(reduced_channel_list[1], out_channel_list[2], 5, 1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convbnrelu(in_channel, out_channel_list[3], 3, 1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1) # concat channels
        return out

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        # 224 - 112 - 56
        self.block1 = nn.Sequential(
            convbnrelu(3, 64, 7, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            convbnrelu(64, 128, 3, 1),
            CBAM(128)
        )
        # 56 - 28
        self.block2 = nn.Sequential(
            BaseInception(128, out_channel_list=[64, 64, 64, 64], reduced_channel_list=[16, 16]),
            CBAM(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 28 - 14
        self.block3 = nn.Sequential(
            BaseInception(256, out_channel_list=[96, 96, 96, 96], reduced_channel_list=[32, 32]),
            CBAM(384),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 14 - 7
        self.block4 = nn.Sequential(
            BaseInception(384, out_channel_list=[128, 128, 128, 128], reduced_channel_list=[48, 48]),
            CBAM(512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 7 - 4
        self.block5 = nn.Sequential(
            BaseInception(512, out_channel_list=[192, 192, 192, 192], reduced_channel_list=[64, 64]),
            CBAM(768),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # (N, 768, 1, 1)
        self.fc = nn.Linear(768, 3)


    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def custom_inception():
    return InceptionNet()

# baseline ANN model
class BaseLine(nn.Module):
    def __init__(self, size):
        super(BaseLine, self).__init__()

        # input fc layer
        self.in_fc = nn.Sequential(
            nn.Linear(size*size*3, 1024),
            nn.ReLU()
        )

        # hidden fc layer
        self.hidden_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # end fc layer
        self.out_fc = nn.Linear(128, 3)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.in_fc(out)
        out = self.hidden_fc(out)
        out = self.out_fc(out)
        return out

def baseline(size):
    return BaseLine(size)

# Plain resblock. Extra CBAM
class ResBlock1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.cbam = CBAM(out_channel)
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride >1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out1 = self.cbam(out1)
        out = out1 + out2
        out = F.relu(out)
        return out

# More conv-layer resblock, bottlneck block
class ResBlock2(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        #self.expansion = 4
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channel * self.expansion),
            nn.ReLU()
        )
        self.cbam = CBAM(out_channel * self.expansion)
        self.shortcut = nn.Sequential()
        if in_channel != out_channel * self.expansion or stride >1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * self.expansion),
        )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out1 = self.cbam(out1)
        out = out1 + out2
        out = F.relu(out)
        return out

# 1st try of combining inception to resblock
class ResBlock3(nn.Module):
    expansion = 2
    inception_layer = 4
    def __init__(self, in_channel, out_channel, stride=1, reduced_channel=16):
        super(ResBlock, self).__init__()
        self.concat_channel = out_channel * self.inception_layer
        self.branch1 = nn.Sequential(
            convbnrelu(in_channel, out_channel, kernel_size=1, stride=1),
        )
        self.branch2 = nn.Sequential(
            convbnrelu(in_channel, reduced_channel, kernel_size=1, stride=1),
            convbnrelu(reduced_channel, out_channel, kernel_size=3, stride=1),
        )
        self.branch3 = nn.Sequential(
            convbnrelu(in_channel, reduced_channel, kernel_size=1, stride=1),
            convbnrelu(reduced_channel, out_channel, kernel_size=5, stride=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convbnrelu(in_channel, out_channel, kernel_size=1, stride=1),
        )
        self.proj = nn.Sequential(
            convbnrelu(self.concat_channel, self.concat_channel, kernel_size=3, stride=stride),
            nn.Conv2d(self.concat_channel, self.concat_channel * self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.concat_channel * self.expansion),
        )
        self.cbam = CBAM(self.concat_channel * self.expansion)
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if in_channel != self.concat_channel * self.expansion or stride >1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.concat_channel * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.concat_channel * self.expansion),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out1 = torch.cat([x1, x2, x3, x4], dim=1)
        out1 = self.proj(out1)
        out2 = self.shortcut(x)
        out1 = self.cbam(out1)
        out = out1 + out2
        out = self.relu(out)
        return out