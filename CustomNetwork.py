import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Custom ResNet-like network (using residual block and CBAM)
class ResBlock(nn.Module):
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

class ResNet(nn.Module):
    # 43 conv layers
    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, 3) # 224 -> 112
        self.layer2 = self.make_layer(ResBlock, 128, 2, 3) # 112 -> 56
        self.layer3 = self.make_layer(ResBlock, 256, 2, 3) # 56 -> 28
        self.layer4 = self.make_layer(ResBlock, 512, 2, 3)  # 28 -> 14
        self.layer5 = self.make_layer(ResBlock, 512, 2, 3) # 14 -> 7
        self.layer6 = self.make_layer(ResBlock, 512, 2, 3) # 7 -> 4
        self.layer7 = self.make_layer(ResBlock, 512, 2, 3) # 4 -> 2

        self.fc = nn.Linear(512, 3)

    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel

        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out) # not need softmax since CrossEntropy loss
        return out

def custom_resnet():
    return ResNet(ResBlock)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAM(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CBAM(256)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CBAM(512)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CBAM(512),
            nn.Dropout(0.25)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CBAM(512),
            nn.Dropout(0.5)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out

def custom_vgg():
    return VGGNet()

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
        self.out_fc = nn.Linear(256, 3)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.in_fc(out)
        out = self.hidden_fc(out)
        out = self.out_fc(out)
        return out

def baseline(size):
    return BaseLine(size)