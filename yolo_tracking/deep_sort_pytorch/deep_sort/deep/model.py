import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


# 通道重排，跨group信息交流
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CBRM(nn.Module):  # conv BN ReLU Maxpool2d
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(CBRM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = ch_out // 2
        assert (self.stride != 1) or (ch_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(ch_in, ch_in, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(ch_in),

                nn.Conv2d(ch_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(ch_in if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 按照维度1进行split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class Net(nn.Module):
    def __init__(self, num_classes=776, reid=False):
        super(Net, self).__init__()
        self.conv = CBRM(3, 24)
        self.stage1 = self._make_stage(24, 48, [(2, 1), (1, 3)])  # 第一个阶段，包含两种步幅设置
        self.stage2 = self._make_stage(48, 96, [(2, 1), (1, 7)])  # 第二个阶段，包含两种步幅设置
        self.stage3 = self._make_stage(96, 192, [(2, 1), (1, 3)])  # 第三个阶段，包含两种步幅设置
        self.additional_conv = nn.Conv2d(192,1024,1,1)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, stride_repeat):
        layers = []
        for stride, repeat_times in stride_repeat:
            for i in range(repeat_times):
                layers.append(Shuffle_Block(in_channels, out_channels, stride=stride if i == 0 else 1))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # 添加新的Conv层
        x = self.additional_conv(x)

        # 添加新的全局平均池化层
        x = self.global_pooling(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 添加新的全连接层
        x = self.fc(x)

        return x



'''class Net(nn.Module):
    def __init__(self, num_classes=776, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32, 32, 2, False)
        # 32 64 32
        self.layer2 = make_layers(32, 64, 2, True)
        # 64 32 16
        self.layer3 = make_layers(64, 128, 2, True)
        # 128 16 8
        self.layer4 = make_layers(128, 256, 2, True)
        # 128 8 4
        self.layer5 = make_layers(256, 256, 1, False)

        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(256*8*4, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True)
        )
        # 256 1 1
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x'''

'''class Net(nn.Module):
    def __init__(self, num_classes=776, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32, 32, 2, False)
        # 32 64 32
        self.layer2 = make_layers(32, 64, 2, True)
        # 64 32 16
        self.layer3 = make_layers(64, 128, 2, True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True)
        )
        # 256 1 1
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x'''

if __name__ == '__main__':
    net = Net(reid=True)
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    import ipdb
    ipdb.set_trace()
