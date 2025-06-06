# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.nets.utils import load_checkpoint

momentum = 0.001


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride=1, num_classes=10, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.classifier = nn.Linear(channels[3], num_classes)

        self.channels = channels[3]
        self.num_features = channels[3]

        self.BNH = nn.BatchNorm2d(self.num_features)
        self.BNM = nn.BatchNorm2d(self.num_features)
        self.BNT = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # auxiliary classifier
        self.aux_classifier1 = nn.Linear(channels[3], num_classes)
        self.aux_classifier2 = nn.Linear(channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.classifier(x)

        out = self.extract(x)
        head_fs, medium_fs, tail_fs = self.BNH(out), self.BNM(out), self.BNT(out)
        fs = torch.cat((head_fs, medium_fs, tail_fs), dim=0)
        out = self.avgpool(fs).view(fs.size(0), -1)

        if only_feat:
            return out

        output = self.classifier(out)
        logitsH, logitsM, logitsT = output.chunk(3)
        logits = (logitsH + logitsM + logitsT) / 3


        aux_output1 = self.aux_classifier1(out)
        aux_logitsH1, aux_logitsM1, aux_logitsT1 = aux_output1.chunk(3)
        aux_logits1 = (aux_logitsH1 + aux_logitsM1 + aux_logitsT1) / 3

        aux_output2 = self.aux_classifier2(out)
        aux_logitsH2, aux_logitsM2, aux_logitsT2 = aux_output2.chunk(3)
        aux_logits2 = (aux_logitsH2 + aux_logitsM2 + aux_logitsT2) / 3

        result_dict = {'feat': out,
                       'logitsH': logitsH, 'logitsM': logitsM, 'logitsT': logitsT, 'logits': logits,
                       'aux_logitsH1': aux_logitsH1, 'aux_logitsM1': aux_logitsM1, 'aux_logitsT1': aux_logitsT1, 'aux_logits1': aux_logits1,
                       'aux_logitsH2': aux_logitsH2, 'aux_logitsM2': aux_logitsM2, 'aux_logitsT2': aux_logitsT2, 'aux_logits2': aux_logits2}

        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix),
                       blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(
                           prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

def wrn_28_2_cpe(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=2, **kwargs)
    if pretrained:
        print('!'*100)
        print("NOT RESUME TRAINING")
        print("LOADING {} EMA".format(pretrained_path))
        print("ALSO LOADING CLASSIFIER")
        print('!'*100)
        model = load_checkpoint(model, pretrained_path)
    return model


def wrn_28_8_cpe(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=8, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model