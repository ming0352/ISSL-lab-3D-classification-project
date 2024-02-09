import torch.nn as nn
import torch
import torch.nn.functional as F

class _swin_transformer(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(_swin_transformer, self).__init__()

        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)
            self.rot_classifier = nn.Linear(self.num_classes, 4)

    #             self.rot_classifier1 = nn.Linear(self.num_classes, 32)
    #             self.rot_classifier2 = nn.Linear(32, 16)
    #             self.rot_classifier3 = nn.Linear(16, 4)



    def forward(self, x, is_feat=False, rot=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x  # last layer feature

        xx = self.classifier(x)

        if (rot):
            #             xy1 = self.rot_classifier1(xx)
            #             xy2 = self.rot_classifier2(xy1)
            xy = self.rot_classifier(xx)
            return [f0, f1, f2, f3, feat], (xx, xy)

        if is_feat:
            return [f0, f1, f2, f3, feat], xx
        else:
            return xx


def resnet12_ssl(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model