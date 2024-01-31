import math
import torch
import torch.nn as nn

from utils.utils import dist2bbox, make_anchors


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='silu'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act == 'silu' else nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MP, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv3 = BasicConv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        out1 = self.conv1(self.mp(x))
        out2 = self.conv3(self.conv2(x))
        return [out2, out1]


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv3 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv5 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(mid_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv4(self.conv3(out2))
        out4 = self.conv6(self.conv5(out3))
        return self.conv7(torch.cat([out4, out3, out2, out1], dim=1))


class ELANW(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELANW, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv3 = BasicConv(out_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv5 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        return self.conv7(torch.cat([out6, out5, out4, out3, out2, out1], dim=1))


class SPPCSPC(nn.Module):
    def __init__(self, in_channels, out_channels, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        mid_channels = int(2 * out_channels * e)
        self.conv1 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv3 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 1, 1)
        self.mps = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv5 = BasicConv(mid_channels * 4, mid_channels, 1, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(mid_channels * 2, out_channels, 1, 1)

    def forward(self, x):
        x_tmp = self.conv4(self.conv3(self.conv1(x)))
        out1 = self.conv6(self.conv5(torch.cat([x_tmp] + [mp(x_tmp) for mp in self.mps], dim=1)))
        out2 = self.conv2(x)
        return self.conv7(torch.cat((out1, out2), dim=1))


class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self, x):
        return self.implicit * x


class IV6Detect(nn.Module):
    def __init__(self, nc=80, ch=(), deploy=True):
        super(IV6Detect, self).__init__()
        self.nc = nc
        self.deploy = deploy
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([ 8., 16., 32.])

        # Decouple
        c2, c3 = max(ch[0] // 4, 16), max(ch[0], self.no - 4)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(BasicConv(x, c2, 3), BasicConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(BasicConv(x, c3, 3), BasicConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max)

        self.ia2 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.ia3 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im2 = nn.ModuleList(ImplicitM(4 * self.reg_max) for _ in ch)
        self.im3 = nn.ModuleList(ImplicitM(self.nc) for _ in ch)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.im2[i](self.cv2[i](self.ia2[i](x[i]))), self.im3[i](self.cv3[i](self.ia3[i](x[i])))), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        if not self.deploy:
            return x, box, cls

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True