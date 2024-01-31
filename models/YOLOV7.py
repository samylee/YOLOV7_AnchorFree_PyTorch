import torch
import torch.nn as nn

from models.common import BasicConv, MP, ELAN, ELANW, SPPCSPC, IV6Detect


class YOLOV7(nn.Module):
    def __init__(self, C=80, deploy=True):
        super(YOLOV7, self).__init__()
        in_channels = 3

        # backbone
        self.conv1 = BasicConv(in_channels, 32, 3, 1)
        self.conv2 = BasicConv(32, 64, 3, 2)
        self.conv3 = BasicConv(64, 64, 3, 1)
        self.conv4 = BasicConv(64, 128, 3, 2)

        self.elan1 = ELAN(128, 256)
        self.mp1 = MP(256, 128)
        self.elan2 = ELAN(256, 512)
        self.mp2 = MP(512, 256)
        self.elan3 = ELAN(512, 1024)
        self.mp3 = MP(1024, 512)
        self.elan4 = ELAN(1024, 1024)

        # head
        self.sppcspc = SPPCSPC(1024, 512)

        self.conv5 = BasicConv(512, 256, 1, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = BasicConv(1024, 256, 1, 1)

        self.elanw1 = ELANW(512, 256)

        self.conv7 = BasicConv(256, 128, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = BasicConv(512, 128, 1, 1)

        self.elanw2 = ELANW(256, 128)
        self.mp4 = MP(128, 128)
        self.elanw3 = ELANW(512, 256)
        self.mp5 = MP(256, 256)
        self.elanw4 = ELANW(1024, 512)

        self.conv9 = BasicConv(128, 256, 3, 1)
        self.conv10 = BasicConv(256, 512, 3, 1)
        self.conv11 = BasicConv(512, 1024, 3, 1)

        self.detect = IV6Detect(nc=C, ch=(256, 512, 1024), deploy=deploy)

    def forward(self, x):
        # backbone
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))

        x = self.elan1(x)
        x_24 = self.elan2(torch.cat(self.mp1(x), dim=1))
        x_37 = self.elan3(torch.cat(self.mp2(x_24), dim=1))
        x = self.elan4(torch.cat(self.mp3(x_37), dim=1))

        # head
        x_51 = self.sppcspc(x)

        x = self.up1(self.conv5(x_51))
        x_37 = self.conv6(x_37)
        x = torch.cat([x_37, x], dim=1)

        x_63 = self.elanw1(x)

        x = self.up2(self.conv7(x_63))
        x_24 = self.conv8(x_24)
        x = torch.cat([x_24, x], dim=1)

        x_75 = self.elanw2(x)
        x_88 = self.elanw3(torch.cat(self.mp4(x_75) + [x_63], dim=1))
        x_101 = self.elanw4(torch.cat(self.mp5(x_88) + [x_51], dim=1))

        x_75 = self.conv9(x_75)
        x_88 = self.conv10(x_88)
        x_101 = self.conv11(x_101)

        y = self.detect([x_75, x_88, x_101])

        return y

