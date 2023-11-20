from models.base_blocks_dev import *
from models.vgg16 import Vgg16
import numpy as np


# networks for DeepHuman
# rules for encoders:
# 1. batch norms only.
# 2. relu for all layers except the last layer (Tanh)
# 3. no fully connect layers
# 4. no pooling (stride only)
# 5. how about resblocks, bottleneck, skip connections?


class ImageEncoderSimple(nn.Module):
    def __init__(self, ninput):
        torch.cuda.set_device (0)
        super (ImageEncoderSimple, self).__init__ ()

        self.conv_init = ConvBlock(ninput, 32, kernel=7, stride=1)
        self.conv1 = ConvBlock (32, 64, kernel=3, stride=2, norm='batch', activate='relu')
        self.conv2 = ConvBlock (64, 128, kernel=3, stride=2, norm='batch', activate='relu')

        self.res_blocks = nn.Sequential(
            ResBlockF(128, 128, kernel=3, stride=1),
            ResBlockF(128, 128, kernel=3, stride=1),
            ResBlockF(128, 128, kernel=3, stride=1),
            ResBlockF(128, 128, kernel=3, stride=1))

        self.conv3 = ConvBlock (128, 256, kernel=3, stride=2, norm='batch', activate='relu')
        self.conv4 = ConvBlock (256, 512, kernel=3, stride=2, norm='batch', activate='relu')

    def forward(self, input):
        output = self.conv_init(input)
        output1 = self.conv1 (output)
        output2 = self.conv2 (output1)
        output3 = self.res_blocks(output2)
        output4 = self.conv3 (output3)
        output5 = self.conv4 (output4)
        return output5


class ImageEncoderVGG(nn.Module):
    def __init__(self, ninput):
        torch.cuda.set_device (0)
        super (ImageEncoderVGG, self).__init__ ()

        self.image_encoder = Vgg16(use_pretrained=True)
        self.latent_encoder = nn.ModuleList()
        self.latent_encoder += ConvBlock(ninput, 32, kernel=7, stride=1)
        self.latent_encoder += ConvBlock (32, 64, kernel=3, stride=2, norm='batch', activate='relu')
        self.latent_encoder += ConvBlock (64, 128, kernel=3, stride=2, norm='batch', activate='relu')
        self.latent_encoder += ResBlockF(128, 128, kernel=3, stride=1)

        self.conv3 = ConvBlock (128, 256, kernel=3, stride=2, norm='batch', activate='relu')
        self.conv4 = ConvBlock (256, 512, kernel=3, stride=2, norm='batch', activate='relu')

    def freeze(self):
        for m in self.latent_encoder():
            for param in m.parameters ():
                param.requires_grad = False

    def defrost(self):
        for m in self.latent_encoder():
            for param in m.parameters ():
                param.requires_grad = True

    def forward(self, input):
        output = self.conv_init(input)
        output1 = self.conv1 (output)
        output2 = self.conv2 (output1)
        output3 = self.res_blocks(output2)
        output4 = self.conv3 (output3)
        output5 = self.conv4 (output4)
        return output5


class ImageDiscriminator(nn.Module):
    def __init__(self, ninput):
        super (ImageDiscriminator, self).__init__ ()
        self.conv_init = ConvBlock(ninput, 32, kernel=7, stride=1, norm='batch', activate='leaky', inplace=False)

        self.conv1 = nn.Sequential (
            ConvBlock (32, 32, kernel=3, stride=2, norm='batch', activate='leaky', inplace=False),
            ConvBlock (32, 32, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False),
            ConvBlock (32, 64, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False)
        )
        self.conv2 = nn.Sequential (
            ConvBlock (64, 64, kernel=3, stride=2, norm='batch', activate='leaky', inplace=False),
            ConvBlock (64, 64, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False),
            ConvBlock (64, 128, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False)
        )
        self.conv3 = nn.Sequential (
            ConvBlock (128, 128, kernel=3, stride=2, norm='batch', activate='leaky', inplace=False),
            ConvBlock (128, 128, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False),
            ConvBlock (128, 256, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False)
        )
        self.conv4 = nn.Sequential (
            ConvBlock (256, 512, kernel=3, stride=2, norm='batch', activate='leaky', inplace=False),
            ConvBlock (512, 512, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False)
        )
        self.conv5 = nn.Sequential (
            ConvBlock (512, 1024, kernel=3, stride=2, norm='batch', activate='leaky', inplace=False),
            ConvBlock (1024, 1024, kernel=3, stride=1, norm='batch', activate='leaky', inplace=False)
        )
        self.conv_last = ConvBlock(1024, 1, kernel=1, stride=1, inplace=False)

    def forward(self, input):
        output1 = self.conv_init(input)
        output2 = self.conv1 (output1)  # down
        output3 = self.conv2 (output2)  # down
        output4 = self.conv3 (output3)  # down
        output5 = self.conv4 (output4)  # down
        output6 = self.conv5 (output5)  # down
        output7 = self.conv_last (output6)  # channel pooling
        return output7
