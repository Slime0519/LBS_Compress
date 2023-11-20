from __future__ import print_function
import torch.utils.data
from models.unet_attention import *
from utils.core.depth2volume import *
from utils.core.im_utils import get_plane_params
from models.deep_human_modules import *

# ###### DeepHumanNet with four types of networks. ######### #
# A-C predict the SDF volume directly, from the input
# Type A: Takes input as the form of images (CSDS, CSDD, CSNS, CSND) (2VF)
# Type B: Same as Type A except the decoder outputs multiple volumes (3D Conv.) (2VB)
# Type C: Takes input as the form of volumes (CSDDV, CSDSV) (2VF)

class BaseModule(nn.Module):
    def __init__(self, split_last=True):
        super(BaseModule, self).__init__()
        self.split_last = split_last
        self.img2normal = ATUNet(in_ch=12, out_ch=6,
                                 split_last=self.split_last)
        self.img2img = ATUNet(in_ch=18, out_ch=6,
                              split_last=self.split_last)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        y_n, f_n = self.img2normal(x)
        y_c, f_c = self.img2img(torch.cat((x, y_n), dim=1))
        return {'pred_color': y_c,
                'pred_normal': y_n,
                'pred_color_feature': f_c,
                'pred_normal_feature': f_n}

class DepthModuleSingle(nn.Module):
    def __init__(self, split_last=True):
        super(DepthModuleSingle, self).__init__()
        self.split_last = split_last
        self.upsample = False
        self.cn2d = ATUNet(in_ch=64, out_ch=2, split_last=self.split_last)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        if self.upsample:
            x = self.up(x)
        pred_d, _ = self.cn2d(x)
        return {'pred_depth': pred_d}

class DeepHumanNet_Famoz(nn.Module):
    def __init__(self, split_last=True):
        super(DeepHumanNet_Famoz, self).__init__()
        self.f2b = BaseModule(split_last=split_last)
        self.cn2d = DepthModuleSingle(split_last=split_last)
    def forward(self, x):
        pred_var = list()
        pred_var.append(self.f2b.forward(x))

        f_c = pred_var[0]['pred_color_feature']  # ch : 12
        f_n = pred_var[0]['pred_normal_feature']  # ch : 12
        pred_var.append(self.cn2d.forward(torch.cat([f_c, f_n], dim=1)))
        return pred_var

def weight_init_basic(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# for test
if __name__ == '__main__':
    # input = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    input = Variable(torch.randn(4, 2, 5, 5)).float().cuda()
    _, b = torch.Tensor.chunk(input, chunks=2, dim=1)
    print(b.shape)

    print(b)
    print(input)

    # target = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    # print(len(input.shape))

    # tmp = [1, 2, 3]
    # print(tmp[-1])
    # model = Hourglass(4, '2D')
    # model = DeepHumanNet_A()
    # print(model.modules)
    # model.freeze()
    # model = VolumeDecoder (3, 128)
    # output = model.forward(input, target)

    # output, pre, post = model(input, None, None)
    # print((output - target))