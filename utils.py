import torch
import gradients
from pytorch_msssim import SSIM
criterion_ssim_ch1 = SSIM(data_range=1.0, size_average=True,
                               nonnegative_ssim=True, channel=1, win_size=5)
criterion_ssim_ch3 = SSIM(data_range=1.0, size_average=True,
                               nonnegative_ssim=True, channel=3, win_size=5)

def get_ssim_loss(pred, target):
    if pred.shape[1] == 1:
        ssim_loss = 1 - criterion_ssim_ch1(pred, target)
    else:
        ssim_loss = 1 - criterion_ssim_ch3(pred, target)
    return ssim_loss

# l1 loss
criterion_l1 = torch.nn.L1Loss()
def get_l1_loss( pred, target):
    loss = criterion_l1(pred, target)
    return loss

def get_l2_loss( pred, target):
    loss = torch.nn.MSELoss()(pred, target)
    return loss

def get_l1_gradient_loss(pred, target):
    pred_grad = gradients.stack_gradients(pred)
    target_grad = gradients.stack_gradients(target)
    return get_l1_loss(pred_grad, target_grad)

def get_l2_gradient_loss(pred, target):
    pred_grad = gradients.stack_gradients(pred)
    target_grad = gradients.stack_gradients(target)
    return get_l2_loss(pred_grad, target_grad)