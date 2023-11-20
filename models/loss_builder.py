import copy
import torch
import cv2
import numpy as np
import torchvision
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils.core.volume2depth import *
from utils.core import depth2volume
from utils.core import gradients
from utils.core.im_utils import get_plane_params
from utils.core.orthographic_warp import *
from utils.normal_gan import ops
from pytorch3d.loss import chamfer_distance
from LBS_unwrapping import utils
import random
import json
import pickle
import pdb

# for all networks
class LossBank:
    def __init__(self, device=None):  # set loss criteria and options
        super(LossBank).__init__()

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_huber = torch.nn.SmoothL1Loss()
        self.criterion_vgg = VGGPerceptualLoss()
        self.criterion_cos = torch.nn.CosineSimilarity(dim=1)
        self.criterion_ssim_ch1 = SSIM(data_range=1.0, size_average=True,
                                       nonnegative_ssim=True, channel=1, win_size=5)
        self.criterion_ssim_ch3 = SSIM(data_range=1.0, size_average=True,
                                       nonnegative_ssim=True, channel=3, win_size=5)

        if device is not None and torch.cuda.is_available():
            self.criterion_l1 = self.criterion_l1.to(device)
            self.criterion_l2 = self.criterion_l2.to(device)
            self.criterion_bce = self.criterion_bce.to(device)
            self.criterion_huber = self.criterion_huber.to(device)
            self.criterion_vgg = self.criterion_vgg.to(device)
            self.criterion_cos = self.criterion_cos.to(device)
            self.criterion_ssim_ch1 = self.criterion_ssim_ch1.to(device)
            self.criterion_ssim_ch3 = self.criterion_ssim_ch3.to(device)

    # l1 loss
    def get_l1_loss(self, pred, target):
        loss = self.criterion_l1(pred, target)
        return loss

    # Huber loss
    def get_huber_loss(self, pred, target):
        loss = self.criterion_huber(pred, target)
        return loss

    # l1 loss
    def get_l2_loss(self, pred, target):
        loss = self.criterion_l2(pred, target)
        return loss

    # binary cross entropy
    def get_bce_loss(self, pred, target):
        loss = self.criterion_bce(pred, target)
        return loss

    def get_smoothness_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def get_cosine_loss(self, pred, target):
        loss = 1 - self.criterion_cos(pred, target).mean()
        return loss

    def get_l1_gradient_loss(self, pred, target):
        pred_grad = gradients.stack_gradients(pred)
        target_grad = gradients.stack_gradients(target)
        return self.get_l1_loss(pred_grad, target_grad)

    def get_l2_gradient_loss(self, pred, target):
        pred_grad = gradients.stack_gradients(pred)
        target_grad = gradients.stack_gradients(target)
        return self.get_l2_loss(pred_grad, target_grad)

    def get_perceptual_loss(self, pred, target):
        loss = self.criterion_vgg(pred, target)
        return loss

    def get_ssim_loss(self, pred, target):
        if pred.shape[1] == 1:
            ssim_loss = 1 - self.criterion_ssim_ch1(pred, target)
        else:
            ssim_loss = 1 - self.criterion_ssim_ch3(pred, target)
        return ssim_loss

    def get_laplacian_loss(self, pred, target, sigma, weight=0.1):
        if weight == 1:
            return torch.mean(torch.abs(pred - target) / sigma + 0.05 * sigma)
        elif weight == 2:
            return torch.mean(torch.abs(pred - target) / sigma + torch.log(sigma + 3))
        elif weight == 3:
            return torch.mean(torch.abs(pred - target) / sigma + 0.05 * torch.sqrt(sigma))
        else:
            return torch.mean(torch.abs(pred - target) / sigma + (1 / 255.0) * sigma)

    def get_exist_loss(self, pred, target):
        return self.get_bce_loss(pred, target)

# added on May 10
# need to verify whether it works correctly
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def depth2pix(depth_map, res):
    h = w = res
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth_map)
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth_map)

    pixel_coords = torch.stack((x_range, y_range, depth_map), dim=0)
    pixel_coords_vec = pixel_coords.reshape(3, -1).int()

    return pixel_coords_vec


# for a specific model
class LossBuilderHuman(LossBank):
    def __init__(self, device=None, data_path='/jisu/dataset/', res=512, real_dist=300.0, weight_conf=1):
        LossBank.__init__(self, device=device)
        # self.loss_type = loss_type
        self.res = res
        self.real_dist=real_dist
        self.device = device
        self.data_path = data_path
        self.weight_conf = weight_conf
        self.build = self.build_loss

    # generator
    def build_loss(self, model, input_var, target_var, xy=None, config=None):
        # image_input, mask_input, shape_input, pose_input -> input_var
        # depth_output, lbs_output, posed_obj -> target_var
        # depth_pred, lbs_pred -> pred_var

        # prediction!!
        pred_var = model(torch.cat([input_var[0], input_var[1]], dim=1)) # to do .. put SMPLX params

        loss, pred_var, target_var = self.loss_exp(input_var, pred_var, target_var, xy=xy, config=config)

        return loss, pred_var, target_var

    def loss_exp(self, input_var, pred_var, target_var, xy=None, config=None):
        tgt_df, tgt_db = torch.Tensor.chunk(target_var[0], chunks=2, dim=1)

        tgt_pf = get_plane_params(z=tgt_df, xy=xy,
                                  pred_res=self.res, real_dist=self.real_dist,
                                  z_real=True, v_norm=True)
        tgt_pb = get_plane_params(z=tgt_db, xy=xy,
                                  pred_res=self.res, real_dist=self.real_dist,
                                  z_real=True, v_norm=True)
        tgt_normal = torch.cat([tgt_pf[:, 0:3, :, :], tgt_pb[:, 0:3, :, :]], dim=1)
        # normal maps are scaled by the scale factor (8.0), see function get_plane_params() for details
        # this is to better predict normal maps during training
        normal_scaler = 8.0
        tgt_normal = torch.chunk(tgt_normal * normal_scaler, chunks=tgt_normal.shape[1] // 3, dim=1)
        tgt_depth = torch.chunk(target_var[0], chunks=target_var[0].shape[1], dim=1)
        tgt_lbs = torch.chunk(target_var[1], chunks=target_var[1].shape[1] // 3, dim=1)

        loss = 0
        # balancing parameters
        w = []
        w.append([0.9, 0.1, 0.85, 0.15, 0.1, 0.5])  # stage 1
        # labels_ones = torch.ones_like(tgt_normal[0][:, 0:1, :, :])
        for k, options in enumerate(config):
            if 'normal' in options and 'pred_normal' in pred_var[k]:
                pred_normal = pred_var[k]['pred_normal'] * normal_scaler
                pred_normal = torch.chunk(pred_normal, chunks=pred_normal.shape[1] // 3, dim=1)
                if 'normal_l2' in options:
                    loss += w[k][0] * self.get_losses(pred_normal, tgt_normal, loss_type='l2')
                if 'normal_cos' in options:
                    loss += w[k][1] * self.get_losses(pred_normal, tgt_normal, loss_type='cos')

            if 'lbs' in options and 'pred_lbs' in pred_var[k]:
                pred_lbs = pred_var[k]['pred_lbs']
                pred_lbs = torch.chunk(pred_lbs, chunks=pred_lbs.shape[1] // 3, dim=1)
                if 'lbs_l2' in options:
                    loss += w[k][2] * self.get_losses(pred_lbs, tgt_lbs, loss_type='l2')
                if 'lbs_vgg' in options:
                    loss += w[k][3] * self.get_losses(pred_lbs, tgt_lbs, loss_type='vgg')

            if 'depth' in options and 'pred_depth' in pred_var[k]:
                pred_depth = pred_var[k]['pred_depth'] 
                pred_df, pred_db = torch.chunk(pred_depth, chunks=target_var[0].shape[1], dim=1)
                predfd2n = get_plane_params(z=pred_df, xy=xy,
                                            pred_res=self.res, real_dist=self.real_dist,
                                            z_real=True, v_norm=True)
                predbd2n = get_plane_params(z=pred_db, xy=xy,
                                            pred_res=self.res, real_dist=self.real_dist,
                                            z_real=True, v_norm=True)
                pred_depth2normal = torch.cat([predfd2n[:, 0:3, :, :],
                                               predbd2n[:, 0:3, :, :]], dim=1)
                pred_depth = torch.chunk(pred_depth, chunks=pred_depth.shape[1], dim=1)
                pred_depth2normal = torch.chunk(pred_depth2normal * normal_scaler,
                                                chunks=(pred_depth2normal.shape[1] // 3), dim=1)

                #loss += w[k][4] * self.get_losses((pred_df - predfd2n[:, 0:3, :, :]), tgt_pf[:, 3:4, :, :], loss_type='l2')
                #loss += w[k][4] * self.get_losses((pred_db - predbd2n[:, 0:3, :, :]), tgt_pb[:, 3:4, :, :], loss_type='l2')

                if 'depth_l2' in options:
                    loss += w[k][0] * self.get_losses(pred_depth, tgt_depth, loss_type='l2')
                if 'depth_ssim' in options:
                    loss += w[k][1] * self.get_losses(pred_depth, tgt_depth, loss_type='ssim')
                if 'depth2norm_l2' in options:
                    loss += w[k][2] * self.get_losses(pred_depth2normal, tgt_normal, loss_type='l2')
                if 'depth2norm_cos' in options:
                    loss += w[k][3] * self.get_losses(pred_depth2normal, tgt_normal, loss_type='cos')

            if 'lbsfix_depth' in options and 'pred_depth' in pred_var[k]:
                #mask = input_var[1] # 4x1x512x512
                #shape = input_var[2] # 4x10
                #pose = input_var[3] # 4x55x3
                (scale, center, K, R, t) = input_var[2]

                for i in range(len(input_var[1])):                        
                    v, u = torch.where(input_var[1][i][0]==1)
                    keypoints = torch.stack((u, v), dim=1)

                    tgt_depth_f = tgt_depth[0][i][0]
                    zf = ((tgt_depth_f - 0.5) * 128 + self.real_dist)[keypoints[:, 1], keypoints[:, 0]]
                    xf = (keypoints[:, 0] - K[i][0][2]) / K[i][0][0] * zf
                    yf = (keypoints[:, 1] - K[i][1][2]) / K[i][1][1] * zf

                    verts_f = torch.stack((xf, yf, zf), dim=1) # front image 3D vertice points

                    full_lbs_f = target_var[2][i][0][keypoints[:, 1], keypoints[:, 0]] # front image lbs

                    tgt_depth_b = tgt_depth[1][i][0]
                    zb = ((tgt_depth_b - 0.5) * 128 + self.real_dist)[keypoints[:, 1], keypoints[:, 0]]
                    xb = (keypoints[:, 0] - K[i][0][2]) / K[i][0][0] * zb
                    yb = (keypoints[:, 1] - K[i][1][2]) / K[i][1][1] * zb

                    verts_b = torch.stack((xb, yb, zb), dim=1) # back image 3D vertice points
                    full_lbs_b = target_var[2][i][1][keypoints[:, 1], keypoints[:, 0]] # back image lbs

                    full_lbs = torch.cat((full_lbs_f, full_lbs_b), dim=0)
                    verts = torch.cat((verts_f, verts_b), dim=0)
                    verts = torch.matmul((verts - t[i].to(self.device)), torch.linalg.inv(R[i]).to(self.device))
                    verts = verts / scale[i][0].to(self.device) + center[i].to(self.device)

                    with open(self.data_path + target_var[3][i], 'r') as f:
                        smpl_params = json.load(f)

                    # real space to smpl space
                    verts = (verts - torch.Tensor(smpl_params['centroid_real']).to(self.device)) * torch.Tensor([smpl_params['scale_real']]).to(self.device)[0]  \
                               / torch.Tensor([smpl_params['scale_smplx']]).to(self.device)[0] + torch.Tensor(smpl_params['centroid_smplx']).to(self.device)

                    #verts, _ = utils.write_ply('posed_temp.ply', verts.detach().cpu().numpy())
                    full_pose = torch.load((self.data_path + target_var[4][i]).replace('obj', 'pt'))

                    smpl_model_posed = utils.set_smplx_model_beta(smpl_params['betas'], full_pose, smpl_path=self.data_path, device=self.device) # posed smpl model
                    smpl_model_canon = utils.set_smplx_model_beta(smpl_params['betas'], smpl_path=self.data_path, device=self.device) #canonical smpl model

                    smpl_temp = trimesh.Trimesh(smpl_model_posed(return_verts=True).vertices.detach().cpu().numpy().squeeze(), smpl_model_posed.faces, process=False)
                    #smpl_temp.export('posed_smpl.obj')

                    # canonicalization
                    verts = utils.deform_vertices(torch.Tensor(verts).to(self.device).unsqueeze(0), smpl_model_posed, full_lbs.to(self.device), full_pose.to(self.device), inverse=True, return_vshape=False, device=self.device)
                    #pred_verts, _ = utils.write_ply('canon_pred.ply', verts[0].detach().cpu().numpy())

                    canon_gt = trimesh.load(self.data_path + target_var[5][i])
                    canon_gt = utils.real2smpl(canon_gt, trimesh.Trimesh(smpl_model_canon(return_verts=True).vertices.detach().cpu().numpy().squeeze(), smpl_model_canon.faces, process=False))

                    pdb.set_trace()
                    #gt_verts, _ = utils.write_ply('canon_gt.ply', canon_gt.vertices)

                    print('chamfer: ', chamfer_distance(verts, torch.Tensor(canon_gt.vertices).to(self.device).unsqueeze(0)))
  
                    # mesh = trimesh.load(self.data_path + target_var[3][0])
                    # print('chamfer: ', chamfer_distance(torch.Tensor(mesh.vertices).to(self.device).unsqueeze(0), verts.unsqueeze(0)))
                    # verts = verts.detach().cpu().numpy()
                    # verts = (verts) / scale[i][0].numpy() + center[i].numpy()
                    # verts, _ = write_ply('temp.ply', verts, rgbs=None, eps=1e-8)
                    pdb.set_trace()


        #pred_var = {'normal': pred_normal, 'depth2normal': pred_depth2normal,
        #            'depth': pred_depth, 'color': pred_color}
        pred_var = {'depth2normal': pred_depth2normal, 'depth': pred_depth, 'lbs': pred_lbs}
        target_var = {'normal': tgt_normal, 'depth': tgt_depth, 'lbs': tgt_lbs}
        return loss, pred_var, target_var

    # custom loss functions here.
    def get_loss(self, pred, target, loss_type='l1', sigma=None, weight=0.1):
        if loss_type == 'l1':
            loss = self.get_l1_loss(pred, target)
        elif loss_type == 'bce':
            loss = self.get_bce_loss(pred, target)
        elif loss_type == 'l2' or loss_type == 'mse':
            loss = self.get_l2_loss(pred, target)
        elif loss_type == 'grad':
            loss = self.get_l1_gradient_loss(pred, target)
        elif loss_type == 'vgg':
            loss = self.get_perceptual_loss(pred, target)
        elif loss_type == 'ssim':
            loss = self.get_ssim_loss(pred, target)
        elif loss_type == 'cos':
            loss = self.get_cosine_loss(pred, target)
        elif loss_type == 'seg':
            loss = self.get_exist_loss(pred, target)
        elif loss_type == 'sigma' and sigma is not None:  # laplacian loss.
            loss = self.get_laplacian_loss(pred, target, sigma, weight=weight)
        elif loss_type == 'smooth' or loss_type == 'smoothness':
            loss = self.get_smoothness_loss(pred, target)
        else:
            loss = self.get_l1_loss(pred, target)
        return loss

    def get_losses(self, pred, target, loss_type='l1', sigma=None, sigma_weight=0.1):
        loss = 0
        for i, p in enumerate(pred):
            if p is not None and target[i] is not None:
                if sigma is None:
                    loss += self.get_loss(p, target[i], loss_type=loss_type)
                else:
                    loss += self.get_loss(p, target[i], loss_type=loss_type, sigma=sigma[i], weight=sigma_weight)
        return loss

    def get_losses_gan(self, pred, label=1.0, loss_type='bce'):
        loss = 0
        for p in pred:
            loss += self.get_loss(p, torch.full_like(p, fill_value=label), loss_type=loss_type)

        return loss


if __name__ == '__main__':
    # arr = np.arange(1, 10)
    a = [1, 2, 3]
    b = [4, 5, 6]
    for k, p in enumerate(a):
        print(k)
    # print(arr)

