import trimesh
import trimesh.proximity
import torch
import numpy as np
import pytorch3d

import pytorch3d.loss as pt3d_loss

def chamfer_distance(mesh1, mesh2):
    """
    :param mesh1: trimesh object
    :param mesh2: trimesh object
    :return: chamfer distance between mesh1 and mesh2
    """
    # return trimesh.proximity.closest_point(mesh1, mesh2.vertices)[1].mean() + trimesh.proximity.closest_point(mesh2, mesh1.vertices)[1].mean()
    
    mesh1 = torch.Tensor(mesh1.vertices.astype(np.float32)).unsqueeze(0).cuda()
    mesh2 = torch.Tensor(mesh2.vertices.astype(np.float32)).unsqueeze(0).cuda()
    return pt3d_loss.chamfer_distance(mesh1, mesh2)


def chamfer_and_p2s(gt, pred, num_samples=10000):
    pred_surf_pts, _ = trimesh.sample.sample_surface(pred, num_samples)
    gt_surf_pts, _ = trimesh.sample.sample_surface(gt, num_samples)

    _, pred_gt_dist, _ = trimesh.proximity.closest_point(gt, pred_surf_pts)
    _, gt_pred_dist, _ = trimesh.proximity.closest_point(pred, gt_surf_pts)

    pred_gt_dist[np.isnan(pred_gt_dist)] = 0
    gt_pred_dist[np.isnan(gt_pred_dist)] = 0

    false_ratio_pred_gt = len(pred_gt_dist[pred_gt_dist > 3.0]) / num_samples
    false_ratio_gt_pred = len(gt_pred_dist[gt_pred_dist > 3.0]) / num_samples
    false_ratio = (false_ratio_pred_gt + false_ratio_gt_pred) / 2 * 100

    pred_gt_dist = pred_gt_dist.mean()
    gt_pred_dist = gt_pred_dist.mean()

    p2s = pred_gt_dist
    p2s_outlier = false_ratio_pred_gt
    chamfer = (pred_gt_dist + gt_pred_dist) / 2
    chamfer_outlier = false_ratio

    return p2s, chamfer, p2s_outlier, chamfer_outlier
    
def p2s_distance(point, mesh):
    """
    :param point: torch.tensor (1, 3)
    :param mesh: trimesh object
    :return: distance between point and mesh
    """
    # return trimesh.proximity.closest_point(mesh, point).mean()
    return pytorch3d.loss(point.unsqueeze(0), mesh.vertices.unsqueeze(0))

def mse_loss(pred, gt):
    """
    :param pred: torch.tensor
    :param gt: torch.tensor
    :return: mean squared error loss between pred and gt
    """
    return torch.nn.MSELoss()(pred, gt)

def kl_divergence(pred, gt):
    """
    :param pred: torch.tensor
    :param gt: torch.tensor
    :return: kl divergence between pred and gt
    """
    return torch.nn.KLDivLoss()(pred, gt)


if __name__ == "__main__":
    mesh1= trimesh.load("measure_LBS_error/sample_RP_T/SMPLX/CANON/rp_adanna_rigged_003_FBX/rp_adanna_rigged_003_FBX.obj")
    mesh2= trimesh.load("measure_LBS_error/sample_RP_T/SMPLX/CANON_SMPLX_TEMP/rp_adanna_rigged_003_FBX/rp_adanna_rigged_003_FBX.obj")
    
    chamfer_dst = chamfer_distance(mesh1, mesh2)
    print(chamfer_dst)
    