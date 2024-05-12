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
    