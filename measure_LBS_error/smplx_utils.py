
import trimesh
import smplx
import torch
import json

import numpy as np
import torch.nn.functional as F

from measure_LBS_error.path import *


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    if inverse is True:
        posed_joints = torch.unsqueeze(posed_joints, dim=-1)
        rel_joints = posed_joints.clone()
        rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
        # rot_inv = torch.transpose(rot_mats.view(-1, 3, 3), dim0=1, dim1=2)
        transforms_mat_inv = transform_mat(
            rot_mats.view(-1, 3, 3),
            torch.zeros_like(rel_joints.view(-1, 3, 1))).view(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat_inv[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat_inv[:, i])
            transform_chain.append(curr_res)

        for i in range(len(transform_chain)):
            transform_chain[i] = torch.inverse(transform_chain[i])
            transform_chain[i][:, :3, 3] = joints[:, i, :, :].view(-1, 3)

        transforms = torch.stack(transform_chain, dim=1)
        joints_homogen = F.pad(posed_joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
        return posed_joints, rel_transforms
    else:
        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms
    
    
# def set_smpl_model(model_folder, device="cuda"):
#     """
#         create smpl-x instance
#         :return: a smpl instance
#     """
#     return smplx.create(model_path=model_folder,
#                         model_type='smplx',
#                         gender='male',
#                         num_betas=10, ext='npz',
#                         use_face_contour=True,
#                         flat_hand_mean=True,
#                         use_pca=False,
#                         ).to(device)
    
def set_smpl_model(smpl_model, smpl_params, device):
    smpl_model.betas = torch.nn.Parameter(torch.tensor([smpl_params['betas']], device=device))
    smpl_model.transl = torch.nn.Parameter(torch.tensor([smpl_params['transl']], device=device))
    smpl_model.expression = torch.nn.Parameter(torch.tensor([smpl_params['expression']], device=device))
    smpl_model.body_pose = torch.nn.Parameter(torch.tensor([smpl_params['body_pose']], device=device))
    smpl_model.global_orient = torch.nn.Parameter(torch.tensor([smpl_params['global_orient']], device=device))
    smpl_model.jaw_pose = torch.nn.Parameter(torch.tensor([smpl_params['jaw_pose']], device=device))
    smpl_model.left_hand_pose = torch.nn.Parameter(torch.tensor([smpl_params['left_hand_pose']], device=device))
    smpl_model.right_hand_pose = torch.nn.Parameter(torch.tensor([smpl_params['right_hand_pose']], device=device))
    smpl_mesh = smpl_model(return_verts=True, return_full_pose=True)
    smpl_mesh.joints = smpl_mesh.joints * torch.nn.Parameter(
        torch.tensor([smpl_params['scale']],
                     device=device))
    smpl_mesh.vertices = smpl_mesh.vertices * torch.nn.Parameter(
        torch.tensor([smpl_params['scale']],
                     device=device))
    return smpl_mesh, smpl_model
 
def get_smpl_model(smpl_params, smpl_model):
    # return None if self.smpl_model is None:
    return smpl_model(transl=smpl_params['transl'],
                            betas=smpl_params['betas'],
                            body_pose=smpl_params['body_pose'],
                            # global_orient=smpl_params['global_orient'],
                            jaw_pose=smpl_params['jaw_pose'],
                            #joints=smpl_params['joints'],
                            expression=smpl_params['expression'],
                            left_hand_pose=smpl_params['left_hand_pose'],
                            right_hand_pose=smpl_params['right_hand_pose'],
                            return_verts=True)

def set_smplx_model(smpl_json, device="cuda:0"):
    with open(smpl_json, 'r') as f:
        smpl_model = json.load(f)
    smpl_model_template = set_smpl_model(SMPLX_MODEL_PATH)
    
    # mesh = trimesh.load(mesh_path)
    smpl_model_template.betas = torch.nn.Parameter(torch.tensor([smpl_model['betas']], device=device))
    
    return smpl_model_template

def deform_vertices(vertices, smpl_model, lbs, full_pose, inverse=False, return_vshape=False, device='cuda:0'):
    v_shaped = smpl_model.v_template + \
               smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    # do not use smpl_model.joints -> it fails (don't know why)
    joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats.to(device), joints[:, :55, :].to(device), smpl_model.parents,
                                             inverse=inverse, dtype=torch.float32)

    weights = lbs.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(weights, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)
    homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32).to(device)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)

    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    if return_vshape:
        return verts, v_shaped
    else:
        return verts
    
def load_mesh(mesh_name, lbs=True):
    mesh_dict_path = get_path_dict(mesh_name)["GT_CANON"] / f"{mesh_name}.npy"
    mesh_path = get_path_dict(mesh_name)["GT_CANON"] / f"{mesh_name}.obj"
    mesh_dict = np.load(mesh_dict_path, allow_pickle=True).item()

    # load mesh
    # mesh = trimesh.Trimesh(vertices=mesh_dict["vertices"], faces=mesh_dict["faces"])
    mesh = trimesh.load(mesh_path)
    lbs = mesh_dict["lbs"]
    print(mesh.vertices.shape)
    print(lbs.shape)
    
    return mesh, lbs

smpl_gender = 'male'
smpl_type = 'smplx'
smpl_flat_hand = True
device = "cuda"
smplx_template = smplx.create(model_path=SMPLX_MODEL_PATH,
                                model_type=smpl_type,
                                gender=smpl_gender,
                                num_betas=10, ext='npz',
                                use_face_contour=True,
                                flat_hand_mean=True,
                                use_pca=True,
                                num_pca_comps=12).to(device)

def get_smplx_model_for_mesh(mesh_name):
    # load json filepath
    smplx_json_path = get_path_dict(mesh_name)["SMPLX_CANON"] / f"{mesh_name}.json"
    
    # load smplx model
    with open(smplx_json_path, 'r') as f:
        smpl_params = json.load(f)
    # smplx_model = set_smplx_model(smplx_json_path)
    
    smpl_output, smpl_model = set_smpl_model(smplx_template, smpl_params, device=device)
    
    return smpl_output, smpl_model, smpl_params

def set_scale(smplx_model):
    v_pose_smpl = trimesh.Trimesh(smplx_model.v_template.cpu(),
                                    smplx_model.faces)
    centroid_smpl = v_pose_smpl.bounding_box.centroid
    scale_smpl = 2.0 / np.max(v_pose_smpl.bounding_box.extents)
    centroid_scan = np.zeros_like(centroid_smpl)
    scale_scan = 1 / (180.0 / 2)
    
    return centroid_scan, scale_scan, centroid_smpl, scale_smpl
    
def real2smpl2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl):
    mesh.vertices = (mesh.vertices - centroid_scan) * scale_scan / scale_smpl + centroid_smpl
    return mesh

def smpl2real2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl):
    mesh.vertices = (mesh.vertices - centroid_smpl) * scale_smpl / scale_scan + centroid_scan
    return mesh

    