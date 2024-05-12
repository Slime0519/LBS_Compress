
import os
import shutil

import trimesh
import torch
import smplx
import json
import pickle
import platform
import random
import numpy as np

# from utils.visualizer import show_meshes
from PIL import Image
from reconstructor.recon_utils.utils import deform_vertices# from human_animator.human_deform.human_body_prior.pose_sampler import ExposeSampler, VposerSampler
from apps.options import Configurator
from tqdm import tqdm


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
def real2smpl(mesh1, mesh2, get_transform=False):
    avg_height = 180.0
    vts1 = mesh1.vertices
    center1 = mesh1.bounding_box.centroid
    scale1 = 2.0 / np.max(mesh1.bounding_box.extents)
    # align mesh1 to mesh2 if mesh2 path provided
    center2 = mesh2.bounding_box.centroid
    scale2 = 2.0 / np.max(mesh2.bounding_box.extents)
    # center2 = np.zeros_like(center1)
    # scale2 = 1 / (avg_height / 2)

    if get_transform:
        return center1, scale1 / scale2, center2

    new_vertices = (vts1 - center1) * scale1 / scale2 + center2
    mesh1.vertices = new_vertices

    return mesh1
def smpl2real(mesh1, mesh2, get_transform=False):
    vts1 = mesh1.vertices
    center1 = mesh1.bounding_box.centroid
    scale1 = 2.0 / np.max(mesh1.bounding_box.extents)
    # align mesh1 to mesh2 if mesh2 path provided
    center2 = mesh2.bounding_box.centroid
    scale2 = 2.0 / np.max(mesh2.bounding_box.extents)

    if get_transform:
        return center1, scale1 / scale2, center2

    new_vertices = (vts1 - center1) * scale1 / scale2 + center2
    mesh1.vertices = new_vertices

    return mesh1
def real2smpl2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl):
    mesh.vertices = (mesh.vertices - centroid_scan) * scale_scan / scale_smpl + centroid_smpl
    return mesh
def smpl2real2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl):
    mesh.vertices = (mesh.vertices - centroid_smpl) * scale_smpl / scale_scan + centroid_scan
    return mesh
def load_gt_mesh(file, avg_height=180.0):
    m = trimesh.load_mesh(file, process=False)
    vertices = m.vertices
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1  # if (vmax - vmin).argmax() == 1 else 2
    center = np.median(vertices, 0)
    center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
    scale = avg_height / (vmax[up_axis] - vmin[up_axis])

    vertices -= center
    vertices *= scale

    return trimesh.Trimesh(vertices=vertices, faces=m.faces, visual=m.visual, process=False)
def param2dic(smpl_model, centroid_scan, scale_scan, centroid_smpl, scale_smpl):
    output = dict()
    output['global_orient'] = list(smpl_model.global_orient[0].
                                   detach().cpu().numpy().reshape(-1).astype(float))
    output['betas'] = list(smpl_model.betas[0]
                           .detach().cpu().numpy().reshape(-1).astype(float))
    output['body_pose'] = list(smpl_model.body_pose[0]
                               .detach().cpu().numpy().reshape(-1).astype(float))
    output['left_hand_pose'] = list(smpl_model.left_hand_pose[0]
                                    .detach().cpu().numpy().reshape(-1).astype(float))
    output['right_hand_pose'] = list(smpl_model.right_hand_pose[0]
                                     .detach().cpu().numpy().reshape(-1).astype(float))
    output['jaw_pose'] = list(smpl_model.jaw_pose[0]
                              .detach().cpu().numpy().reshape(-1).astype(float))
    output['expression'] = list(smpl_model.expression[0]
                                .detach().cpu().numpy().reshape(-1).astype(float))
    output['scale_real'] = scale_scan
    output['centroid_real'] = list(centroid_scan)
    output['scale_smplx'] = scale_smpl
    output['centroid_smplx'] = list(centroid_smpl)

    return output

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 0. Parse options
    config = Configurator()
    hostname = platform.node()
    if hostname == 'jumi' or hostname == 'mpark-ubuntu':
        dataset = 'RP_T'
        data_root = '/mnt/DATASET8T/home/jumi/DATASET/RP'
        data_root = os.path.join(data_root,  dataset)
        canon_scan_root = os.path.join(data_root, 'CANON_GT')
        canon_scan_posed_root = os.path.join(data_root, 'CANON_DEFORM_ALIGNED', '%s' % dataset, 'OBJ')
        smpl_gt_root = os.path.join(data_root, 'SMPLX_GT')
        smpl_posed_root = os.path.join(data_root, 'CANON_DEFORM_ALIGNED', '%s' % dataset, 'SMPLX')
    else:
        dataset = 'RP_T_DIFFUSION'
        data_root = '/media/jisu/DATASET/ECCV2024'
        data_root = os.path.join(data_root,  dataset)
        canon_scan_root = os.path.join(data_root, 'CANON_GT_TEMP')
        canon_scan_posed_root = os.path.join(data_root, 'CANON_DEFORM_ALIGNED_SCANIMATE', '%s' % dataset, 'OBJ')
        smpl_gt_root = os.path.join(data_root, 'SMPLX')
        smpl_posed_root = os.path.join(data_root, 'CANON_DEFORM_ALIGNED_SCANIMATE', '%s' % dataset, 'SMPLX')
    params = config.parse()
    params.device = 'cuda:0'

    # smpl parameter setting
    params.smpl_gender = 'male'
    params.smpl_type = 'smplx'
    params.smpl_path = params.smpl_path
    params.smpl_flat_hand = True
    smplx_template = smplx.create(model_path=params.smpl_path,
                                  model_type=params.smpl_type,
                                  gender=params.smpl_gender,
                                  num_betas=10, ext='npz',
                                  use_face_contour=True,
                                  flat_hand_mean=True,
                                  use_pca=True,
                                  num_pca_comps=12).to(params.device)

    # pose sample extraction
    # pose_dataset_path = os.path.join(data_root, 'pose_sample.npy')
    # pose_dataset = np.load(pose_dataset_path, allow_pickle=True)
    # pose_per_sample = 30

    pose_dataset_path = os.path.join(data_root, 'data/train/example_03375_shortlong/seqs')
    npz_files = sorted([f for f in os.listdir(pose_dataset_path) if f.endswith('.npz')])



    data_list = sorted(os.listdir(canon_scan_root))
    # data_list = data_list[3:]
    for i in data_list:
        # num_proc = random.randint(30, len(pose_dataset))
        # num_proc = min(num_proc, len(pose_dataset))
        # pose_sample_set = np.array_split(pose_dataset, num_proc)
        #
        # pose_sample_set = [pose for pose in pose_sample_set[0]]
        # pose_set = pose_dataset[np.random.choice(len(pose_dataset), pose_per_sample, replace=False)]

        pose_set = []
        trans_set = []
        for npz_file in npz_files:
            # Load the npz file data
            npz_data = np.load(os.path.join(pose_dataset_path, npz_file))
            pose = npz_data['pose']
            pose_set.append(pose[3:66])

        canon_scan_path = os.path.join(canon_scan_root, i)


        data_name = canon_scan_path.split('/')[-1]
        canon_scan_data = os.path.join(canon_scan_path, '%s.obj' % data_name)

        if not os.path.isfile(canon_scan_data):
            canon_scan_data = os.path.join(canon_scan_path, '0_0_00.obj')
        real_scale_smpl_data = os.path.join(smpl_gt_root, i, '%s.obj' % data_name)
        if not os.path.isfile(real_scale_smpl_data):
            real_scale_smpl_data = os.path.join(smpl_gt_root, i, 'smplx_mesh.obj')
        canon_scan_lbs_file = os.path.join(canon_scan_path, '%s.npy' % data_name)
        if not os.path.isfile(canon_scan_lbs_file):
            canon_scan_lbs_file = os.path.join(canon_scan_path, '%s.pkl' % data_name)
            if not os.path.isfile(canon_scan_lbs_file):
                canon_scan_data = os.path.join(canon_scan_path, '0_0_00.pkl')
                if not os.path.isfile(canon_scan_lbs_file):
                    print('scan_lbs file error ... ')

        smpl_param_path = os.path.join(smpl_gt_root, i, '%s.json' % data_name)
        if not os.path.isfile(smpl_param_path):
            # smpl_param_path = os.path.join(smpl_gt_root, i, 'smplx_mesh.json')
            smpl_param_path = os.path.join(smpl_gt_root, i, i + '.json')
        with open(smpl_param_path, 'r') as f:
            smpl_params = json.load(f)
        # checkpoint. smpl_output/model/mesh are in the smpl coordinate
        smpl_output, smpl_model = set_smpl_model(smplx_template, smpl_params, device=params.device)
        smpl_mesh = trimesh.Trimesh(
            smpl_output.vertices.detach().cpu().numpy().squeeze(),
            smplx_template.faces, process=False)

        v_pose_smpl = trimesh.Trimesh(smpl_model.v_template.cpu(),
                                      smpl_model.faces)
        centroid_smpl = v_pose_smpl.bounding_box.centroid
        scale_smpl = 2.0 / np.max(v_pose_smpl.bounding_box.extents)
        centroid_scan = np.zeros_like(centroid_smpl)
        scale_scan = 1 / (180.0 / 2)

        canon_smpl_vertices = smpl_model.v_template + smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
        canon_smpl_mesh = trimesh.Trimesh(canon_smpl_vertices.detach().squeeze(0).detach().cpu().numpy(),
                                          smpl_model.faces, process=False)

        if canon_scan_lbs_file[-3:] == 'npy':
            canon_scan_lbs = np.load(canon_scan_lbs_file, allow_pickle=True)
            canon_scan_lbs = canon_scan_lbs.item()
            # canon_scan_mesh = trimesh.Trimesh(canon_scan_lbs['vertices'], canon_scan_lbs['faces'],
            #                                   vertex_colors=canon_scan_lbs['color'], process=False)
            canon_scan_input = trimesh.load(canon_scan_data)

            canon_scan_mesh = trimesh.Trimesh(vertices=canon_scan_input.vertices,
                                              faces=canon_scan_input.faces, process=False)
            scan_lbs = torch.Tensor(canon_scan_lbs['lbs']).cuda()
        elif canon_scan_lbs_file[-3:] == 'pkl':
            with open(canon_scan_lbs_file, 'rb') as f:
                scan_lbs = pickle.load(f)
            scan_lbs = scan_lbs['lbs']

        # smpl gt to canonical pose
        scale_tensor = torch.FloatTensor(smpl_params['scale']).to(params.device)
        transl_tensor = torch.FloatTensor(smpl_params['transl']).to(params.device)
        smpl_canon_vertices = torch.FloatTensor(canon_smpl_mesh.vertices).to(params.device)[None, ...]
                              #* smpl_params['scale'][0])

        canon_scan_tmp = real2smpl2(canon_scan_mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
        scan_canon_vertices = (torch.FloatTensor(canon_scan_tmp.vertices).to(params.device)[None, ...])

        for j, pose in enumerate(tqdm(pose_set)):
            pose = torch.FloatTensor(pose).to(params.device)

            full_pose = torch.zeros((55, 3)).to(params.device)
            # full_pose[0:1, :] = torch.FloatTensor(smpl_params['global_orient']).reshape(1, 3).to(params.device)
            full_pose[1:22, :] = pose.reshape(21, 3).to(params.device)
            full_pose = full_pose[None, ...]

            smpl_deformed_vertices = deform_vertices(
                vertices=smpl_canon_vertices,
                smpl_model=smplx_template,
                lbs=smplx_template.lbs_weights,
                inverse=False,
                full_pose=full_pose,
                device=params.device)
            # smpl_deformed_vertices = (smpl_deformed_vertices + transl_tensor) #) * scale_tensor

            smpl_deformed_mesh = trimesh.Trimesh(smpl_deformed_vertices.detach().squeeze().cpu().numpy(),
                                                 smpl_mesh.faces)
            # scan canon-T to posed
            deformed_posed_vertices = deform_vertices(
                vertices=scan_canon_vertices, # / scale_tensor - transl_tensor,
                smpl_model=smplx_template,
                lbs=scan_lbs,
                inverse=False,
                full_pose=full_pose,
                device=params.device)
            # deformed_posed_vertices = (deformed_posed_vertices + transl_tensor) #) * scale_tensor
            deformed_scan_mesh = canon_scan_mesh.copy()
            deformed_scan_mesh.vertices = deformed_posed_vertices.detach().squeeze().cpu().numpy()


            # print(data_name, i)
            deformed_scan_mesh = smpl2real2(deformed_scan_mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
            smpl_deformed_mesh = smpl2real2(smpl_deformed_mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)

            deform_scan_save_path = os.path.join(canon_scan_posed_root, '%s_%02d' % (data_name, j))
            os.makedirs(deform_scan_save_path, exist_ok=True)

            deformed_scan_mesh.export(os.path.join(deform_scan_save_path, '%s_%02d.obj' % (data_name, j)))
            torch.save(torch.squeeze((full_pose).detach().cpu()),
                       os.path.join(deform_scan_save_path, '%s_%02d.pt' % (data_name, j)))

            deform_smpl_save_path = os.path.join(smpl_posed_root, '%s_%02d' % (data_name, j))
            os.makedirs(deform_smpl_save_path, exist_ok=True)
            deform_smpl_obj_save_path = os.path.join(deform_smpl_save_path, '%s_%02d.obj' % (data_name, j))
            smpl_deformed_mesh.export(deform_smpl_obj_save_path)

            output_dic = param2dic(smpl_output, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
            with open(os.path.join(deform_smpl_save_path, '%s_%02d.json' % (data_name, j)), 'w') as fp:
                json.dump(output_dic, fp, indent="\t")