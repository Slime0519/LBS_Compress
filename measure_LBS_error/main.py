import trimesh
import torch
import smplx
import numpy as np

import sys
sys.path.append("/home/junmyeong/workspace/JM_LBSRecon")

from measure_LBS_error.path import *
from measure_LBS_error.smplx_utils import *
from measure_LBS_error.metrics import *
from measure_LBS_error.lbs_autoencoder.inference import compress_lbs, decompress_lbs

def main():
    # initialize smpl-x model
    # compare gt / autoencoder output        
    mesh, lbs = load_mesh(MESH_NAME_LIST[0])
    
    pose = POSE_SET[0]
    full_pose = torch.zeros((55, 3))
    full_pose[1:22] = torch.tensor(pose.reshape(-1, 3).astype(np.float32))
    print(pose.shape)
    
    smplx_output, smplx_model, smplx_params = get_smplx_model_for_mesh(MESH_NAME_LIST[0])
    
    smplx_mesh = trimesh.Trimesh(
        smplx_output.vertices.detach().cpu().numpy().squeeze(),
        smplx_template.faces, process=False
    )
    
    v_pose_smpl = trimesh.Trimesh(smplx_model.v_template.cpu(),
                                      smplx_model.faces)
    centroid_smpl = v_pose_smpl.bounding_box.centroid
    scale_smpl = 2.0 / np.max(v_pose_smpl.bounding_box.extents)
    centroid_scan = np.zeros_like(centroid_smpl)
    scale_scan = 1 / (180.0 / 2)
    
    
    canon_smpl_vertices = smplx_model.v_template + smplx.lbs.blend_shapes(smplx_model.betas, smplx_model.shapedirs)
    canon_smpl_mesh = trimesh.Trimesh(canon_smpl_vertices.detach().squeeze(0).detach().cpu().numpy(),
                                          smplx_model.faces, process=False)

    # smplx_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/smplx_mesh.obj")
    # centroid_scan, scale_scan, centroid_smplx, scale_smplx = set_scale(smplx_model)
    # print(centroid_scan, scale_scan, centroid_smplx, scale_smplx)

    scale_tensor = torch.FloatTensor(smplx_params['scale']).to(device)
    transl_tensor = torch.FloatTensor(smplx_params['transl']).to(device)
    smpl_canon_vertices = torch.FloatTensor(canon_smpl_mesh.vertices).to(device)[None, ...]

    # canon_scan_tmp = real2smpl2(mesh, centroid_scan, scale_scan, centroid_smplx, scale_smplx)
    # canon_scan_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/canon_scan_mesh.obj")

    canon_scan_tmp = real2smpl2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
    
    scan_canon_vertices = torch.FloatTensor(canon_scan_tmp.vertices).to("cuda")[None, ...]
    transl_tensor = torch.FloatTensor(smplx_params['transl']).to("cuda")


    
    # ------ test lbs compression and decompression ------
    # compress and decompress lbs
    compressed_lbs = compress_lbs(lbs)
    restored_lbs = decompress_lbs(compressed_lbs)
    
    # measure mse and kl div loss
    print(mse_loss(lbs, restored_lbs))
    print(kl_divergence(lbs, restored_lbs))
    # ------ end test lbs compression and decompression ------
    
    
    deformed_posed_vertices = deform_vertices(
            vertices=scan_canon_vertices,
            smpl_model=smplx_template,
            lbs=lbs,
            inverse=False,
            full_pose=full_pose,
            device="cuda")
    deformed_posed_vertices += transl_tensor
    deformed_scan_mesh = mesh.copy()
    deformed_scan_mesh.vertices = deformed_posed_vertices.detach().squeeze().cpu().numpy()
    deformed_scan_mesh = smpl2real2(deformed_scan_mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
    deformed_scan_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/deformed_scan_mesh.obj")
    
    
    
    deformed_posed_vertices_compressed = deform_vertices(
            vertices=scan_canon_vertices,
            smpl_model=smplx_template,
            lbs=restored_lbs,
            inverse=False,
            full_pose=full_pose,
            device="cuda")
    deformed_posed_vertices_compressed += transl_tensor
    deformed_scan_mesh_compressed = mesh.copy()
    deformed_scan_mesh_compressed.vertices = deformed_posed_vertices_compressed.detach().squeeze().cpu().numpy()
    deformed_posed_vertices_compressed = smpl2real2(deformed_scan_mesh_compressed, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
    deformed_scan_mesh_compressed.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/deformed_scan_mesh_compressed.obj")
    
    chamfer_dist = chamfer_distance(deformed_scan_mesh, deformed_scan_mesh_compressed)
    chamfer_dist = chamfer_distance(deformed_scan_mesh, deformed_scan_mesh)
    print(chamfer_dist)
    
if __name__ == "__main__":
    main()