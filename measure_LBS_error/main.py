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
    
    centroid_real, scale_real = smplx_params["centroid_real"], smplx_params["scale_real"]
    canon_scan_tmp = real2smpl2(mesh, centroid_real, scale_real, centroid_smplx, scale_smplx)
    
    # smplx_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/smplx_mesh.obj")
    # centroid_scan, scale_scan, centroid_smplx, scale_smplx = set_scale(smplx_model)
    # print(centroid_scan, scale_scan, centroid_smplx, scale_smplx)

    
    smpl_canon_vertices = torch.FloatTensor(smplx_mesh.vertices).to("cuda")[None, ...] \
                              * smplx_params['scale'][0]
    # canon_scan_tmp = real2smpl2(mesh, centroid_scan, scale_scan, centroid_smplx, scale_smplx)
    canon_scan_mesh = trimesh.Trimesh(canon_scan_tmp.vertices, canon_scan_tmp.faces, process=False)
    canon_scan_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/canon_scan_mesh.obj")
    
    
    scan_canon_vertices = torch.FloatTensor(canon_scan_tmp.vertices).to("cuda")[None, ...]
    transl_tensor = torch.FloatTensor(smplx_params['transl']).to("cuda")

    
    smpl_deformed_vertices = deform_vertices(
        vertices=smpl_canon_vertices,
        smpl_model=smplx_template,
        lbs=smplx_template.lbs_weights,
        inverse=False,
        full_pose=full_pose ,
        device=device)
    
    smplx_mesh_deformed = trimesh.Trimesh(
        smpl_deformed_vertices.detach().squeeze().cpu().numpy(),
        smplx_mesh.faces, process=False
    )
    smplx_mesh_deformed.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/smplx_mesh_deformed.obj")
    
    smpl_deformed_vertices += transl_tensor
    smpl_deformed_mesh = trimesh.Trimesh(smpl_deformed_vertices.detach().squeeze().cpu().numpy(),
                                            smplx_mesh.faces, process=False)
    
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
    deformed_scan_mesh.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/deformed_scan_mesh.obj")
    # mesh_deformed.export("/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/mesh_deformed.obj")
    # mesh_deformed = mesh
    
    chamfer_dist = chamfer_distance(mesh_deformed, mesh)
    print(chamfer_dist)
    
    # ------ test lbs compression and decompression ------
    # compress and decompress lbs
    compressed_lbs = compress_lbs(lbs)
    resotred_lbs = decompress_lbs(compressed_lbs)
    
    # measure mse and kl div loss
    print(mse_loss(lbs, resotred_lbs))
    print(kl_divergence(lbs, resotred_lbs))
    # ------ end test lbs compression and decompression ------
    
    
    
if __name__ == "__main__":
    main()