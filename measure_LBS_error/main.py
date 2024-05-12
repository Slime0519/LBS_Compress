import trimesh
import torch
import smplx
import numpy as np
import pickle
import sys
sys.path.append("/home/junmyeong/workspace/JM_LBSRecon")

from tqdm import tqdm
from measure_LBS_error.path import *
from measure_LBS_error.smplx_utils import *
from measure_LBS_error.metrics import *
from measure_LBS_error.lbs_autoencoder.inference import compress_lbs, decompress_lbs

def deform_mesh(vertices, lbs, full_pose, transl, scale_params, template_mesh):
    deformed_posed_vertices = deform_vertices(
            vertices=vertices,
            smpl_model=smplx_template,
            lbs=lbs,
            inverse=False,
            full_pose=full_pose,
            device="cuda")
     
    deformed_posed_vertices += transl
    deformed_mesh = template_mesh.copy()
    deformed_mesh.vertices = deformed_posed_vertices.detach().squeeze().cpu().numpy()
    deformed_mesh = smpl2real2(deformed_mesh, *scale_params)
    
    return deformed_mesh
    
def deform_test_for_single_mesh(mesh_name, pose_list, verbose=False):
    # initialize smpl-x model
    # compare gt / autoencoder output        
    mesh, lbs = load_mesh(mesh_name=mesh_name)
    
    smplx_output, smplx_model, smplx_params = get_smplx_model_for_mesh(mesh_name=mesh_name)
    scale_params = set_scale(smplx_model)
    centroid_scan, scale_scan, centroid_smpl, scale_smpl = scale_params
    
    canon_scan_tmp = real2smpl2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
    scan_canon_vertices = torch.FloatTensor(canon_scan_tmp.vertices).to("cuda")[None, ...]
    smplx_transl_tensor = torch.FloatTensor(smplx_params['transl']).to("cuda")

    # ------ test lbs compression and decompression ------
    # compress and decompress lbs
    compressed_lbs = compress_lbs(lbs)
    restored_lbs = decompress_lbs(compressed_lbs)
    
    if verbose:
        # measure mse and kl div loss
        mseloss = mse_loss(lbs, restored_lbs)
        kldiv = kl_divergence(lbs, restored_lbs)
        print(f"mse loss: {mseloss} / kl divergence: {kldiv}")
    # ------ end test lbs compression and decompression ------
    
    chamfer_dist_list = []
    
    for pose in tqdm(pose_list):
        full_pose = torch.zeros((55, 3))
        full_pose[1:22] = torch.tensor(pose.reshape(-1, 3).astype(np.float32))
        
        # deform mesh
        deformed_scan_mesh = deform_mesh(scan_canon_vertices, lbs, full_pose, smplx_transl_tensor, scale_params, mesh)
        deformed_scan_mesh_compressed = deform_mesh(scan_canon_vertices, restored_lbs, full_pose, smplx_transl_tensor, scale_params, mesh)
        
        chamfer_dist = chamfer_distance(deformed_scan_mesh, deformed_scan_mesh_compressed)[0]
        chamfer_dist_list.append(chamfer_dist.cpu().numpy())
    
    return chamfer_dist_list, np.mean(np.array(chamfer_dist_list))
    
def main():
    total_list = []
    chamfer_dict = dict()
    
    # mesh_list = MESH_NAME_LIST
    for mesh_name in MESH_NAME_LIST:
        # print(f"evaluation for {mesh_name}...")
        chamfer_list, chamfer_mean = deform_test_for_single_mesh(mesh_name, POSE_SET)
        total_list += chamfer_list
        chamfer_dict[mesh_name] = np.array(chamfer_list)
        
        print(f"chamfer mean for {mesh_name}: {chamfer_mean}")
        
    with open("chamfer_dict.pkl", "wb") as f:
        pickle.dump(chamfer_dict, f)
    
    total_list = np.array(total_list)
    np.save("total_chamfer_list.npy", total_list)
    
    print(f"total chamfer mean: {np.mean(np.array(total_list))}")
    
if __name__ == "__main__":
    main()