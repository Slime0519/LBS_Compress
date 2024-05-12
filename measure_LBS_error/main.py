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
from measure_LBS_error.lbs_autoencoder.utils import *


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
    
def deform_test_for_single_mesh(mesh_name, pose_list, lbs_flatten=False, verbose=False):
    # initialize smpl-x model
    # compare gt / autoencoder output        
    mesh, lbs = load_mesh(mesh_name=mesh_name)
    
    
    smplx_output, smplx_model, smplx_params = get_smplx_model_for_mesh(mesh_name=mesh_name)
    scale_params = set_scale(smplx_model)
    centroid_scan, scale_scan, centroid_smpl, scale_smpl = scale_params
    
    canon_scan_tmp = real2smpl2(mesh, centroid_scan, scale_scan, centroid_smpl, scale_smpl)
    # eyeremoved_mesh = postprocess_mesh(canon_scan_tmp, num_faces=10000)
    # eyeremoved_mesh.export(f"/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/eyeremoved_mesh_smplx.obj")
    # exit()
    canon_smplx_vertices = smplx_model.v_template + smplx.lbs.blend_shapes(smplx_model.betas, smplx_model.shapedirs)
    canon_smplx_mesh = trimesh.Trimesh(canon_smplx_vertices.detach().squeeze(0).detach().cpu().numpy(),
                                          smplx_model.faces, process=False)

    if lbs_flatten:
        lbs = flatten_lbs(canon_smplx_mesh, canon_scan_tmp.vertices, lbs)
    
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
    
    p2s_dist_list = []
    p2s_outlier_list = []
    chamfer_dist_list = []
    chamfer_outlier_list = []
    
    for pose in tqdm(pose_list):
        full_pose = torch.zeros((55, 3))
        full_pose[1:22] = torch.tensor(pose.reshape(-1, 3).astype(np.float32))
        
        # deform mesh
        deformed_scan_mesh = deform_mesh(scan_canon_vertices, lbs, full_pose, smplx_transl_tensor, scale_params, mesh)
        deformed_scan_mesh_compressed = deform_mesh(scan_canon_vertices, restored_lbs, full_pose, smplx_transl_tensor, scale_params, mesh)
        
        # chamfer_dist = chamfer_distance(deformed_scan_mesh, deformed_scan_mesh_compressed)[0]
        p2s, chamfer, p2s_out, chamfer_out = chamfer_and_p2s(deformed_scan_mesh, deformed_scan_mesh_compressed)
        chamfer_dist_list.append(chamfer)
        p2s_dist_list.append(p2s)
        p2s_outlier_list.append(p2s_out)
        chamfer_outlier_list.append(chamfer_out)
    
    return p2s_dist_list, chamfer_dist_list, p2s_outlier_list, chamfer_outlier_list
    
def main():
    total_chamfer_list = []
    total_p2s_list = []
    total_p2s_out_list = []
    total_chamfer_out_list = []
    
    chamfer_dict = dict()
    
    for mesh_name in MESH_NAME_LIST:
        # print(f"evaluation for {mesh_name}...")
        p2s_list, chamfer_list, p2s_out_list, chamfer_out_list  = deform_test_for_single_mesh(mesh_name, POSE_SET[:10], lbs_flatten=True, verbose=False)
        
        chamfer_dict[mesh_name] = dict()
        chamfer_dict[mesh_name]["chamfer"] = np.array(chamfer_list)
        chamfer_dict[mesh_name]["p2s"] = np.array(p2s_list)
        chamfer_dict[mesh_name]["p2s_out"] = np.array(p2s_out_list)
        chamfer_dict[mesh_name]["chamfer_out"] = np.array(chamfer_out_list)
        
        total_chamfer_list.extend(chamfer_list)
        total_p2s_list.extend(p2s_list)
        total_p2s_out_list.extend(p2s_out_list)
        total_chamfer_out_list.extend(chamfer_out_list)
        
        print(f"chamfer mean for {mesh_name}: {np.mean(np.array(chamfer_list))}")
        print(f"p2s mean for {mesh_name}: {np.mean(np.array(p2s_list))}")
        
        
    with open("chamfer_dict.pkl", "wb") as f:
        pickle.dump(chamfer_dict, f)
    
    total_list = np.array(total_chamfer_list)
    total_p2s_list = np.array(total_p2s_list)
    np.save("total_chamfer_list.npy", total_list)
    np.save("total_p2s_list.npy", total_p2s_list)
    
    print(f"total chamfer mean: {np.mean(np.array(total_list))}")
    print(f"total p2s mean: {np.mean(np.array(total_p2s_list))}")
    
if __name__ == "__main__":
    main()