import numpy as np
import trimesh, json
import sys

sys.path.append("/home/junmyeong/workspace/JM_LBSRecon")

from glob import glob
from pathlib import Path


SMPLX_MODEL_PATH = '/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/resource/smpl_models/smplx/SMPLX_NEUTRAL_2020.npz'
SMPLX_GENDER = "male"

ROOT = "/home/junmyeong/workspace/JM_LBSRecon/measure_LBS_error/sample_RP_T"

# CANON_GT_DIR = Path(ROOT) / "MESH" /"CANON_GT"
GT_CANON = Path(ROOT) / "MESH" /"CANON_GT_TEMP"
SMPLX_CANON = Path(ROOT) / "SMPLX"  / "CANON_SMPLX_TEMP"
POSE_EXAMPLE = Path(ROOT) / "pose_sample.npy"
MESH_NAME_LIST = [
    PATH.name for PATH in GT_CANON.glob("*/")
]

POSE_SET = np.load(POSE_EXAMPLE)

def get_path_dict(mesh_name):
    return {
        "GT_CANON": GT_CANON / mesh_name,
        "SMPLX_CANON": SMPLX_CANON / mesh_name,
    }

    
    
