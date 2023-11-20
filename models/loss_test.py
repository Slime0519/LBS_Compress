import trimesh
from LBS_unwrapping import utils


verts = utils.deform_vertices(torch.Tensor(verts).to(self.device).unsqueeze(0), smpl_model_posed, full_lbs.to(self.device), full_pose.to(self.device), inverse=True, return_vshape=False, device=self.device)
