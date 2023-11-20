import torch, os
import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class LBSDataset(Dataset):
    def __init__(self, data_root, mode = "compressed", angles = [-30, 30]):
        super(LBSDataset, self).__init__()
        self.data_root = data_root
        self.color_root = os.path.join(self.data_root, "COLOR/ALIGNED")
        
        self.mode = mode
        if mode == "compressed":
            self.lbs_root = os.path.join(self.data_root, "LBS/ALIGNED")
        else:
            self.lbs_root = os.path.join(self.data_root, "LBS/FULL_RENDER")
        
        self.data_dict_list = []
        self.init_loader()
        
    def init_loader(self,): 
        self.data_dict_list = []
        
        posed_names = os.listdir(self.lbs_root)
        for dataname in posed_names:
            color_data_path = os.path.join(self.color_root, dataname)
            lbs_data_path = os.path.join(self.lbs_root, dataname)
            
            data_list = self.get_datalist_from_dir(color_data_path, lbs_data_path)
            self.data_dict_list += data_list
    
    def get_datalist_from_dir(self, color_dir, lbs_dir):
        filelist = glob(os.path.join(color_dir, "*_front.png"))
        backlist = [filename.replace("_front", "_back") for filename in filelist]
        
        front_filenames = [os.path.basename(filename) for filename in filelist]
        
        color_dicts = [{"front":front, "back":back} for front, back in zip(filelist, backlist)]
        
        if self.mode=="compressed":
            lbs_filelist = [os.path.join(lbs_dir, filename) for filename in front_filenames]
            lbs_backlist = [filename.replace("_front", "_back") for filename in lbs_filelist]
            
            lbs_dicts = [{"front":front, "back":back} for front, back in zip(lbs_filelist, lbs_backlist)]
        else:
            lbs_ptlist = glob(os.path.join(lbs_dir, filename.replace(".png", ".npz").replace("_front", "")) for filename in front_filenames)
            lbs_dicts = [{"data":data} for data in lbs_ptlist]
            
        data_dicts = [{"color":color, "lbs":lbs} for color, lbs in zip(color_dicts, lbs_dicts)]
        
        return data_dicts
    
    def __len__(self):
        return len(self.data_dict_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_dict_list[idx]
        color_data = data_dict["color"]
        lbs_data = data_dict["lbs"]
        
        color_front = cv2.imread(color_data["front"])/255.
        color_back = cv2.imread(color_data["back"])/255.
        
        if self.mode=="compressed":
            lbs_front = cv2.imread(lbs_data["front"])/255.
            lbs_back = cv2.imread(lbs_data["back"])/255.
        else:
            lbs_data = np.load(lbs_data["data"]).get('lbs')
            lbs_front = lbs_data[0]
            lbs_back = lbs_data[1]
        

        color_concat = np.concatenate([color_front, color_back], axis = 2)
        # lbs_concat = np.concatenate([lbs_front[None, ...], lbs_back[None, ...]], axis = 0)
        lbs_concat = np.concatenate([lbs_front, lbs_back], axis = 2)
        # color_concat = color_front
        color_concat = color_concat.transpose(2, 0, 1)
        # lbs_concat = lbs_concat.transpose(0, 3, 1, 2)
        # lbs_concat = lbs_front
        lbs_concat = lbs_concat.transpose(2,0,1)
        
        # print(color_concat.shape)
        # print(lbs_concat.shape)
        
        return torch.tensor(np.asarray(color_concat, dtype=np.float32)), torch.tensor(np.asarray(lbs_concat, dtype=np.float32))