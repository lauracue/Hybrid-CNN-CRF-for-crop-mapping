from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
import kornia.augmentation as K
import torch.nn as nn

logger = getLogger()

class MyAugmentationPipeline(nn.Module):
    def __init__(self) -> None:
        super(MyAugmentationPipeline, self).__init__()
        self.h = K.RandomHorizontalFlip3D(p=0.5, return_transform=False)
        self.v = K.RandomVerticalFlip3D(p=0.5, return_transform=False)
        self.n = K.RandomRotation3D(degrees=[(0.0,0.0),(90.0,90.0),(90.0,90.0)], return_transform=False)
        self.e = K.RandomRotation3D(degrees=[(0.0,0.0),(90.0,90.0),(180.0,180.0)], return_transform=False)
        self.s = K.RandomRotation3D(degrees=[(0.0,0.0),(90.0,90.0),(270.0,270.0)], return_transform=False)

    def forward(self, input, mask):
        L, H, W = mask.shape
        mask = torch.reshape(mask, (1, L, H, W))
     
        h_params = self.h.forward_parameters(input.shape)
        input = self.h(input, h_params)
        mask = self.h(mask.float(), h_params)
          
        v_params = self.v.forward_parameters(input.shape)
        input = self.v(input, v_params)
        mask = self.v(mask.float().float(), v_params)
        
        n_params = self.n.forward_parameters(input.shape)
        input = self.n(input, n_params)
        mask = self.n(mask.float().float(), n_params)

        e_params = self.e.forward_parameters(input.shape)
        input = self.e(input, e_params)
        mask = self.e(mask.float().float(), e_params)

        s_params = self.s.forward_parameters(input.shape)
        input = self.s(input, s_params)
        mask = self.s(mask.float().float(), s_params)        
        
        return input, mask


class DatasetFromCoord(Dataset):
    def __init__(self, 
                 data_img, 
                 labels,
                 coords, 
                 psize,
                 samples=False,
                 augm=False):

        super(DatasetFromCoord, self).__init__()
        self.data_img = data_img
        self.coord = coords
        self.psize = psize
        self.labels = labels
        self.augm = None
        self.samples = samples
        
        if self.augm:

            self.trans = MyAugmentationPipeline()
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.coord)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.data_img[:,:,self.coord[idx][0]-self.psize//2:self.coord[idx][0]+self.psize//2,
                              self.coord[idx][1]-self.psize//2:self.coord[idx][1]+self.psize//2]
        image = torch.from_numpy(image.astype(np.float32))
        
        ref = self.labels[:,self.coord[idx][0]-self.psize//2:self.coord[idx][0]+self.psize//2,
                              self.coord[idx][1]-self.psize//2:self.coord[idx][1]+self.psize//2]
        ref = torch.from_numpy(ref.astype(np.uint8))
        
        if self.augm:
            image = self.trans(image,ref)

        return image, ref
