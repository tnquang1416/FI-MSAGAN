'''
Created on Nov 20, 2020

@author: Quang Tran
'''
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import join, isdir
from PIL import Image

import numpy as np


class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory or from input tensor list
    """

    def __init__(self, db_dir=None, resize=None, tensor_list=None, path_list=None, path_file=None, patch_size=3):
        self.patch_size = patch_size
        if db_dir is not None:
            self._load_from_dir(db_dir, resize)
            self.mode = 1
        elif tensor_list is not None:
            self._load_from_tensor_list(tensor_list)
            self.mode = 2
        elif path_list is not None:
            self._load_from_dir(path_list, resize)
            self.mode = 3
        elif path_file is not None:
            self._load_from_file_path(path_file, resize)
            
    # end __init__
            
    def _load_from_tensor_list(self, tensor_list):
        '''
        Load from numpy tensor list (no_triplets, triplets_index, c, w, h)
        :param tensor_list:
        '''
        self.imgs_list = tensor_list
        
    def _load_from_dir(self, db_dir, resize=None):
        '''
        DBreader reads all triplet set of frames in a directory.
        '''
        files_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])

        self._load_from_list_path(files_list, resize)
    
    # end _load_from_dir
    
    def _load_from_list_path(self, path_list, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.files_list = path_list
        self.imgs_list = []
        flag = False
        try:
            Image.open("%s/frame%d.png" % (self.files_list[0], 0))
        except:
            flag = True

        for path in self.files_list:
            data = []
            if flag:
                for i in range(self.patch_size): data.append(self.transform(Image.open("%s/im%d.png" % (path, i + 1))))
            else:
                for i in range(self.patch_size): data.append(self.transform(Image.open("%s/frame%d.png" % (path, i))))
            self.imgs_list.append(data)
    
    # end _load_from_list_path
    
    def _load_from_file_path(self, txt_file_path, resize=None):
        lines = open(txt_file_path, "r")
        dir_path = txt_file_path[:txt_file_path.find("/t")]
        file_list = []
        for line in lines:
            if len(line.rstrip()) < 1: continue
            file_list.append("%s/%s/%s" % (dir_path, "sequences", line.strip()))
            
        return self._load_from_list_path(file_list, resize)

    # end _load_from_file_path
    
    def __getitem__(self, index):
        frame0 = self.imgs_list[index][0]
        frame1 = self.imgs_list[index][1]
        frame2 = self.imgs_list[index][2]
        
        if self.patch_size == 3:
            return frame0, frame1, frame2
        
        frame3 = self.imgs_list[index][3]
        frame4 = self.imgs_list[index][4]

        return frame0, frame1, frame2, frame3, frame4

    def __len__(self):
        return len(self.files_list)
