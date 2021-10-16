'''
Created on Jun 10, 2021

@author: Quang TRAN
'''

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch import nn

from utils import dataset_handler, calculator


def cal_psnr_tensor(img1, img2):
    '''
    Calculate PSNR from two tensors of images
    :param img1: tensor
    :param img2: tensor
    '''
    diff = (img1 - img2)
    diff = diff ** 2
        
    if diff.sum().item() == 0:
        return float('inf')

    rmse = diff.sum().item() / (img1.shape[0] * img1.shape[1] * img1.shape[2])
    psnr = 20 * np.log10(1) - 10 * np.log10(rmse)
        
    return psnr


def cal_ssim_tensor(X, Y,):
    return calculator.cal_ssim_tensor(X, Y, data_range=1.0).item()


def write_to_text_file(path, content, is_exist=True):
    '''
    create new file then write
    :param path:
    :param content:
    '''
    out = open(path, 'a') if is_exist else open(path, 'w')
    out.write(content)
    out.close();


def load_dataset_from_dir(batch_size, data_path, is_testing, patch_size):
    print("Loading dataset: %s" % data_path)
    dataset = dataset_handler.DBreader_frame_interpolation(db_dir=data_path, patch_size=patch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_testing, num_workers=0)
    print("Done loading dataset with %d patches." % (dataset.__len__()))
    
    return data_loader

# end load_dataset_from_dir


def load_dataset_from_path_file(batch_size, txt_path, is_testing, patch_size):
    print("Loading dataset: %s" % txt_path)
    dataset = dataset_handler.DBreader_frame_interpolation(path_file=txt_path, patch_size=patch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_testing, num_workers=0)
    print("Done loading dataset with %d patches." % (dataset.__len__()))
    
    return data_loader

# end load_dataset_from_path_file


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# end _weights_init_normal
