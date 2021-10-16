'''
Created on Jan 22, 2021

@author: Quang Tran
'''

import torch
import math

from torch.nn import functional as F


def pre_processing(frame_tensor, size):
    '''
    Apply padding on the input to make it fit the size of network
    :param frame_tensor:
    :param size:
    '''
    h = int(list(frame_tensor.size())[2])
    w = int(list(frame_tensor.size())[3])
        
    if (h < size):
        pad_h = size - h
        frame_tensor = F.pad(frame_tensor, (0, 0, 0, pad_h))
            
    if (w < size):
        pad_w = size - w
        frame_tensor = F.pad(frame_tensor, (0, pad_w, 0, 0))
    
    return frame_tensor, h, w

# end pre_processing


def post_processing(frame_tensor, h, w):
    '''
    Remove padded values if necessary
    :param frame_tensor:
    :param h:
    :param w:
    '''
    if h > 0:
        frame_tensor = frame_tensor[:, :, 0:h, :]
    if w > 0:
        frame_tensor = frame_tensor[:, :, :, 0:w]
            
    return frame_tensor

# end post_processing
