'''
Created on Jun 9, 2021

@author: Quang TRAN
'''

import torch
from torch import nn, optim
from typing import Dict

from nets import base_net, blocks
from utils import CONSTANT, general_utils

res_map: Dict[int, int] = {
    1: 4,
    2: 6,
    3: 6,
    4: 8
    }


class FrameInterpolationGenerator(base_net.BaseModel):
    '''
    classdocs
    '''

    def __init__(self, nfg, depth, configs, epoch=None):
        '''
        Constructor
        '''
        super(FrameInterpolationGenerator, self).__init__(nfg, depth, epoch)
        self.nfg = nfg
        
        self._init_models(depth)        
        
        self.optimizer = None if configs is None else optim.Adam(self.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))
        self.apply(general_utils.weights_init_normal)
        
    def _init_models(self, depth):
        c = 6 if depth == 1 else 9
        
        self.frame_input_layer = nn.Conv2d(c, self.nfg, kernel_size=5, stride=1, padding=2)
        self.frame_synthesis_residuals = self._init_ftr_sequential(depth)
        self.frame_last_conv = nn.Conv2d(self.nfg, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
        self.map_input_layer = nn.Conv2d(c, self.nfg, kernel_size=5, stride=1, padding=2)
        self.map_residuals = self._init_ftr_sequential(depth)
        self.map_last_conv = nn.Conv2d(self.nfg, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    # end _init_models
    
    def _init_ftr_sequential(self, depth):
        ress = []
        for _ in range(res_map[depth]):
            ress.append(blocks.ResidualBlock(self.nfg))
            
        return nn.Sequential(*ress)
        
    # end _init_ftr_sequential
        
    def forward(self, pres, mid, lat):
        input_frames = torch.cat((pres, lat), 1) if mid is None else torch.cat((pres, mid, lat), 1)
        # frame
        ftr = self.frame_input_layer(input_frames)
        residual = ftr
        
        ftr = self.frame_synthesis_residuals(ftr)
        syn_out = ftr + residual
        syn_out = self.frame_last_conv(syn_out)
        syn_out = self.tanh(syn_out)
        
        # attention masks
        mask = self.map_input_layer(input_frames)
        mask_residual = mask
        mask = self.map_residuals(mask)
        mask = self.map_last_conv(mask + mask_residual)
        mask = self.sigmoid(mask)
        
        # calculate output frames
        out = mask * syn_out + (1 - mask) * pres
        
        return out
        
    # end forward
    
    def optimize(self):
        self.optimizer.step()
        
    def zero_grad_optim(self):
        self.optimizer.zero_grad()
        
    def do_zero_grad(self):
        self.zero_grad_optim()
        for param in self.parameters():
            param.grad = None
            
    # end zero_grad_for_training
    
    def save(self, path):
        checkpoint = {CONSTANT.LABEL_EPOCH: self.epoch,
                      CONSTANT.LABEL_MODEL_STATE_DICT: self.state_dict(),
                      CONSTANT.LABEL_OPTIMIZER_STATE_DICT: self.optimizer.state_dict()
                      }
        
        torch.save(checkpoint, path);
        
    def load(self, path):
        cpt = torch.load(path)
        self.load_state_dict(cpt[CONSTANT.LABEL_MODEL_STATE_DICT])
        self.epoch = cpt[CONSTANT.LABEL_EPOCH]
         
        if self.optimizer is None:
            return
         
        self.optimizer.load_state_dict(cpt[CONSTANT.LABEL_OPTIMIZER_STATE_DICT])
        # copy tensor into GPU manually
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                    
    # end load
