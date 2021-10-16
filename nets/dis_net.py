'''
Created on Jun 9, 2021

@author: Quang TRAN
'''

import torch
from torch import nn, optim

import nets
from utils import CONSTANT, general_utils


class FrameInterpolationDiscriminator(nets.base_net.BaseModel):
    '''
    classdocs
    '''

    def __init__(self, nfg, depth, configs, epoch=None):
        '''
        Constructor
        '''
        super(FrameInterpolationDiscriminator, self).__init__(nfg, depth, epoch)
        self.nfg = nfg
        
        self._init_models(depth)
        
        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))
        self.apply(general_utils.weights_init_normal)
        
    def _init_models(self, depth):
        models = []
        self.block_1 = nets.blocks.DenseBlock(3, self.nfg, kernel_size=3, stride=1)
        
        for i in range(depth):
            models.append(nets.blocks.DenseBlock(self.nfg * 2 ** i, self.nfg * 2 ** (i + 1)))
        
        self.block_n = nn.Conv2d(self.nfg * 2 ** depth, 1, kernel_size=8, stride=2, padding=1, bias=False)
            
        self.blocks = nn.Sequential(*models)
            
        self.activation = nn.Sigmoid()
    
    # end _init_models
        
    def forward(self, frame):
        out = self.block_1(frame)
        out = self.blocks(out)
        out = self.block_n(out)
            
        out = self.activation(out)
        
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
        
    # end save
        
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
        
# end class FrameInterpolationDiscriminator


class FramesSequenceDiscriminator(FrameInterpolationDiscriminator):

    def __init__(self, nfg, configs, epoch=None):
        '''
        Constructor
        '''
        super(FramesSequenceDiscriminator, self).__init__(nfg, 4, configs, epoch)
        
    def _init_models(self, depth=1):
        models = []
        models.append(nets.blocks.DenseBlock(9, self.nfg))
        
        for i in range(3):
            models.append(nets.blocks.DenseBlock(self.nfg * 2 ** i, self.nfg * 2 ** (i + 1)))
        
        self.blocks = nn.Sequential(*models)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8)  # 8 is the size of input N x C x 8 x 8
        self.activation = nn.Sigmoid()
        
    # end _init_models
        
    def forward(self, pres, mids, lats):
        input_frames = torch.cat((pres, mids, lats), 1)
        
        out = self.blocks(input_frames)            
        out = self.global_avg_pool(out)
        out = self.activation(out)
        
        return out
    
    # end forward
