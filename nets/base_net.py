'''
Created on Jun 9, 2021

@author: Quang TRAN
'''

from torch import nn


class BaseModel(nn.Module):
    '''
    classdocs
    '''

    def __init__(self, nfg, depth, epoch=None):
        '''
        Constructor
        '''
        super(BaseModel, self).__init__()
        if depth not in range(1, 5): print("The given depth of model does not in the expected range [1,4].")
        self.epoch = 0 if epoch is None else epoch
        self.nfg = nfg
    
    # end _init_models
        
    def increase_epoch(self):
        self.epoch += 1
        
    def get_epoch(self):
        return self.epoch
        
    # end forward
    
    def optimize(self):
        raise NotImplementedError
        
    def zero_grad_optim(self):
        raise NotImplementedError
        
    def do_zero_grad(self):
        raise NotImplementedError
            
    # end zero_grad_for_training
    
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
