'''
Created on Jun 9, 2021

@author: Quang TRAN
'''

import torch

Tensor = torch.cuda.FloatTensor

# Message
MESS_NO_CUDA = "/!\Cuda is not available."
MESS_CONFLICT_DATA_SIZES = "/!\Different in size of output and ground truth."

# saved model's label
LABEL_EPOCH = "epoch"
LABEL_MODEL_STATE_DICT = "state_dict"
LABEL_OPTIMIZER_STATE_DICT = "optimizer"
LABEL_LOSS = "loss"

# path
LOG_FILE = "log.txt"
