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

TRAINING_OUTPUT = "E:/2020_fi_training/train_"
TRAINING_OUTPUT_DEBUG = "E:/2020_fi_training_fast/train_"

TEST_PATH = "tests"
TEST_PATH_DEBUG = "tests_fasttrain"

# dataset paths
D_UCF101_BODY_TRAIN_DIR_DEV_3 = 'D:/gan_testing/fi_dataset_dirs/3/db_body_train_dev_128'
D_UCF101_BODY_TEST_DIR_DEV_3 = "D:/gan_testing/fi_dataset_dirs/3/db_body_test_dev_no_crop/"
D_UCF101_VAL_DIR_RELEASE_3 = 'D:/gan_testing/fi_dataset_dirs/3/db_body_train_release_128_small'

D_VIMEO_TRAIN_PATHS_3 = 'D:/gan_testing/fi_dataset_dirs/3/vimeo_training_release/tri_vallist.txt'
D_VIMEO_TEST_PATHS_3 = 'E:/dataset/vimeo_triplet/vimeo_triplet/tri_testlist.txt'
D_VIMEO_VAL_PATHS_3 = 'D:/gan_testing/fi_dataset_dirs/3/vimeo_training_release/tri_vallist.txt'
D_VIMEO_COLLECTION = [D_VIMEO_TRAIN_PATHS_3, D_VIMEO_TEST_PATHS_3, D_VIMEO_VAL_PATHS_3]

D_UCF101_BODY_TRAIN = 'D:/gan_testing/data/ucf101_body_motion/train'
