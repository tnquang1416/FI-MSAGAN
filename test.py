'''
Created on Jun 10, 2021

@author: Quang TRAN
'''

import torch
from torch import cuda, nn
from torchvision.utils import save_image

import os
import argparse
import numpy as np
import time

from utils import CONSTANT, general_utils, frame_utils
import shutil
from nets import net

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--input_size", type=int, default=128, help="size of the input of network")
parser.add_argument("--nfg", type=int, default=32, help="feature map size of networks")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--path", type=str, default="output", help="training image folder")
parser.add_argument("--gen_path", type=str, required=True, help="loaded generator for testing")
parser.add_argument("--version", type=str, default="1.0.0", help="version of model")

opt = parser.parse_args()

net_gens = []


def logging(path, content, is_exist=True):
    general_utils.write_to_text_file("%s/%s" % (path, CONSTANT.LOG_FILE), content + "\n", is_exist)
    
# end logging


def _cal_metrics(gt_tensors, gen_tensors):
    if (gen_tensors.data[0].shape[1] != gt_tensors.data[0].shape[1]):
        print(CONSTANT.MESS_CONFLICT_DATA_SIZES)
        
    list_psnr = []
    list_ssim = []
        
    for i in range(gen_tensors.data.shape[0]):
        list_psnr.append(general_utils.cal_psnr_tensor(gen_tensors.data[i], gt_tensors.data[i]))
        list_ssim.append(general_utils.cal_ssim_tensor(gen_tensors.data[i].view(1, gen_tensors.shape[1], gen_tensors.shape[2], gen_tensors.shape[3]),
                                                gt_tensors.data[i].view(1, gt_tensors.shape[1], gt_tensors.shape[2], gt_tensors.shape[3])))
        list_ssim.append(0)
            
    return list_psnr, list_ssim

# end _cal_metrics


def _generate_full_frame(input1, input2):
    '''
    Generate a full frame (larger than training input 128x128) from given frames.
    :param input1: previous frame
    :param input2: latter frame
    '''
    h = math.ceil(input1.shape[2]/division) * division
    w = math.ceil(input1.shape[3]/division) * division
    size = max(h, w)
    temp1, h1, w1 = frame_utils.pre_processing(input1, size)
    temp2, h2, w2 = frame_utils.pre_processing(input2, size)
    temp_gen = _generate(temp1, temp2)
    
    assert h1 == h2
    assert w1 == w2

    return temp_gen.cuda()

# end _generate_full_frame


def _generate(block1, block2):
    output = None
    for i in range(len(net_gens)):
        temp_pre = nn.functional.interpolate(block1, scale_factor=2 ** (i - 3), mode="bilinear") if i < 3 else block1
        temp_lat = nn.functional.interpolate(block2, scale_factor=2 ** (i - 3), mode="bilinear") if i < 3 else block2
        if output is not None:
            output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear")
        output = net_gens[i](temp_pre, output, temp_lat)
    
    return output

# end _generate_a_block


def _run_test_one_batch(imgs, channels):
    '''
    Generate frames and calculate metrics for evaluation for previous versions of v2.0.0
    :param imgs: image data patch (three images)
    '''
    input1 = imgs[0].to('cuda')
    input2 = imgs[2].to('cuda')
    gt_frames = imgs[1].to('cuda')
        
    # Process full frame test by processing each partition
    temp_start = time.time()
    gen_frames = _generate_full_frame(input1, input2)        
    runtime = time.time() - temp_start

    psnr, ssim = _cal_metrics(gt_frames, gen_frames)
    
    samples = torch.cat((input1, gt_frames, gen_frames, input2), 1)
    samples = samples.view(samples.shape[0], -1, channels, samples.shape[2], samples.shape[3])
    
    return samples, [runtime], psnr, ssim

# end _run_test_one_batch

    
def run_test(dataloader, output_path, channels, batch_size):
    '''
    Run test on whole loaded dataset
    :param dataloader: dataloader object (loading dataset)
    :param output_path: path for saving output
    '''
    psnr_list = []
    ssim_list = []
    time_list = []
    current_progress = 0
    
    for i in range(len(net_gens)):
        net_gens[i].eval()
    
    for i, imgs in enumerate(dataloader):
        process = 100.0 * i / len(dataloader.dataset)
        if process - current_progress >= 10:
            current_progress += 10
            print(str(current_progress) + "%")
        
        # Process full frame test by processing each partition
        samples, runtime, psnr, ssim = _run_test_one_batch(imgs, channels)
        time_list.extend(runtime)
        psnr_list.extend(psnr)
        ssim_list.extend(ssim)
        
        temp_log = ""
        for j in range(samples.shape[0]):
            temp_log = temp_log + ("\n" if j > 0 else "") + "%d\t%f\t%f" % (i * batch_size + j, psnr[j], ssim[j])
            path = "%s/%s_%s.png" % (output_path, opt.version, format(i * batch_size + j, '05d'))
            
            save_image(samples[j], path, padding=10, nrow=samples.shape[1])
        
        logging(output_path, content=temp_log, is_exist=True)

    print("Calculate metrics...")
    minPSNR = min(psnr_list)
    maxPSNR = max(psnr_list)
    avgPSNR = np.average(np.array(psnr_list))
    minSSIM = min(ssim_list)
    maxSSIM = max(ssim_list)
    avgSSIM = np.average(np.array(ssim_list))
    avgTime = np.average(np.array(time_list))
    
    log = ""
    log += "Test on %d patches.\n" % (len(dataloader.dataset))
    log += "Min/Max/Avg PSNR value is %1.2f/%1.2f/%1.2f dB\n" % (minPSNR, maxPSNR, avgPSNR)
    log += "Min/Max/Avg SSIM value is %1.3f/%1.3f/%1.3f\n" % (minSSIM, maxSSIM, avgSSIM)
    log += "Average generate time: %1.4f s." % (avgTime)
    print(log)
    print("Done. See %s for the details." % (output_path))
    logging(path=output_path, content=log, is_exist=True)
        
    return batch_size * len(dataloader.dataset), minPSNR, maxPSNR, avgPSNR, minSSIM, maxSSIM, avgSSIM, avgTime;

# end run_test


def main():
    if not cuda.is_available():
        print(CONSTANT.MESS_NO_CUDA);
        return;

    data_path = "data/tri_testlist.txt"
    print(opt)

    opt.path = "gen_output"
    try:
        shutil.rmtree(opt.path, ignore_errors=False)
        print("----------------------------------------------\nDeleted old results at %s\n----------------------------------------------" % opt.path)
    except FileNotFoundError:
        pass
    os.makedirs(opt.path, exist_ok=False);

    dataloader = general_utils.load_dataset_from_dir(batch_size=1, data_path=data_path, is_testing=True, patch_size=3)
    print("----------------------------------------------")
    
    for i in range(1, 5):
        # coarsest at 1 and finest at 4
        model = net.FrameInterpolationGenerator(nfg=opt.nfg, depth=i, configs=None, epoch=None)
        # models are saved as net_gen_0.pt, net_gen_1.pt, ...
        model.load("%s/net_gen_%d.pt" % (opt.gen_path, i))
        model.to('cuda')
        net_gens.append(model)
        print(net_gens[i - 1])

    print("%s\n----------------------------------------------" % net_gens[0].__class__.__name__)
    print("Start testing at %d epoch." % net_gens[0].get_epoch())
    logging(opt.path, "Testing:\n%s\n%s\nDataset: %s\nEpoch: %d" % (str(opt), str(net_gens[0]), data_path, net_gens[0].get_epoch()), is_exist=False)
 
    run_test(dataloader, opt.path, opt.channels, opt.batch_size)


main()

if __name__ == '__main__':
    pass
