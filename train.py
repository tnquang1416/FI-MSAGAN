'''
Created on Jun 6, 2021

@author: Quang TRAN
@see: Multi-scale Attention GANs for Video Frame Interpolation paper (2020)
'''
from utils import CONSTANT, general_utils

import argparse
import torch
from torch import cuda, nn
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import os
import time
from nets import net, dis_net
from utils.vgg_loss import VggLoss

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--input_size", type=int, default=128, help="size of the input of network")
parser.add_argument("--nfg", type=int, default=64, help="feature map size of networks")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--decay_steps", type=int, default=20, help="no.steps to decay learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="number of batches between image sampling") 
parser.add_argument("--lambda_adv", type=float, default=0.0001, help="the default weight of adv Loss")
parser.add_argument("--lambda_vgg", type=float, default=0.001, help="the default weight of VGG features Loss")
parser.add_argument("--lambda_l1", type=float, default=1.0, help="the default weight of L1 Loss")
parser.add_argument("--path_gen", type=str, default=None, help="loaded generator for training")
parser.add_argument("--path_dis", type=str, default=None, help="loaded discriminator for training")
parser.add_argument("--version", type=str, default="temp", help="version of model")
parser.add_argument("--path", type=str, default=None, help="training folder")

opt = parser.parse_args()

net_gens = []
net_diss = []
seq_dis = None

# loss functions
lambda_advs = [1.0, 1.0, 1.0, 1.0]
adversarial_loss = nn.BCEWithLogitsLoss().cuda()
l1_loss = nn.L1Loss().cuda()
vgg_loss = VggLoss()


def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# end _weights_init_normal


def train(dataloader):
    # loss functions
    g_loss = 0
    d_loss = 0
    
    logging(opt.path, "[Epoch]\t[Batch]\t[Total_batch]\t[D loss]\t[G loss]\t[PSNR]")
    schedulers = []
    for i in range(len(net_gens)):
        schedulers.append(StepLR(net_gens[i].optimizer, step_size=max(opt.decay_steps, opt.n_epochs // 10), gamma=0.1))
        schedulers.append(StepLR(net_diss[i].optimizer, step_size=max(opt.decay_steps, opt.n_epochs // 10), gamma=0.1))
    
    while (net_gens[0].get_epoch() < opt.n_epochs):
        # enable training mode
        _specify_train()
        
        temp_log = ""
        for i, imgs in enumerate(dataloader):
            in_pres = imgs[0].to('cuda')
            in_lats = imgs[2].to('cuda')
            gt = imgs[1].to('cuda')
            
            gen_imgs, d_loss, d_seq_loss, g_loss = _train_interval(in_pres, in_lats, gt)
            
            # Show progress
            if i > 0: temp_log += "\n"
            temp_log = _show_progress(gen_imgs, gt, net_gens[0].get_epoch(), i, dataloader.__len__(), d_loss, d_seq_loss, g_loss)
        
        logging(opt.path, temp_log)
        
        for i in range(4):
            net_gens[i].increase_epoch()
            net_diss[i].increase_epoch()
        
        _save_models(path=opt.path, epoch=net_gens[0].get_epoch(), override=True, is_cpt=False)
        
        for i in range(len(net_gens)):
            schedulers[i].step()
    
    return net_gens[0].get_epoch(), g_loss, d_loss, d_seq_loss;

# end train


def _train_interval(in_pres, in_lats, gt):
    # Prepare data
    valids = []
    fakes = []

    gen_imgs = []
    gt_scales = []
        
    for i in range(len(net_gens)):
        temp_pre = nn.functional.interpolate(in_pres, scale_factor=2 ** (i - 3), mode="bilinear") if i < 3 else in_pres
        temp_lat = nn.functional.interpolate(in_lats, scale_factor=2 ** (i - 3), mode="bilinear") if i < 3 else in_lats
        temp_gt = nn.functional.interpolate(gt, scale_factor=2 ** (i - 3), mode="bilinear") if i < 3 else gt
        gt_scales.append(temp_gt)
        if i == 0:
            gen_imgs.append(net_gens[i](temp_pre, None, temp_lat))  # generate output images at coarsest level
        else:
            temp = nn.functional.interpolate(gen_imgs[i - 1], scale_factor=2, mode="bilinear")
            gen_imgs.append(net_gens[i](temp_pre, temp, temp_lat))
            
    # prepare multi-scale input
    
    for i in range(len(gt_scales)):
        valids.append(CONSTANT.Tensor(gt.shape[0], 1, 1, 1).fill_(0.95))
        fakes.append(CONSTANT.Tensor(gt.shape[0], 1, 1, 1).fill_(0.1))
        valids[i].requires_grad_(False)
        fakes[i].requires_grad_(False)
        
    # prepare training data

    # ------------------------
    #  Train Discriminator
    # ------------------------
    for i in range(len(net_diss)):
        net_diss[i].do_zero_grad()

        # Calculate gradient for D
        gt_distingue = net_diss[i](gt_scales[i])
        fake_distingue = net_diss[i](gen_imgs[i].detach())
        real_loss = adversarial_loss(gt_distingue, valids[i])
        fake_loss = adversarial_loss(fake_distingue, fakes[i])
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        
        net_diss[i].optimize()
    
    # end training single frame
        
    # train on sequence input for finest model only
    valid_seq = CONSTANT.Tensor(gt.shape[0], 512, 1, 1).fill_(0.95)
    faks_seq = CONSTANT.Tensor(gt.shape[0], 512, 1, 1).fill_(0.95)
    valid_seq.requires_grad_(False)
    faks_seq.requires_grad_(False)
    gt_distingue = seq_dis(in_pres, gt_scales[3], in_lats)
    fake_distingue = seq_dis(in_pres, gen_imgs[3].detach(), in_lats)
    real_seq_loss = adversarial_loss(gt_distingue, valid_seq)
    fake_seq_loss = adversarial_loss(fake_distingue, faks_seq)
    d_seq_loss = (real_seq_loss + fake_seq_loss) / 2
    d_seq_loss.backward()
    
    seq_dis.optimize()  # update D's weights
        
    # end training discriminator

    # ------------------------
    #  Train Generator
    # ------------------------
    adv_f_losses = 0
    for i in range(len(net_gens)):
        net_gens[i].do_zero_grad()
            
        adv_f_losses += (adversarial_loss(net_diss[i](gen_imgs[i]), valids[i]) * lambda_advs[i])
        
    adv_s_losses = adversarial_loss(seq_dis(in_pres, gt_scales[3], in_lats), valid_seq)
    lf_loss = l1_loss(gen_imgs[3], gt_scales[3])
    l_vgg = vgg_loss(gen_imgs[3], gt_scales[3])
    
    g_loss = opt.lambda_adv * (adv_f_losses + adv_s_losses) + opt.lambda_l1 * lf_loss + opt.lambda_vgg * l_vgg
    
    g_loss.backward()
                        
    for i in range(len(net_gens)):        
        net_gens[i].optimize()  # update D's weights
        
    # end training generator
    
    return gen_imgs[3], d_loss.item(), d_seq_loss.item(), g_loss.item()

# end _train_interval


def _show_progress(gen_imgs, groundtruth, epoch, batch_index, total_batch, d_loss, d_seq_loss, g_loss):
    psnr = general_utils.cal_psnr_tensor(gen_imgs[0].cpu(), groundtruth.data[0].cpu())
    log = ("%s: [Epoch %d] [Batch %d/%d] [D loss: %1.5f] [D Seq loss: %1.5f] [G loss: %1.5f] [PSNR: %1.2f]" 
                % (opt.version, epoch, batch_index, total_batch, d_loss, d_seq_loss, g_loss, psnr))
             
    # Display result (input and output) after every opt.sample_intervals
    batches_done = epoch * total_batch + batch_index
        
    if batches_done % opt.sample_interval == 0:
        save_image(gen_imgs.data[:16], opt.path + "/train_%d.png" % (batches_done), nrow=4, normalize=True)
        print("Saved train_%d.png" % batches_done)
        print(log)
    elif batches_done % 20 == 0:
        print(log)
    
    return "%d\t%d/%d\t%f\t%f\t%f\t%f" % (epoch, batch_index, total_batch, d_loss, d_seq_loss, g_loss, psnr)

# end _show_progress


def logging(path, content, is_exist=True):
    general_utils.write_to_text_file("%s/%s" % (path, CONSTANT.LOG_FILE), content + "\n", is_exist)
    
# end logging


def train_unlimited(dataloader, test_dataloader):
    # loss functions
    g_loss = 0
    d_loss = 0

    schedulers = []
    for i in range(len(net_gens)):
        schedulers.append(StepLR(net_gens[i].optimizer, step_size=opt.decay_steps, gamma=0.1))
        schedulers.append(StepLR(net_diss[i].optimizer, step_size=opt.decay_steps, gamma=0.1))
    
    logging(opt.path, "[Epoch]\t[Batch]\t[Total_batch]\t[D loss]\t[G loss]\t[PSNR]")
    
    while (True):    
        # enable training mode
        _specify_train()
        
        temp_log = ""
        for i, imgs in enumerate(dataloader):
            in_pres = imgs[0].to('cuda')
            in_lats = imgs[2].to('cuda')
            gt = imgs[1].to('cuda')
            
            gen_imgs, d_loss, d_seq_loss, g_loss = _train_interval(in_pres, in_lats, gt, 1)
            
            # Show progress
            temp_log = _show_progress(gen_imgs, gt, net_gens[0].get_epoch(), i, dataloader.__len__(), d_loss, d_seq_loss, g_loss)
        # end an epoch of training
        
        logging(opt.path, temp_log)
        
        for i in range(4):
            net_gens[i].increase_epoch()
            net_diss[i].increase_epoch()
            
        _save_models(path=opt.path, epoch=net_gens[0].get_epoch(), override=True, is_cpt=False)
        
        for i in range(len(net_gens)):
            schedulers[i].step()
    
    return net_gens[0].get_epoch(), g_loss, d_loss, d_seq_loss;

# end train_unlimited


def _specify_train():
    for i in range(4):
        net_gens[i].train()
        net_diss[i].train()
    
    seq_dis.train()
# end specify_train


def _specify_cuda():
    for i in range(4):
        net_gens[i].to('cuda')
        net_diss[i].to('cuda')
    
    seq_dis.to('cuda')
    
# end _specify_cuda


def _save_models(path, epoch, override, is_cpt=False):
    '''
    
    :param path: directed parent directory of the saved model folder
    :param override:
    :param is_cpt:
    '''
    if is_cpt:
        path = os.path.join(path, "cpt", str(epoch))
    elif not override:
        path = os.path.join(path, str(int(time.time())))
    else:
        path = os.path.join(path, "models")
    
    os.makedirs(path, exist_ok=True);
        
    for i in range(4):
        net_gens[i].save(path="%s/net_gen_%d.pt" % (path, i + 1))
        net_diss[i].save(path="%s/net_dis_%d.pt" % (path, i + 1))
        
    seq_dis.save(path="%s/net_seq_dis.pt" % (path))

# end _save_models


def main():
    if not cuda.is_available():
        print(CONSTANT.MESS_NO_CUDA);
        return;
    
    opt.path = "output_" if opt.path is None else opt.path
    data_dir = ["data/tri_trainlist.txt", "data/tri_vallist.txt"]

    print(opt)
        
    os.makedirs(opt.path, exist_ok=True);
    
    print("==============<Prepare networks/>============================")
    t1 = time.time()

    globals()['seq_dis'] = dis_net.FramesSequenceDiscriminator(nfg=opt.nfg, configs=opt, epoch=None)
    for i in range(1, 5):
        # coarsest at 1 and finest at 4
        net_gens.append(net.FrameInterpolationGenerator(nfg=opt.nfg, depth=i, configs=opt, epoch=None))
        net_diss.append(dis_net.FrameInterpolationDiscriminator(nfg=opt.nfg, depth=i, configs=opt, epoch=None))
        
    _specify_cuda()
    
    print("%s\n%s" % (net_gens[3].__class__.__name__, str(net_gens[3])))
    print("%s\n%s" % (net_diss[3].__class__.__name__, str(net_diss[3])))
    print("%s\n%s" % (seq_dis.__class__.__name__, str(seq_dis)))
    print("==> Takes total %1.4fs" % ((time.time() - t1)))
    logging(opt.path, "%s\n%s\n%s\nDataset: %s" % (str(opt), str(net_gens[3]), str(net_diss[3]), data_dir), is_exist=False)
    
    print("==============<Prepare dataset/>=============================")
    t1 = time.time()
    p_size = 3
    dataloader = general_utils.load_dataset_from_path_file(batch_size=opt.batch_size, txt_path=data_dir[0], is_testing=False, patch_size=p_size)
    test_dataloader = general_utils.load_dataset_from_path_file(batch_size=1, txt_path=data_dir[1], is_testing=True, patch_size=p_size)
    print("==> Takes total %1.4fs" % ((time.time() - t1)))

    print("==============<Training/>====================================")
    t1 = time.time()
    if opt.n_epochs > 0:
        train(dataloader) 
    else:
        train_unlimited(dataloader, test_dataloader)
    print("==> Takes total %1.2fmins" % ((time.time() - t1) / (60)))
    logging(opt.path, "Training takes %1.2fmins" % ((time.time() - t1) / (60)), is_exist=True)
    
    _save_models(path=opt.path, epoch=net_gens[0].get_epoch(), override=True)

    
main()

if __name__ == '__main__':
    pass
