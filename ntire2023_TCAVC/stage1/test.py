from __future__ import print_function

import argparse
import math
import os
import queue
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform_lib
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import CenterCrop
from traceback import print_exc

import lib.TrainTransforms as transforms
from lib.videoloader_imagenet_stego import  VideosDataset_ImageNet_10k
from models.ColorVidNet import ColorVidNet
from models.FrameColor_stego import frame_colorization
from models.NonlocalNet_stego import  VGG19_pytorch

from utils.util import (batch_lab2rgb_transpose_mc,lab2rgb_transpose_mc, feature_normalize, l1_loss_my,
                        mkdir_if_not, mse_loss, parse, tensor_lab2rgb,
                        center_l,uncenter_l, weighted_l1_loss, weighted_mse_loss,save_frames)
from utils.util_distortion import (CenterPad_threshold, Normalize, RGB2Lab,CenterPad,
                                   ToTensor)
#from multiprocessing import set_start_method
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root_val", default="/dataset/videvo/test/imgs", type=str)

parser.add_argument("--gpu_ids", type=str, default="0,1", help="separate by comma")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--image_size", type=int, default=[216, 384])
parser.add_argument("--video_size", type=int, default=[268, 480])   #536*960  268*480
parser.add_argument("--test_video_size", type=int, default=[480, 848]) 
parser.add_argument("--ic", type=int, default=4)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--color_prior", type=int, default=[0.5, 0.5, 0.5, 0.5])  #[0.8, 0.6, 0.4, 0.3]  [0.5, 0.5, 0.5, 0.5]
                                                                                #[0.9621, 0.2731, 0.7017, 0.1786] [0.731, 0.3866, 0.6009, 0.3393]
parser.add_argument("--resume_iter", type=int, default=340000)
parser.add_argument("--strict_load", type=bool, default=True)

parser.add_argument("--checkpoint_dir", type=str, default="./stage1/checkpoints")
parser.add_argument("--test_video_output_path", type=str, default="./stage1_test_results")

parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')

def gpu_setup_DDP():
    cudnn.benchmark = True
    device = torch.device("cuda",opt.local_rank)
    return device		

def worker_init_fn(worker_id):
    return np.random.seed(torch.initial_seed()%(2**31)+worker_id)

def load_data_imagenet10k(loadmode,batch_size=14):
    if opt.local_rank==0:
        print("initializing dataloader")

    transforms_imagenet = [CenterPad(opt.test_video_size), RGB2Lab(), ToTensor(), Normalize()]
    train_dataset_imagenet = VideosDataset_ImageNet_10k(
        data_root=opt.data_root_val,
        image_size=opt.test_video_size,
        transforms_imagenet=transforms_imagenet,
        loadmode = loadmode
    )
    #imagenet_training_length = len(train_dataset_imagenet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_imagenet)

    data_loader = DataLoader(
        train_dataset_imagenet,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn = worker_init_fn,
        sampler = train_sampler,
    )
    return  data_loader



def resume_model():									   #继续学习 ，加载参数
    if opt.local_rank==0:
        print("resuming the learning")
    if opt.resume_iter:
        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "colornet_iter_%d.pth" % opt.resume_iter),map_location='cpu')
        colornet.module.load_state_dict(checkpoint,strict=opt.strict_load)
    return


def to_device_DDP(
        vggnet,
        colornet,
):
    if opt.local_rank==0:
        print("moving models to device DDP")
    colornet = torch.nn.parallel.DistributedDataParallel(colornet.to(device), device_ids=[opt.local_rank])
    vggnet = vggnet.to(device)
    return (
        vggnet,
        colornet,
    )									 #to gpu


def test_video():
    val_dataset = load_data_imagenet10k("NTIRE_test",batch_size=1)

    for iter,data in enumerate(tqdm(val_dataset)):

        (   I_lab,
            I_rgb,
            subdir,
        ) = data

       
        I_current_lab = I_lab
        I_current_lab = I_current_lab.cuda(non_blocking=True)
        I_current_l = I_current_lab[:, 0:1, :, :]										  # l a b -- 0 1 2
        I_current_lab = nn.functional.interpolate(I_current_lab, size = opt.image_size, mode="bilinear")

        ###### COLORIZATION ######						###### COLORIZATION ######
        I_last_lab = torch.zeros_like(I_current_lab)
        b,_,h,w=I_current_lab.shape
        color_prior = torch.tensor(opt.color_prior).to(device).view(1,4,1,1).repeat(b,1,h,w) # the color prior is abandoned and fixed to 0.5
        with torch.no_grad():
            I_current_ab_predict  = frame_colorization(
            I_current_lab,
            I_last_lab,
            color_prior,
            None,
            vggnet,
            colornet,
            feature_noise=0,
            luminance_noise=0,
            joint_training = False
            )
        I_last_lab = torch.cat((I_current_lab[:,0:1,:,:],I_current_ab_predict),dim=1)
        I_current_ab_predict = nn.functional.interpolate(I_current_ab_predict, size = opt.test_video_size, mode="bilinear")
        I_current_ab_predict = I_current_ab_predict.cpu()
        I_current_l = I_current_l.cpu()
        for i in range(len(I_current_l)):
            I_current_rgb = lab2rgb_transpose_mc(I_current_l[i], I_current_ab_predict[i])
            video_output_path = opt.test_video_output_path
            mkdir_if_not(video_output_path)
            save_frames(I_current_rgb, video_output_path, image_name = "f%s.png" % subdir)
    return 


if __name__ == "__main__":
    
    opt = parse(parser)
    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl')

    device = gpu_setup_DDP()

    # define network																						       #network
    colornet = ColorVidNet(8)
    vggnet = VGG19_pytorch()
    #vggnet.load_state_dict(torch.load("/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/vgg19_conv.pth",map_location='cpu'))
    vggnet.eval()																							  #加载VGG19并固定参数
    for param in vggnet.parameters():
        param.requires_grad = False

    # move to GPU processing
    (
        vggnet,
        colornet,
    ) = to_device_DDP(
        vggnet,
        colornet,
        )

    resume_model()
    
    n_parameters_colornet = sum(p.numel() for n,p in colornet.named_parameters())
    n_parameters_vggnet = sum(p.numel() for n,p in vggnet.named_parameters())        #计算参数量的方法
    n_parameters_all = n_parameters_colornet+n_parameters_vggnet
    print('number of params:',n_parameters_all)  #32802658 20024384 52827042

    print("start stage1!")
    cdc_test = test_video()
    print("stage1 done!")
