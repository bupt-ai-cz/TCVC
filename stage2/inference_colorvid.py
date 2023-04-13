'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import math
import os
import sys
from typing import Iterable
from PIL import Image
import torch
import torch.nn.functional as F
import datasets.transforms as T
from util.util_vec import *
import numpy as np
import glob
import time
import datetime
from pathlib import Path
import util.misc as utils
import csv , time
from models.vctran_colorvid import build_model
from models.transformer_inter import TransformerInternLayer
from models.FID import FID_utils,LPIP_utils 
from models.warping import WarpingLayer
from util.flowlib import read_flow
import torch.backends.cudnn as cudnn
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, default="r101_pretrained.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--model_path', type=str, 
                        default="./stage2/checkpoints/checkpoint_encoder_finetune02010000.pth")
    parser.add_argument('--decoder_path', type=str, 
                        default="./stage2/checkpoints/checkpoint_decoder_finetune02010000.pth")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--test_path', type=str,default='/dataset/temp/test_input')  #/dataset/videvo/test/imgs /dataset/DAVIS/Test/1/imgs

    parser.add_argument('--test_output_path', type=str,default='./stage2_test_results') 
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument("--img_size", type=int, default=[480 , 848] )  
    parser.add_argument("--nhead_warp", type=int, default=2 ) 
    parser.add_argument('--num_frames', default=4, type=int,
                        help="Number of frames")
    parser.add_argument('--scale_size', default=[216,384], type=float)                    

    parser.add_argument('--ref_path', default='./stage1_test_results',)  #./stage1_test_results

    return parser


    '''
    args.num_frames   test img length per clip, note that it doesn't contain ref img
    args.test_path    test img dir e.g. Davis/Val/
    args.test_output_path  remember to define your output path for per specific trained model
    args.out_csv_name     name of output csv file
    args.decoder_path     path of decoder
    args.model_path       path of encoder
    args.img_size         test img size
    args.scale_factor     scale imgs to accelerate inference
    args.mode             linear,parallel for decoder ; testonly when not do testing ; note that interlayers closed  in linear mode; nowe when do not caculate warp error.
    '''


@torch.no_grad()
def main_testonly_parallel(args):
    print("mode testonly!")

    device = torch.device(args.device)
    model, decoder  = build_model(args)
    interlayer_save = TransformerInternLayer(384,384,4)
    model.to(device)
    decoder.to(device)
    interlayer_save.to(device)
    checkpoint_vctran = torch.load(args.model_path,map_location='cpu')
    model.load_state_dict(checkpoint_vctran['model'],strict=True)
    interlayer_save.load_state_dict(checkpoint_vctran['inter_save'],strict=True)
    print("loaded encoder weight")
    checkpoint_decoder = torch.load(args.decoder_path, map_location='cpu')
    decoder.load_state_dict(checkpoint_decoder['model'],strict=True)
    print("loaded decoder weight")
    model.eval()
    decoder.eval()
    interlayer_save.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in interlayer_save.parameters():
        p.requires_grad = False
    
    n_parameters_model = sum(p.numel() for n,p in model.named_parameters())
    n_parameters_decoder = sum(p.numel() for n,p in decoder.named_parameters())        #计算参数量的方法
    n_parameters_inter = sum(p.numel() for n,p in interlayer_save.named_parameters()) 
    n_parameters_all = n_parameters_model+n_parameters_decoder+n_parameters_inter
    print('number of params:',n_parameters_all)

    #-------------------------------------------------------------#
    test_num_frames = args.num_frames
    subdirs = sorted(os.listdir(args.test_path))
    if args.ref_path:
        reference_imgs = sorted(glob.glob(os.path.join(args.ref_path,"*.JPEG"))+glob.glob(os.path.join(args.ref_path,"*.png")))
    for index,subdir in enumerate(tqdm(subdirs)):
        torch.cuda.empty_cache()
        # path = os.path.join(args.test_path, subdir,"2-1")
        path = os.path.join(args.test_path, subdir)
        # print("precessing:",path)
        #imgs = sorted()
        imgs = glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg'))
        imgs.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))
        #print(imgs)
        Clip = []
        for i in range(0,len(imgs),1):
            img = Image.open(imgs[i]).convert('RGB')
            Clip.append(transform(img).unsqueeze(0))
        Clip = torch.cat(Clip,dim=0)
        Clip.requires_grad =False
        if args.ref_path:
            ref = reference_imgs[index]
            ref = Image.open(ref).convert('RGB')
            ref = transform(ref).unsqueeze(0).cuda()
            #print(ref.shape)
        else:
            ref = Clip[0:1,:,:,:]
        tail_flag = 0
        for j in range(0,len(Clip),test_num_frames-1):
            clip = Clip[j:j+test_num_frames-1,:,:,:].cuda()
            #GT = GGT[j:j+test_num_frames-1,:,:,:]
            clip_large = torch.cat([ref,clip],dim=0)
            clip = F.interpolate(clip_large,size = args.scale_size,mode="bilinear")
            corr_num_frames = clip.shape[0]
            if corr_num_frames < test_num_frames:
                model.num_frames = corr_num_frames
                model.backbone[1].frames = corr_num_frames
                args.num_frames  = corr_num_frames
                tail_flag=1

            clip_l =clip[:,0:1,:,:]                                           # b c h w
            clip_ref_ab = clip[0:1,1:3,:,:]
            clip_rgb_from_gray = gray2rgb_batch(clip_l[1:]).to(device)     #  25 3 h w
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(clip_l[0:1,:,:,:]), clip_ref_ab), dim=1))
            clip_rgb_from_gray = torch.cat([I_reference_rgb,clip_rgb_from_gray],dim=0)
            #-------------------------------main process-------------------------#
            with torch.no_grad():
                if j==0:
                    out,out_trans,features,pos = model(clip_rgb_from_gray)
                else:
                    out,out_trans,features,pos = model(clip_rgb_from_gray,features_inter)
                out,warped_result,_= decoder(out,out_trans,clip_ref_ab.unsqueeze(0),features,clip_l.unsqueeze(0),pos=pos,temperature_warp=1e-10)                             # b n c h w
                if j==0:
                    features_inter = out_trans
                else:
                    features_inter = interlayer_save(features_inter,out_trans)
            del features , warped_result
            out_ab = out.squeeze(0)                             # n c h w
            #clip_l = clip_l.data.cpu()
            clip_l_large = clip_large[:,0:1,:,:].data.cpu()
            out_ab=F.interpolate(out_ab,size = args.img_size,mode="bilinear").data.cpu()
            #print("evaluation:",clip_l.shape,out_ab.shape)
            if tail_flag:
                model.num_frames = test_num_frames
                model.backbone[1].frames = test_num_frames
                args.num_frames  = test_num_frames
                tail_flag=0
            for i in range(corr_num_frames-1):
                clip_l_corr = clip_l_large[i+1:i+2,:,:,:]
                out_ab_corr = out_ab[i+1:i+2,:,:,:]
                outputs_rgb = batch_lab2rgb_transpose_mc(clip_l_corr,out_ab_corr)
                output_path = os.path.join(args.test_output_path,subdir)
                mkdir_if_not(output_path) 
                save_frames(outputs_rgb, output_path, image_name = "f%03d.png" % (j+i+1))
    print("done!")
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    cudnn.benchmark = True
    transform = T.Compose([
        T.CenterPad(args.img_size), T.RGB2Lab(), T.ToTensor(), T.Normalize()
        ])
    transform_GT = T.Compose([
        T.CenterPad(args.img_size),#T.ToTensor()
        ])
    transform2 = T.Compose(
        [T.CenterPad_vec(args.img_size), T.ToTensor()]
    )
    transform3 = T.Compose(
        [T.CenterPad(args.img_size), T.ToTensor()]
    )
    print("start stage2!")
    main_testonly_parallel(args)
    print("done stage2!")
    