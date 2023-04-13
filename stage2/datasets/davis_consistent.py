"""
DAVIS data loader
"""
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torchvision
import datasets.transforms as T
import os
from PIL import Image
from random import randint
import cv2
import random
import glob
from util.util_vec import gray2rgb_batch
from torchvision.transforms import CenterCrop
from util.flowlib import read_flow
class CenterCrop_np(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        #print("in np:",image_size)
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy

class Dataset:
    def __init__(self, img_folder, img_folder2, transform_video,num_frames,img_size = [216,384],crop_size=[216,384],
    mask_folder1=None,mask_folder2=None):
        self.img_folder = img_folder
        self.img_folder2 = img_folder2
        self.num_frames = num_frames
        self.img_pairs = self.parse_img(self.img_folder,maskroot=mask_folder1,gap = 2)
        self.img_pairs += self.parse_img(self.img_folder2,maskroot=mask_folder2,gap = 1)
        self.transform_video = transform_video
        self.CenterPad = T.CenterPad([img_size[0],img_size[1]])
        self.CenterPad_mask = T.CenterPad_mask([img_size[0],img_size[1]])
        self.CenterCrop = CenterCrop(crop_size)
        self.ToTensor = T.ToTensor()


    def parse_img(self,data_root,maskroot=None,gap = 1):
        image_pairs = []
        subdirs = sorted(os.listdir(data_root))
        for subdir in subdirs:
            path = os.path.join(data_root, subdir)
            if not os.path.isdir(path):
                print("img folder",path,"is not a correct folder!")
                continue
            imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
            img_len = len(imgs)
            gaped_len = 2 * gap * (self.num_frames-1)+1
            if img_len < gaped_len:
                print("img folder",path,"doesn't have enough imgs!")
                temp_gap = img_len // self.num_frames
                temp_gaped_len = temp_gap * (self.num_frames-1)+1
                maskpath = os.path.join(maskroot, subdir)
                for i in range(img_len-temp_gaped_len+1):
                    image_pairs.append([path,i,maskpath,temp_gap])
                continue
            maskpath = os.path.join(maskroot, subdir)
            for i in range(0,img_len-gaped_len+1,3):
                image_pairs.append([path,i,maskpath,gap])
        return image_pairs

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        path , id,maskpath,gap = self.img_pairs[idx]
        imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
        # if maskpath:
        masks = sorted(glob.glob(os.path.join(maskpath, '*.png'))+glob.glob(os.path.join(maskpath, '*.jpg')))
        clip = []
        mask_clip = []
        #ref = Image.open(imgs[ np.random.randint(0, id+1) ]).convert('RGB')
        ref = Image.open(imgs[id]).convert('RGB')
        ref = self.transform_video(self.CenterCrop(self.CenterPad(ref)))
        img_len = len(imgs) - id -1
        clip_length = img_len // gap
        clip.append(ref.unsqueeze(0))
        for i in range(1,clip_length):
            img = Image.open(imgs[gap*i+id]).convert('RGB')
            #img = self.transform_video(self.CenterPad(img))
            img = self.transform_video(self.CenterCrop(self.CenterPad(img)))
            #img_1 = img[:,:,0:1,:,:]                # B N C H W
            #img_ab = img[:,:,1:3,:,:]
            #samples_rgb_from_gray = gray2rgb_batch(samples_1)
            clip.append(img.unsqueeze(0)) 
            # if maskpath:
            mask = Image.open(masks[gap*i+id]).convert('RGB')           # b n-1 c h w
            mask = self.ToTensor(self.CenterCrop(self.CenterPad_mask(mask))) / 255.
            #print("mask:",mask.shape)
            mask_clip.append(mask.unsqueeze(0))
        
        return torch.cat(clip,dim=0), torch.cat(mask_clip,dim=0),[path,id]      #这里的img是常规输出，target是包含GT的字典


class Dataset_withflow:
    def __init__(self, img_folder, img_folder2,img_folder3, transform_video,num_frames,img_size = [216,384],crop_size=[216,384],
    mask_folder1=None,mask_folder2=None,flow_folder1=None,flow_folder2=None,flow_folder3=None,flowmask_folder1=None,
    flowmask_folder2=None,flowmask_folder3=None):
        self.img_folder = img_folder
        self.img_folder2 = img_folder2
        self.num_frames = num_frames
        self.img_pairs = self.parse_img(self.img_folder,maskroot=mask_folder1,flowroot=flow_folder1,flowmaskroot=flowmask_folder1,gap = 1)
        self.img_pairs += self.parse_img(self.img_folder2,maskroot=mask_folder2,flowroot=flow_folder2,flowmaskroot=flowmask_folder2,gap = 1)
        self.img_pairs += self.parse_img(img_folder3,flowroot=flow_folder3,flowmaskroot=flowmask_folder3,gap = 1 ,head_gap = 8)  # wodouble wotrans
        self.transform_video = transform_video
        self.CenterPad = T.CenterPad([img_size[0],img_size[1]])
        self.CenterPad_vec = T.CenterPad_vec([img_size[0],img_size[1]])
        self.CenterPad_mask = T.CenterPad_mask([img_size[0],img_size[1]])
        self.CenterCrop = CenterCrop(crop_size)
        #print("in davis,cropsize:",crop_size)
        self.CenterCrop_np = CenterCrop_np(crop_size)
        self.ToTensor = T.ToTensor()


    def parse_img(self,data_root,maskroot=None,flowroot=None,flowmaskroot=None,gap = 1,head_gap=3):  # 3 wodouble wotrans
        image_pairs = []
        subdirs = sorted(os.listdir(data_root))
        for subdir in subdirs:
            path = os.path.join(data_root, subdir)
            if not os.path.isdir(path):
                print("img folder",path,"is not a correct folder!")
                continue
            imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
            img_len = len(imgs)
            gaped_len = 2 * gap * (self.num_frames-1)+1
            if img_len < gaped_len:
                print("img folder",path,"doesn't have enough imgs!")
                temp_gap = img_len // self.num_frames
                temp_gaped_len = temp_gap * (self.num_frames-1)+1
                if maskroot:
                    maskpath = os.path.join(maskroot, subdir)
                else:
                    maskpath = None
                flowpath = os.path.join(flowroot, subdir)
                flowmaskpath = os.path.join(flowmaskroot, subdir)
                for i in range(img_len-temp_gaped_len+1):
                    image_pairs.append([path,i,maskpath,flowpath,flowmaskpath,temp_gap])
                continue
            if maskroot:
                maskpath = os.path.join(maskroot, subdir)
            else:
                maskpath = None
            flowpath = os.path.join(flowroot, subdir)
            flowmaskpath = os.path.join(flowmaskroot, subdir)
            for i in range(0,img_len-gaped_len+1,head_gap):
                image_pairs.append([path,i,maskpath,flowpath,flowmaskpath,gap])
        return image_pairs

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        path , id,maskpath,flowpath,flowmaskpath,gap = self.img_pairs[idx]
        imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
        if maskpath:
            masks = sorted(glob.glob(os.path.join(maskpath, '*.png'))+glob.glob(os.path.join(maskpath, '*.jpg')))
        flows= sorted(glob.glob(os.path.join(flowpath, '*.flo')))
        flowmasks = sorted(glob.glob(os.path.join(flowmaskpath, '*.png')))
        # try:
        #     print("try problems in dataset:",flowpath,"|",len(flows),"|",flowmaskpath,"|",len(flowmasks))
        # except:
        #     print("except problems in dataset:",flowpath,"|",flowmaskpath)
        clip = []
        mask_clip = []
        flow_clip = []
        flowmask_clip = []
        #ref = Image.open(imgs[ np.random.randint(max(0,id-4), id+1) ]).convert('RGB')   #wotrans
        ref = Image.open(imgs[id]).convert('RGB')
        ref = self.transform_video(self.CenterCrop(self.CenterPad(ref)))
        img_len = len(imgs) - id -1
        clip_length = min(img_len // gap,11)
        #clip_length = img_len
        clip.append(ref.unsqueeze(0))
        for i in range(1,clip_length):
            img = Image.open(imgs[gap*i+id]).convert('RGB')
            #img = self.transform_video(self.CenterPad(img))
            img = self.transform_video(self.CenterCrop(self.CenterPad(img)))
            #img_1 = img[:,:,0:1,:,:]                # B N C H W
            #img_ab = img[:,:,1:3,:,:]
            #samples_rgb_from_gray = gray2rgb_batch(samples_1)
            clip.append(img.unsqueeze(0)) 
            if maskpath:
                mask = Image.open(masks[gap*i+id]).convert('RGB')           # b n-1 c h w
                mask = self.ToTensor(self.CenterCrop(self.CenterPad_mask(mask))) / 255.
            else:
                mask = torch.ones_like(img)
            #print("mask:",mask.shape)
            mask_clip.append(mask.unsqueeze(0))
            flow = read_flow(flows[gap*(i-1)+id])
            flowmask = Image.open(flowmasks[gap*(i-1)+id])
            flow = self.ToTensor(self.CenterCrop_np(self.CenterPad_vec(flow)))
            flowmask = self.ToTensor(self.CenterCrop(self.CenterPad(flowmask)))
            flowmask[flowmask<125]=0
            flowmask[flowmask>=125]=1
            flow_clip.append(flow.unsqueeze(0))
            flowmask_clip.append(flowmask.unsqueeze(0))

        
        return torch.cat(clip,dim=0), torch.cat(mask_clip,dim=0),torch.cat(flow_clip,dim=0),torch.cat(flowmask_clip,dim=0),[path,id]      #这里的img是常规输出，target是包含GT的字典


transform_video = [
        T.RGB2Lab(),
        T.ToTensor(),
        T.Normalize(),
    ]

def build_dataset(image_set, args):                                #imageset = train
    img_folder1 = Path(args.davis_path)
    img_folder2 = Path(args.fvi_path)
    img_folder3 = Path(args.videvo_path)
    mask_folder1 = args.davis_mask
    mask_folder2 = args.fvi_mask
    assert img_folder1.exists(), f'provided DAVIS path {root} does not exist'
    # PATHS = {
    #     "train": (root / "Train" ,root2 / "Train" ),
    #     "val": (root / "Val" , root2 / "test"),
    # }
    #img_folder1 , img_folder2 = PATHS[image_set]
    # dataset = Dataset(img_folder1, img_folder2,transform_video=T.Compose(transform_video), 
    # num_frames = args.num_frames,img_size = args.img_size,crop_size=args.crop_size,
    #                 mask_folder1=mask_folder1,mask_folder2=mask_folder2)
    flow_folder1 = args.davis_flow
    flowmask_folder1 = args.davis_flowmask
    flow_folder2 = args.fvi_flow
    flowmask_folder2 = args.fvi_flowmask
    flow_folder3 = args.videvo_flow
    flowmask_folder3 = args.videvo_flowmask
    dataset = Dataset_withflow(img_folder1, img_folder2,img_folder3,transform_video=T.Compose(transform_video), 
    num_frames = args.num_frames,img_size = args.img_size,crop_size=args.crop_size,
    mask_folder1=mask_folder1,mask_folder2=mask_folder2,flow_folder1=flow_folder1,flow_folder2=flow_folder2,
    flow_folder3=flow_folder3,flowmask_folder1=flowmask_folder1,flowmask_folder2=flowmask_folder2,flowmask_folder3=flowmask_folder3)
    return dataset
