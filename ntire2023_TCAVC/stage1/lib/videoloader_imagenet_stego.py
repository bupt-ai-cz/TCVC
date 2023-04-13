import os
import os.path as osp
import struct

import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils.util_distortion import CenterPadCrop_numpy,CenterPad, Distortion_with_flow, Normalize, RGB2Lab, ToTensor
import yaml
from traceback import print_exc

cv2.setNumThreads(0)

def pil_loader_new(path):
    img = Image.open(path)
    if len(img.getbands()) == 1:
        #print("gray img in :",path)
        return 0,0
    return img.convert("RGB"),1

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def parse_video_test(dir):
    print("video dir is: ", dir)
    dir = osp.expanduser(dir)
    image_pairs = []
    subdirs = sorted(os.listdir(dir))
    if 'train_gt' in dir:
        subdirs = subdirs[180:200]
    for subdir in subdirs:
        path = os.path.join(dir, subdir)
        if not os.path.isdir(path):
            print("img folder",path,"is not a correct folder!")
            continue
        imgs = sorted(glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg')))
        item =  (subdir, imgs,"NTIRE")
        image_pairs.append(item)
    #f_gray.close()
    return image_pairs

class VideosDataset_ImageNet_10k(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        transforms_imagenet=None,
        loadmode = "ImageNet"
    ):
        image_pairs = []
        curr_video_pairs = parse_video_test(data_root)
        image_pairs += curr_video_pairs
        if not image_pairs:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        #self.ToTensor = ToTensor()
        #self.Normalize = Normalize()
        self.transforms_raft = transforms.Compose([CenterPad(image_size),transforms.ToTensor()])
        self.transforms_stego = transforms.Compose([CenterPad(image_size),transforms.ToTensor(),normalize])
        
    def __getitem__(self, index):
        try:
            name, image_a_path, loadmode = self.image_pairs[index]

            img_path = image_a_path[0]
            I1,flag = pil_loader_new(img_path)
        
            I1_lab= self.transforms_imagenet(I1)
            I1_rgb= self.transforms_raft(I1)
                
            outputs = [
                I1_lab,
                I1_rgb,
                name,
            ]

            
        except Exception as e:
            print("problem in, ", image_a_path,)
            print_exc()
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len


