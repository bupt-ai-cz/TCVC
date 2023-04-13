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
import yaml
from tqdm import tqdm
#from utils.util_distortion import CenterPadCrop_numpy, Distortion_with_flow, Normalize, RGB2Lab, ToTensor

if __name__ == "__main__":

    dir = "/dataset/ImageNet_train/"
    dir = osp.expanduser(dir)

    yampath = osp.join(dir,"gray_imgs.yaml")
    print("saving yaml to:",yampath)
    file_gray=open(yampath,'w',encoding='utf-8')
    yampath_small = osp.join(dir,"small_imgs.yaml")
    file_small=open(yampath_small,'w',encoding='utf-8')
    yampath_mono = osp.join(dir,"mono_imgs.yaml")
    file_mono=open(yampath_mono,'w',encoding='utf-8')
    yampath_error = osp.join(dir,"error_imgs.yaml")
    file_error=open(yampath_error,'w',encoding='utf-8')
    gray_dict={}
    small_dict={}
    mono_dict={}
    error_dict={}

    # dir_analogy = osp.join(dir,"pairs/analogies")
    # analogies = sorted(glob.glob(os.path.join(dir_analogy, '*.npy')))
    data_path = osp.join(dir,"imgs")
    subdirs = sorted(os.listdir(data_path))
    for i in tqdm(range(len(subdirs))):    # go through 1000 categories  len(analogies)  len(subdirs)
        subdir = subdirs[i]
        # analogy_name = osp.basename(analogy_path)
        # analogy = np.load(analogy_path,mmap_mode = 'r')
        analogy_name = "analogies_%s.npy" % subdir
        print("subdir:",subdir,"analogy:", analogy_name)
        gray_dict[analogy_name]=[]
        small_dict[analogy_name]=[]
        mono_dict[analogy_name]=[]
        error_dict[analogy_name]=[]
        bad_img = 0
        # for j in range(len(analogy)):   # go through lines in analogy_nxxxxx.npy
        #     pair = analogy[j]
        img_paths = sorted(glob.glob(os.path.join(data_path,subdir,'*.JPEG')))
        print("len img_path:",len(img_paths))
        for r in range(len(img_paths)):
            img_path = img_paths[r]
            ref_img = img_path.split("imgs/")[1]
            img = Image.open(img_path)
            try:
                if img.layers == 1:
                    #if ref_img not in gray_dict[analogy_name]:
                        #print("gray img in:",ref_img)
                    gray_dict[analogy_name].append(ref_img)
                    bad_img+=1
                h, w = img.size
                if h < 256 or w < 256:
                    #if ref_img not in small_dict[analogy_name]:
                    small_dict[analogy_name].append(ref_img)
                    #print("too small in:",ref_img)
                    bad_img+=1
                v = img.histogram()
                percentage_monochrome = max(v) / float(h*w)
                if percentage_monochrome > 0.8:
                    #if ref_img not in mono_dict[analogy_name]:
                    mono_dict[analogy_name].append(ref_img)
                    #print("monochrome in:",ref_img)
                    bad_img+=1

            except Exception as e:
                print("problem in, ", img_path)
                error_dict[analogy_name].append(ref_img)
                print(e)
        print("bad imgs:",bad_img)            
        
    yaml.dump(gray_dict,file_gray)
    yaml.dump(small_dict,file_small)
    yaml.dump(mono_dict,file_mono)
    yaml.dump(error_dict,file_error)
    file_gray.close()
    file_small.close()
    file_mono.close()
    file_error.close()

def filter_bad():
    good_images = []
    bad_images = []
    for filename in images:
        try:
            img = Image.open(filename)
            # pixel distribution
            v = img.histogram()
            h, w = img.size
            percentage_monochrome = max(v) / float(h*w)

            # filter bad and small images
            if percentage_monochrome > 0.8 or h < 300 or w <300:
                bad_images.append(filename)
            else:
                good_images.append(filename)
        except:
            pass
    print("Number of good images: {}\n".format(len(good_images)))
    print("Number of bad images: {}\n".format(len(bad_images)))
    return good_images, bad_images





