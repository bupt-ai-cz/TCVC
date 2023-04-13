import argparse
import glob
import os

import numpy as np
from torchvision import models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:0")

def transform_img(image_size):
    transform = transforms.Compose([            
    transforms.Resize(image_size),                    
    transforms.CenterCrop(image_size),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])
    return transform

def got_gt_class(analogy_path,val_10000_path):
    rf = np.load(analogy_path)
    imgs_10k = sorted(glob.glob(os.path.join(val_10000_path, '*.png')))
    class_gts=np.empty([10000,2]).astype('str')
    for i in range(10000):
        #print("processing:",imgs_10k[i])
        No = imgs_10k[i].split("_")[-1]
        #print("No:",No)
        No = int(No.split(".")[0])-1
        #print("No:",No)
        rf_corr = rf[No]
        #print("rf_corr:",rf_corr)
        class_gt = rf_corr[1]
        #print("class",class_gt)
        class_gts[i]=[rf_corr[0],class_gt]
    np.save("./class_gt.npy", class_gts)
    return class_gts

def got_saved_gt_classes(class_path):
    gt = np.load(class_path)
    return gt


class VideosDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        analogy_path = "/dataset/ImageNet_val/pairs/analogies/analogies_val.npy",
        ctest10k_path = "/dataset/ImageNet_val/ctest10k",
        class_path = "utils/class_gt.npy"
    ):
        self.imgs_val = sorted(glob.glob(os.path.join(data_root, '*.png')))
        #self.classes_gt = got_gt_class(analogy_path,ctest10k_path)
        #self.classes_gt = got_saved_gt_classes(class_path)
        self.transforms_imagenet = transform_img(image_size)
        self.image_size = image_size
        self.real_len = len(self.imgs_val)
        
    def __getitem__(self, index):
        try:
            image_a_path = self.imgs_val[index]
            I1 = Image.open(image_a_path)#.convert("L").convert("RGB")
            I1 = self.transforms_imagenet(I1)
            # gt_class = self.classes_gt[index]
            # gt_class = np.array(gt_class[1])
            # print("gt:",gt_class)
            outputs = [
                I1,
                index
            ]
        except Exception as e:
            print("problem in, ", image_a_path)
            print(e)
            return self.__getitem__(0)
            #return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len


def got_class(index):   # 1000
    with open('utils/imagenet_classes.txt', 'r') as f:
        class_id_to_key = f.readlines()
    class_id_to_key = [x.strip() for x in class_id_to_key]
    keys = [class_id_to_key[id] for id in index]
    return keys

def get_top1_5(path,batch_size=32,num_workers=4,image_size=224):
    dataset_imagenet = VideosDataset(path,image_size)
    data_loader = DataLoader(
        dataset_imagenet,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    resnet = models.resnet101(pretrained=True)
    # Second, put the network in eval mode
    resnet.eval()
    resnet.cuda()
    class_path = "utils/class_gt.npy"
    classes_gt = got_saved_gt_classes(class_path)
    top_1=0
    top_5=0
    for id , data in enumerate(tqdm(data_loader)):
        [img,img_id] = data
        with torch.no_grad():
            img = img.cuda()
            out = resnet(img)  # bs * 1000
            # Forth, print the top 5 classes predicted by the model   bs * 1000
            value, index = torch.topk(out,5)  # bs * 5
            index = index.cpu()
            for i in range(len(index)):  # every batch
                key_batch = got_class(index[i])
                key_gt = classes_gt[img_id[i]][1]
                if key_gt == key_batch[0]:
                    top_1+=1
                if key_gt in key_batch:
                    top_5+=1
                # else:
                #     print("mismatch:",key_batch,key_gt)
    print("result: top-1 ",top_1 / 10000,"top-5",top_5 / 10000)
    return top_1 / 10000 , top_5 / 10000



if __name__ == '__main__':
    #dir(models)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_size", type=int, default=224, help="the image size, eg. [216,384]")   # 544 960
    parser.add_argument("--clip_path", type=str, default='/dataset/checkpoints/spcolor/val_10000/stego/1', help="path of input clips")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    opt = parser.parse_args()

    dataset_imagenet = VideosDataset(opt.clip_path,opt.image_size)
    data_loader = DataLoader(
        dataset_imagenet,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    resnet = models.resnet101(pretrained=True)
    # Second, put the network in eval mode
    resnet.eval()
    resnet.to(device)
    class_path = "utils/class_gt.npy"
    classes_gt = got_saved_gt_classes(class_path)
    top_1=0
    top_5=0
    for id , data in enumerate(tqdm(data_loader)):
        [img,img_id] = data
        with torch.no_grad():
            img = img.to(device)
            out = resnet(img)  # bs * 1000
            # Forth, print the top 5 classes predicted by the model   bs * 1000
            value, index = torch.topk(out,5)  # bs * 5
            index = index.cpu()
            for i in range(len(index)):  # every batch
                key_batch = got_class(index[i])
                key_gt = classes_gt[img_id[i]][1]
                if key_gt == key_batch[0]:
                    top_1+=1
                if key_gt in key_batch:
                    top_5+=1
                # else:
                #     print("mismatch:",key_batch,key_gt)
    print("result: top-1 ",top_1 / 10000,"top-5",top_5 / 10000)

        
            




