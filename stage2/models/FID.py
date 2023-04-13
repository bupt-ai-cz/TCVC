# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the Fréchet Inception Distance (FID) metric."""
import numpy as np
import torch
import torch.nn as nn
from pytorch_fid.fid_score import calculate_frechet_distance#,compute_statistics_of_path ,calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
import lpips


class LPIP_utils:
    def __init__(self,device = 'cuda'):
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True)  # Can set net = 'squeeze' or 'vgg'or 'alex'
        self.loss_fn = self.loss_fn.to(device)
        self.device = device
    def compare_lpips(self,img_fake,img_real,data_range=255.):         # input: torch 1 c h w    / h w c
        if not ( isinstance(img_fake, torch.FloatTensor) and isinstance(img_real, torch.FloatTensor)) :
            img_fake = torch.FloatTensor(img_fake)
            img_real = torch.FloatTensor(img_real)
        if img_fake.ndim==3:
            img_fake = img_fake.permute(2,0,1).unsqueeze(0)
            img_real = img_real.permute(2,0,1).unsqueeze(0)
        img_fake = img_fake.to(self.device)
        img_real = img_real.to(self.device)
        img_fake /= data_range
        img_real /= data_range
        
        dist = self.loss_fn.forward(img_fake,img_real)
        return dist.mean().detach().cpu().numpy()
    def compare_lpips_batch(self,img_fake,img_real,data_range=255.):         # input: torch b c h w
        #if not ( isinstance(img_fake, np.ndarray) and isinstance(img_real, np.ndarray)) :
        #    img_fake = np.array(img_fake).astype(np.float32).transpose((2, 0, 1))   # c h w
        #    img_real = np.array(img_real).astype(np.float32).transpose((2, 0, 1)) 
        #img_fake = torch.from_numpy(img_fake)
        #img_real = torch.from_numpy(img_real)
        img_fake /= data_range
        img_real /= data_range
        dist_=[]
        loss_fn = lpips.LPIPS(net='vgg', spatial=True)  # Can set net = 'squeeze' or 'vgg'or 'alex'
        for i in range(len(img_fake)):
            dist = loss_fn.forward(img_fake[i],img_real[i])
            dist_.append(dist.mean().item())
        return  dist_ , sum(dist_)/len(img_fake)







class FID_utils(nn.Module):
    """Class for computing the Fréchet Inception Distance (FID) metric score.
    It is implemented as a class in order to hold the inception model instance
    in its state.
    Parameters
    ----------
    resize_input : bool (optional)
        Whether or not to resize the input images to the image size (299, 299)
        on which the inception model was trained. Since the model is fully
        convolutional, the score also works without resizing. In literature
        and when working with GANs people tend to set this value to True,
        however, for internal evaluation this is not necessary.
    device : str or torch.device
        The device on which to run the inception model.
    """

    def __init__(self, resize_input=True, device="cuda"):
        super(FID_utils, self).__init__()
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model = InceptionV3(resize_input=resize_input).to(device)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device)
        self.model = self.model.eval()

    def get_activations(self,batch):                   # 1 c h w
        with torch.no_grad():
            pred = self.model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            #pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            print("error in get activations!")
        #pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        return pred


    def _get_mu_sigma(self, batch,data_range):
        """Compute the inception mu and sigma for a batch of images.
        Parameters
        ----------
        images : np.ndarray
            A batch of images with shape (n_images,3, width, height).
        Returns
        -------
        mu : np.ndarray
            The array of mean activations with shape (2048,).
        sigma : np.ndarray
            The covariance matrix of activations with shape (2048, 2048).
        """
        # forward pass
        if not isinstance(batch, torch.FloatTensor):
            batch = torch.FloatTensor(batch)
        if batch.ndim ==3 and batch.shape[2]==3:
            batch=batch.permute(2,0,1).unsqueeze(0) 
        batch /= data_range           
        #batch = torch.tensor(batch)#.unsqueeze(1).repeat((1, 3, 1, 1))
        batch = batch.to(self.device, torch.float32)
        #(activations,) = self.model(batch)
        activations = self.get_activations(batch)
        activations = activations.detach().cpu().numpy().squeeze(3).squeeze(2)

        # compute statistics
        mu = np.mean(activations,axis=0)
        sigma = np.cov(activations, rowvar=False)

        return mu, sigma

    def score(self, images_1, images_2,data_range=255.):
        """Compute the FID score.
        The input batches should have the shape (n_images,3, width, height). or (h,w,3)
        Parameters
        ----------
        images_1 : np.ndarray
            First batch of images.
        images_2 : np.ndarray
            Section batch of images.
        Returns
        -------
        score : float
            The FID score.
        """
        mu_1, sigma_1 = self._get_mu_sigma(images_1,data_range)
        mu_2, sigma_2 = self._get_mu_sigma(images_2,data_range)
        score = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

        return score
    #def score_path(self, path):
    #    """Compute the FID score.
    #    The input batches should have the shape (n_images,3, width, height). or (h,w,3)
    #    Parameters
    #    ----------
    #    images_1 : np.ndarray
    #        First batch of images.
    #    images_2 : np.ndarray
    #        Section batch of images.
    #    Returns
    #    -------
    #    score : float
    #        The FID score.
    #    """
    #    device = torch.device('cuda')
    #    #mu_1, sigma_1 = compute_statistics_of_path(path[0], self.model, 1, 2048, device,)
    #    #mu_2, sigma_2 = compute_statistics_of_path(path[1], self.model, 1, 2048, device,)
    #    #score = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
    #    score = calculate_fid_given_paths(path,
    #                                      1,
    #                                      device,
    #                                      2048,
    #                                      1)

    #    return score