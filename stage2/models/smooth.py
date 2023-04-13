import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util_vec import uncenter_l


def find_local_patch(x, patch_size):
    N, C, H, W = x.shape
    x_unfold = F.unfold(
        x, kernel_size=(patch_size, patch_size), padding=(patch_size // 2, patch_size // 2), stride=(1, 1)
    )

    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage_color(nn.Module):
    """
    smooth the image according to the color distance in the LAB space
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color, self).__init__()

    def forward(self, x_l,x_ab, x_ab_predict, patch_size=3, alpha=1, scale_factor=1):    # x_l is uncentered
        """ alpha=0: less smooth; alpha=inf: smoother """
        #x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = x_l
        a = x_ab[:, 0:1, :, :]                                     # bn c h w
        b = x_ab[:, 1:2, :, :]
        a_predict = x_ab_predict[:, 0:1, :, :]
        b_predict = x_ab_predict[:, 1:2, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_a_predict = find_local_patch(a_predict, patch_size)
        local_b_predict = find_local_patch(b_predict, patch_size)

        local_color_difference = (local_l - l) ** 2 + (local_a - a) ** 2 + (local_b - b) ** 2
        correlation = nn.functional.softmax(
            -1 * local_color_difference / alpha, dim=1
        )  # so that sum of weights equal to 1

        return torch.cat(
            (
                torch.sum(correlation * local_a_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_b_predict, dim=1, keepdim=True),
            ),
            1,
        )

class WeightedAverage_color_vec(nn.Module):
    """
    smooth the image according to the color distance in the LAB space
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color, self).__init__()

    def forward(self, x_lab, x_lab_predict, patch_size=3, alpha=1, scale_factor=1):    # x_l is uncentered
        """ alpha=0: less smooth; alpha=inf: smoother """
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = x_lab[:, 0:1, :, :]
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        a_predict = x_lab_predict[:, 1:2, :, :]
        b_predict = x_lab_predict[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_a_predict = find_local_patch(a_predict, patch_size)
        local_b_predict = find_local_patch(b_predict, patch_size)

        local_color_difference = - (local_l - l) ** 2 - (local_a - a) ** 2 - (local_b - b) ** 2
        N = patch_size **2

        correlation = nn.functional.softmax(
             local_color_difference / alpha, dim=1
        )  # so that sum of weights equal to 1

        return torch.cat(
            (
                torch.sum(correlation * local_a_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_b_predict, dim=1, keepdim=True),
            ),
            1,
        )

class WeightedAverage_color_temporal(nn.Module):
    """
    smooth the image according to the color distance along the temporal channel
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color_temporal, self).__init__()

    def forward(self, x_lab, x_ab_predict, alpha=1):    # x_l is uncentered   input b n c h w
        """ alpha=0: less smooth; alpha=inf: smoother """
        b,n,_,h,w = x_lab.shape
        difference = torch.Tensor(b,n,n,3,h,w).cuda()
        for i in range(n):
            difference[:,i,:,:,:,:]= (x_lab - x_lab[:,i:i+1,:,:,:])**2    # b n 1 h w
        difference = torch.sum(difference,dim=3,keepdim=True)

        correlation = F.softmax(            # b n n 1 h w
            -1 * difference / alpha, dim=2
        )  # so that sum of weights equal to 1

        return torch.sum(correlation.repeat(1,1,1,2,1,1) * x_ab_predict.unsqueeze(1).repeat(1,n,1,1,1,1), dim=2).flatten(0,1)  # b n n 1 h w * b 1 n 2 h w -- b n 2 h w
            