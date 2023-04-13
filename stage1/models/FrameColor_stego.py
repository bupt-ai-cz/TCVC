import torch
from utils.util import *

def frame_colorization(
    IA_lab,
    IB_lab,
    color_prior,
    ref_lab,
    # cluster_value_current,
    # cluster_preds_current,
    # cluster_preds_ref,
    vggnet,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        out_tensor_warp=torch.cat((color_prior,IB_lab,IA_l),dim=1)
        IA_ab_predict = colornet(out_tensor_warp)

    return IA_ab_predict

