"""
VisTR model and criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
                       

from .backbone_vec import build_backbone

from .transformer_colorvid import build_transformer
from .transformer_inter  import TransformerInternLayer

class VCTran(nn.Module):

    def __init__(self, backbone, transformer, num_frames,  aux_loss=False):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)              #  backbone.num_channels=2048        hidden_dim = 
        self.backbone = backbone

    def forward(self, samples,features_inter=None):        #输入确实需要是  N C H W 的格式，这一步是从哪里变过来的？
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        pos_decode = [pos[-3],pos[-1]]
        #pos_decode = pos[-3]
        #print("debug:",pos_decode[0].shape,pos_decode[1].shape)
        pos = pos[-1]                                                          #[-1]取什么,取了最后一层的特征
        src, mask = features[-1].decompose()
        src2 , _  = features[-2].decompose()
        #print("src shape before encoder:",src.shape,src2.shape)
        src = torch.cat((src,src2),dim=1)                         # bn c h w
        #print("src shape before encoder:",src.shape)             #torch.Size([36, 2048, 8, 8])    torch.Size([36, 2048, 16, 16])  
        #src_proj = self.input_proj(src)                          # n c h w
        n,c,h,w = src.shape
        assert mask is not None
        src = src.reshape(n//self.num_frames, self.num_frames, c, h, w)   #  b n c h w 
        mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)                #    b n hw
        pos = pos.permute(0,2,1,3,4).flatten(-2) 
        #pos_decode = pos_decode.permute(1,3,4,0,2).flatten(0,2)                                    #   b n c h w --  b c n h w        -- b c n hw
        pos_decode[0] = pos_decode[0].permute(1,3,4,0,2).flatten(0,2)
        pos_decode[1] = pos_decode[1].permute(1,3,4,0,2).flatten(0,2)
        #print("debug2:",pos_decode[0].shape,pos_decode[1].shape)
        #print("src shape just before encoder:",src_proj.shape)                       #     torch.Size([1, 384, 36, 64])
        hs,hs_trans = self.transformer(src, mask, pos,features_inter)           # [0]取什么 取原decoder的输出，1是原encoder的输出？
        del src2 ,mask,pos
        return   hs,hs_trans.detach() , features[:-2],pos_decode


def build_model(args):

    backbone = build_backbone(args)            #resnet

    transformer_encoder , transformer_decoder = build_transformer(args)      #transformer

    model = VCTran(
        backbone,
        transformer_encoder,
        num_frames=args.num_frames,
        
    )

    return model, transformer_decoder
