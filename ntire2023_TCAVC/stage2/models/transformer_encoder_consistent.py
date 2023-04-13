import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sys
from models.mhattention import MultiheadAttention_VC
from models.transformer_inter import TransformerInternLayer


class vc_layer(nn.Module):
    def __init__(self, d_model_cnn , d_model_trans, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention_VC(d_model_trans, nhead, dropout=dropout)         # batch_size, seq_length, d_model
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model_trans, dim_feedforward)                              # nn.Linear 只改变最后一位的维度
        self.linear2 = nn.Linear(dim_feedforward, d_model_trans)
        self.d_model_trans = d_model_trans
        self.d_model_out = d_model_cnn
        self.d_model = d_model_cnn
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(d_model_cnn, d_model_cnn,3,dilation=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(d_model_cnn),
            nn.ReplicationPad2d(1),    #nn.ReflectionPad2d(1),
            nn.Conv2d(d_model_cnn, d_model_cnn,3),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(d_model_cnn),
            nn.ReplicationPad2d(2),
            nn.Conv2d(d_model_cnn, d_model_cnn,3,dilation=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(d_model_cnn),
        )
        self.fuse_cnn2trans = nn.Sequential(
            nn.Linear(d_model_cnn+d_model_trans, d_model_trans),
        )
        self.fuse_trans2cnn = nn.Sequential(
            nn.Conv2d(d_model_cnn+d_model_trans, d_model_cnn,1),
        )
        self.instancenorm = nn.InstanceNorm2d(d_model_cnn)
        self.norm1 = nn.LayerNorm(d_model_trans)
        self.norm2 = nn.LayerNorm(d_model_trans)
        #self.pos_alpha = nn.Parameter(torch.Tensor([1.,1.]))

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def without_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor - self.pos_alpha*pos
    def forward(self,
                     src_cnn,src_trans,
                     shape,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     temperature=1,
                     ):
        b,n,h,w = shape
        src_conv = self.conv(src_cnn)
        src_cnn2trans = src_cnn.view(b,n,self.d_model,h,w
                            ).permute(1,3,4,0,2).flatten(0,2)     # bn c h w  -- nhw b c 
        src_trans = self.fuse_cnn2trans(torch.cat([src_trans,src_cnn2trans],dim=2))
        src_trans_norm = self.norm1(src_trans)

        # q = k = self.with_pos_embed(src_trans_norm, pos)                                # nhw b c
        q = self.with_pos_embed(src_trans_norm, pos) 
        k = self.with_pos_embed(src_trans_norm, pos) 
        src_trans2 = self.self_attn(q, k, value=src_trans_norm, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,temperature=temperature,need_weights=True)[0]
        src_trans = src_trans + self.dropout1(src_trans2)
        src_trans2 = self.norm2(src_trans)
        src_trans2 = self.linear2(self.dropout(self.activation(self.linear1(src_trans2))))
        src_trans = src_trans + self.dropout2(src_trans2)

        src_trans2cnn = src_trans.view(n,h,w,b,self.d_model_trans
                            ).permute(3,0,4,1,2).flatten(0,1)
        src_trans2cnn = self.fuse_trans2cnn(torch.cat([src_conv,src_trans2cnn],dim=1))  # bn c h w
        src_cnn = self.instancenorm(src_cnn + src_trans2cnn)
        del q,k,
        return src_cnn,src_trans

class Transformer_encoder_parallel(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):                                #True
        super().__init__()

        encoder_layer = vc_layer(512,384,nhead,2048,dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self._reset_parameters()
        self.d_model = d_model   #384
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed,features_inter):
        # flatten NxCxHxW to HWxNxC                                   # batch_size, seq_length, d_model
        b, n, c,h,w = src.shape                                       # b c n hw  -> b c nhw
        #src = src.flatten(2).permute(2, 0, 1)                         #  nhw b c
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)             #  nhw b c
        mask = mask.flatten(1)                                         # b n hw -> b nhw

        src ,src_trans= self.encoder(src,features_inter, src_key_padding_mask=mask, pos=pos_embed)
        del pos_embed,mask
        return src,src_trans #.view(b,n, 512, h,w).permute(0,2,1,3,4).flatten(-2)                #  nhw b c --  b c nhw  -- b c n hw

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.project2cnn = nn.Conv2d(2048,512,1)
        self.project2trans = nn.Linear(2048,384)
        self.interlayer_request = TransformerInternLayer(384,384,4)

    def forward(self, src,features_inter,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        b,n,c,h,w = src.shape
        src_trans = self.project2trans(src.permute(1,3,4,0,2).flatten(0,2))
        if features_inter != None:
            #print("here requist")
            src_trans = self.interlayer_request(src_trans,features_inter)
        src = self.project2cnn( src.flatten(0,1))
        for layer in self.layers:
            src,src_trans = layer(src,src_trans,[b,n,h,w], src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        #del src_trans
        return src,src_trans

# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = MultiheadAttention_VC(d_model, nhead, dropout=dropout,add_bias_kv=False)         # batch_size, seq_length, d_model
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)                              # nn.Linear 只改变最后一位的维度
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None):
#         #print("debug:", src.shape,pos.shape)
#         q = k = self.with_pos_embed(src, pos)                                # nhw b c
#         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         del src2 ,q,k
#         return src

#     def forward_pre(self, src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         del src2 ,q,k
#         return src

#     def forward(self, src,
#                 src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])   # 深拷贝相互独立

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {activation}.")