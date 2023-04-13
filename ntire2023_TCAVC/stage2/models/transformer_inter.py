import torch
import torch.nn.functional as F
from torch import nn,Tensor
from models.mhattention import MultiheadAttention_VC
from typing import Optional, List

class TransformerInternLayer(nn.Module):

    def __init__(self, d_model , d_model_out, nhead, dim_feedforward=2048, dropout=0.1,):
        super().__init__()
        self.self_attn = MultiheadAttention_VC(d_model, nhead, dropout=dropout)         # batch_size, seq_length, d_model
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)                              # nn.Linear 只改变最后一位的维度
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model_out)
        self.norm1 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_model_out = d_model_out
        self.norm2 = nn.LayerNorm(d_model_out)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,value,                                 # src: intern features  | value: features_conv31
                     src_mask: Optional[Tensor] = None,         # src: features_fuse1   | value: intern features
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     temperature=1,
                     ):
        src2 = self.self_attn(src, value, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,temperature=temperature,need_weights=True)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        del src2
        return src

