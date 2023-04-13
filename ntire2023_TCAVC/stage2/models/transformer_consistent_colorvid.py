"""
VisTR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sys
from models.mhattention import MultiheadAttention_VC
from models.transformer_encoder_consistent import Transformer_encoder_parallel

class Transformer_decoder_parallel(nn.Module):

    def __init__(self, d_model=512, nhead=8,head_warp = 2,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):                                #True
        super().__init__()
        warp_layer = TransformerMHWarpLayer(512,head=head_warp)
        # decoder_layers = nn.ModuleList()
        # decoder_layers.append(warp_layer)
        # decoder_layers.append(vc_layer(512,384, int(nhead/8), 1024, dropout, activation))
        # decoder_layers.append(vc_layer(512,384, int(nhead/8), 1024, dropout, activation))
        # decoder_layers.append(vc_layer(512,384, int(nhead/8), 1024, dropout, activation))
        self.decoder = TransformerDecoder_colorvid(TransformerDecoderLayer(384,384, int(nhead/8), 2048,
                                                 dropout, activation))
        self.warper = TransformerDecoder_warper(warp_layer)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.upsample = nn.Upsample(scale_factor=4)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,src,src_trans,ref_ab,features,tensor_1 ,pos = None,temperature_warp=0.01):
        memory ,warped_result,src_map = self.warper(src,ref_ab,features,pos=pos,temperature_warp=temperature_warp)
        memory = self.upsample(memory)
        memory  = self.decoder(memory,src_trans,tensor_1)
        return memory ,warped_result ,src_map        #  b n c h w

class vc_layer(nn.Module):
    def __init__(self, d_model_cnn , d_model_trans, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention_VC(d_model_trans, nhead, dropout=dropout)         # batch_size, seq_length, d_model
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model_trans, dim_feedforward)                              # nn.Linear 只改变最后一位的维度
        self.linear2 = nn.Linear(dim_feedforward, d_model_trans)
        self.d_model_trans = d_model_trans
        self.d_model_cnn = d_model_cnn
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
        self.layernorm = nn.LayerNorm(d_model_trans)
        self.layernorm2 = nn.LayerNorm(d_model_trans)
        #self.pos_alpha = nn.Parameter(torch.Tensor([1.,1.]))

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

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
        src_cnn2trans = src_cnn.view(b,n,self.d_model_cnn,h,w
                            ).permute(1,3,4,0,2).flatten(0,2)     # bn c h w  -- nhw b c 
        src_trans = self.fuse_cnn2trans(torch.cat([src_trans,src_cnn2trans],dim=2))
        src_trans_norm = self.layernorm(src_trans)

        #q = k = self.with_pos_embed(src_trans_norm, pos)                                # nhw b c
        q = self.with_pos_embed(src_trans_norm, pos) 
        k = self.with_pos_embed(src_trans_norm, pos) 
        src_trans2 = self.self_attn(q, k, value=src_trans_norm, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,temperature=temperature,need_weights=True)[0]
        src_trans = src_trans + self.dropout1(src_trans2)
        src_trans2 = self.layernorm2(src_trans)
        src_trans2 = self.linear2(self.dropout(self.activation(self.linear1(src_trans2))))
        src_trans = src_trans + self.dropout2(src_trans2)

        src_trans2cnn = src_trans.view(n,h,w,b,self.d_model_trans
                            ).permute(3,0,4,1,2).flatten(0,1)
        src_trans2cnn = self.fuse_trans2cnn(torch.cat([src_conv,src_trans2cnn],dim=1))  # bn c h w
        src_cnn = self.instancenorm(src_cnn + src_trans2cnn)
        del q,k,
        return src_cnn,src_trans


class ResBlock(nn.Module):
    def __init__(self,d_model,d_model_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model_out, kernel_size=3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model_out, d_model_out, kernel_size=3,padding = 1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(d_model_out),
        )
    def forward(self,x):
        return self.conv(x)+x


class TransformerDecoder_warper(nn.Module):
    
    def __init__(self, decoder_layers, norm=None):
        super().__init__()
        self.warp_layer  = decoder_layers
        self.project_feature1 = nn.Conv2d(512, self.warp_layer.d_model//4, kernel_size=1)
        self.project_feature2 = nn.Conv2d(256, self.warp_layer.d_model//4, kernel_size=1)
        self.fuse_conv1_1 = ResBlock(self.warp_layer.d_model, self.warp_layer.d_model)
        self.fuse_conv1_2 = ResBlock(self.warp_layer.d_model, self.warp_layer.d_model)
        self.fuse_conv1_3 = ResBlock(self.warp_layer.d_model, self.warp_layer.d_model)
        self.Conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), 
            nn.Conv2d(512, self.warp_layer.d_model//2, 3, 1, 1),
            nn.PReLU(),
            nn.InstanceNorm2d(256),
            )
        self.instancenorm = nn.InstanceNorm2d(128)
        #self.fuse1_alpha = nn.Parameter(torch.Tensor([0.5]))
    def forward(self, src,ref_ab,features,                     #  features bn c h w
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                temperature_vc=1,
                temperature_warp = 0.01):
        b, _ ,_,h,w = ref_ab.shape
        n = src.shape[0] // b                                          # bn c h w

        ref_ab = ref_ab.view(b*1,2,h,w) / 128.                     # b n c h w -- bn c h w
  #---------------------- fuse feature ----------------------------#
        h_temp = int(h/8)
        w_temp = int(w/8)                    # b c n hw -- b n c hw     -- bn c h w 
        #print("configure src shape:",src.shape)        #torch.Size([1, 384, 4, 576])             
        #src = src.permute(0,2,1,3).contiguous().view(b*n,512,h_temp,w_temp)          
        src = self.Conv_up(src)                                        # 4*4
        h_temp *= 2
        w_temp *= 2
        feature_temp = self.instancenorm(self.project_feature1(features[-1].tensors))                           # bn c h w   4*4
        feature_temp_2 = self.instancenorm(self.project_feature2(features[-2].tensors))
        feature_temp_2_down = F.interpolate(feature_temp_2,scale_factor=0.5)                #   2*2  -- 4*4
        src = torch.cat((feature_temp_2_down,feature_temp,src),dim=1)
        src1 = self.fuse_conv1_1(src)                                           # bn c h w
        src1 = self.fuse_conv1_2(src1) 
        src1 = self.fuse_conv1_3(src1)
        src1[:,256:512,:,:] = 0.2 * src1[:,256:512,:,:] + 0.8 *src[:,256:512,:,:] 
        ref_ab_temp = F.interpolate(ref_ab,scale_factor=1/4).view(b,1,2,h_temp,w_temp).permute(1,3,4,0,2).view(1*h_temp*w_temp,b,2)                                                              
  #----------------------warp----------------------------#
        src=src1.view(b,n,self.warp_layer.d_model,h_temp,w_temp).permute(0,2,1,3,4).view(b,self.warp_layer.d_model,n,h_temp*w_temp)
        src,warped_result,src_map = self.warp_layer(src,ref_ab_temp,pos=pos[0],temperature=temperature_warp)
  #-----------------------intermediate-------------------#
        src = src.view(n,h_temp,w_temp,b,self.warp_layer.d_model_out).permute(3,0,4,1,2).view(b*n,self.warp_layer.d_model_out,h_temp,w_temp)
        #src = torch.cat((src,src1),dim=1)         #bn c h w
        warped_result = warped_result.view(n-1,h_temp,w_temp,b,self.warp_layer.head*2).permute(3,0,4,1,2).contiguous().view(b*(n-1)*self.warp_layer.head,2,h_temp,w_temp)   #nhw b c --bn c h w
        
        return src , warped_result * 128,src_map         #-128 ~ 128  , 0~1

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model , d_model_out, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", use_layernorm=True):
        super().__init__()
        self.self_attn = MultiheadAttention_VC(d_model, nhead, dropout=dropout)         # batch_size, seq_length, d_model
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)                              # nn.Linear 只改变最后一位的维度
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model_out)
        #self.linear3 = nn.Linear(d_model, d_model_out)
        self.norm1 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_model_out = d_model_out
        self.norm2 = nn.LayerNorm(d_model_out)
        self.activation = _get_activation_fn(activation)
        # if use_layernorm==False:
        #     self.norm1 = nn.InstanceNorm1d(d_model)
        #     self.norm2 = nn.InstanceNorm1d(d_model_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     temperature=1,
                     ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)                                # nhw b c
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,temperature=temperature,need_weights=True)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        del q,k,src2
        return src

class TransformerMHWarpLayer(nn.Module):

    def __init__(self, d_model, head=2):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=2)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.d_model_out = 3*head
        self.head = head
        self.head_dim = self.d_model // self.head
        self.theta = nn.Linear(self.head_dim,self.head_dim)
        self.phi = nn.Linear(self.head_dim,self.head_dim)
        self.pos_embed = nn.Linear(384,self.head_dim)
        self.refine_net = nn.Sequential(
        nn.Linear(self.d_model+self.head,self.head),
        nn.PReLU()
        )
        #self.pos_alpha = nn.Parameter(torch.Tensor([1.,1.,1.,-1.]))
        #self.similarity_threshold = torch.nn.Parameter(torch.Tensor([0.01]))
        # self.mask_net = nn.Sequential(
        #     nn.Linear(self.head_dim,256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256,128),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(128),
        #     nn.Linear(128,64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64,1),
        #     nn.Sigmoid(),
        # )
    def RefineWarpedResult(self,results,src_sim,src_gray):  #  nhw b head*2   |  nhw b head  | nhw b c
        results_sim = self.refine_net(torch.cat((src_sim,src_gray),dim=2))+src_sim
        results_sim = results_sim - results_sim.mean(dim=(0,1),keepdim=True)           #center the distribution
        results_sim_norm = torch.norm(results_sim,2,(0,1),keepdim=True) + sys.float_info.epsilon
        results_sim = torch.div(results_sim,results_sim_norm)
        #results_sim = torch.cat([results_sim,self.similarity_threshold.expand_as(results_sim[:,:,0:1])],dim=2)
        results_sim = F.softmax(results_sim/0.01,dim=-1)
        #results_sim = results_sim[:,:,0:2] 
        results = results * results_sim[:,:,(0,0,1,1)]
        return results,results_sim * src_sim

    def nonlocal_operation(self,A_features,B_features,B_ab,temperature=0.01):    #  nhw b c  
        theta = self.theta(A_features).permute(1,0,2)  # 2*256*(feature_height*feature_width) -- b nhw c
        theta = theta - theta.mean(dim=-2, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, -1, keepdim=True) + sys.float_info.epsilon   #L2归一化后欧氏距离与余弦相似度等价
        theta = torch.div(theta, theta_norm)
        phi = self.phi(B_features).permute(1,2,0)  # 2*256*(feature_height*feature_width)  -- b c nhw
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, -2, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)                      # -- b c nhw
        f = torch.matmul(theta, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        similarity_map = torch.max(f, -1, keepdim=True)[0].permute(1,0,2)      # b nhw 1 -- nhw b 1

        # f can be negative
        f = f / temperature
        f = F.softmax(f , dim=-1)  # 2*1936*1936;
        B_ab = B_ab.permute(1,0,2)                      #nhw b c  --  b nhw c
        # multiply the corr map with color
        #print("evaluation:",f.shape,B_ab.shape)      #   torch.Size([1, 2352, 336]) torch.Size([1, 312, 2])
        y = torch.matmul(f, B_ab).permute(1,0,2)  # b n c -- nhw b c
        del f,theta,phi,B_ab
        return   y,similarity_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        #print("pos:",tensor.shape,pos.shape)
        return tensor if pos is None else tensor + pos
    def without_pos_embed(self, tensor, pos: Optional[Tensor]):
        #print("pos:",tensor.shape,pos.shape)
        return tensor if pos is None else tensor - pos
    def forward(self,
                     src,ref_ab,                                             # 
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     temperature = 0.01):
        src_ref = src[:,:,0:1,:]
        src_gray = src[:,:,1:,:]                                             # b c n hw
        src_ref = src_ref.flatten(2).permute(2, 0, 1)                         #  nhw b c
        src_gray = src_gray.flatten(2).permute(2, 0, 1)                         #  nhw b c
        q = src_gray
        k = src_ref
        n_q , b , c = q.shape                                            #nhw b c   gray
        n_k , _ , _ = k.shape                                            #          ref
        pos = self.pos_embed(pos)
        q = q.contiguous().view(n_q,b*self.head,self.head_dim)           # n b*head 
        k = k.contiguous().view(n_k,b*self.head,self.head_dim)
        # q[:,0:b,:]=self.with_pos_embed(q[:,0:b,:],pos[n_k:(n_q+n_k),:,:])
        # k[:,0:b,:]=self.without_pos_embed(k[:,0:b,:],pos[0:n_k,:,:])
        #---------------minus pe for head 2-------------#
        q[:,b:2*b,:] = self.with_pos_embed(q[:,b:2*b,:],pos[n_k:(n_q+n_k),:,:])
        k[:,b:2*b,:] = self.without_pos_embed(k[:,b:2*b,:],pos[0:n_k,:,:])
        v = ref_ab.repeat(1,self.head,1)                                 # nhw b c
        assert self.head_dim * self.head == self.d_model, "MHW:embed_dim must be divisible by num_heads"
        src_y , src_map  = self.nonlocal_operation(q,k,v,temperature=temperature)
        src_map = src_map.contiguous().view(n_q,b,self.head)             # nhw b*head 1 -- nhw b head
        src_y = src_y.contiguous().view(n_q,b,2*self.head)                        # nhw b*head 2 -- nhw b head*2
        #---------------refine warped result------------#
        #src_y,src_map = self.RefineWarpedResult(src_y,src_map,src_gray)
        #-----------------------------------------------#
        similarity_map_ref = torch.ones([n_k,b,self.head],device='cuda')    # n b 1
        similarity_map = torch.cat((similarity_map_ref,src_map),dim = 0)    # complete s  | nhw b c
        src = torch.cat((ref_ab.repeat(1,1,self.head),src_y),dim = 0)       # complete ab  |nhw b c
        src = torch.cat((src,similarity_map),dim = 2)         
        #mask_predict = self.mask_net(q[:,b:2*b,:])

        del src_ref , src_gray , similarity_map , similarity_map_ref,q,k,v
        return src,src_y,src_map

class TransformerDecoder_colorvid(nn.Module):

    def __init__(self, decoder_layers, norm=None):
        super().__init__()
        self.vc_layer0 = decoder_layers
        self.conv_start = nn.Sequential(nn.Conv2d(7, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)             # down 2 
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)           # down 4
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)           #down 8
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)                                  #up 2
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)                                 #up 4
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)                                 #up 8
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        # add self.relux_x
        self.relu1_1 = nn.ReLU(inplace=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.relu6_3 = nn.ReLU(inplace=True)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.relu7_3 = nn.ReLU(inplace=True)
        self.relu8_1_comb = nn.ReLU(inplace=True)
        self.relu8_2 = nn.ReLU(inplace=True)
        self.relu8_3 = nn.ReLU(inplace=True)
        self.relu9_1_comb = nn.ReLU(inplace=True)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.relu10_1_comb = nn.ReLU(inplace=True)
        self.relu10_2 = nn.LeakyReLU(0.2, True)

        print("replace all deconv with [nearest + conv]")
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))

        print("replace all batchnorm with instancenorm")
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_2norm = nn.InstanceNorm2d(128)

        self.fuse_trans2cnn1 = nn.Conv2d(512+384, 512, 1)
        self.fuse_cnn2trans1 = nn.Linear(512+384, 384)
        self.conv5_3norm2 = nn.InstanceNorm2d(512)
        


    def forward(self, src,src_trans,tensor_1,                     #  features bn c h w
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        b,n,_,h,w = tensor_1.shape
        tensor_1 = tensor_1.flatten(0,1) /50.                   # -- b n c hw     -- bn c h w                      
  #-----------------------intermediate-------------------#
        src = torch.cat((tensor_1,src),dim=1)
  #----------------------colornet-------------------------#   bn c h w
        conv1_1 = self.relu1_1(self.conv_start(src))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = self.relu2_1(self.conv2_1(conv1_2norm_ss))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = self.relu3_1(self.conv3_1(conv2_2norm_ss))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)  # down 8
        conv4_1 = self.relu4_1(self.conv4_1(conv3_3norm_ss))  # 256 -  512
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        #-----------------------vc layer 0 -----------------------#
        src = self.conv4_3norm(conv4_3)     # bn c h w  -- nhw b c
        src_cnn2trans = self.fuse_cnn2trans1(torch.cat([src.view(b,n,512,h//8,w//8
            ).permute(1,3,4,0,2).flatten(0,2) ,src_trans],dim=2))
        src_trans = self.vc_layer0(src_cnn2trans,temperature=1)
        src_trans2cnn = src_trans.view(n,h//8,w//8,b,384).permute(3,0,4,1,2).flatten(0,1)
        del src_trans
        #---------------------------------------------------------#
        conv5_1 = self.relu5_1(self.conv5_1(src))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        conv5_3 = self.conv5_3norm2(conv5_3)
        conv5_3 = self.fuse_trans2cnn1(torch.cat([conv5_3,src_trans2cnn],dim=1))
        conv5_3 += src 
        #-----------------------vc layer 1 -----------------------#
        src = self.conv5_3norm(conv5_3)
        #---------------------------------------------------------#
        conv6_1 = self.relu6_1(self.conv6_1(src))
        conv6_2 = self.relu6_2(self.conv6_2(conv6_1))
        conv6_3 = self.relu6_3(self.conv6_3(conv6_2))
        #-----------------------vc layer 2 -----------------------#
        src = self.conv6_3norm(conv6_3)
        #---------------------------------------------------------#
        conv7_1 = self.relu7_1(self.conv7_1(src))
        conv7_2 = self.relu7_2(self.conv7_2(conv7_1))
        conv7_3 = self.relu7_3(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = self.relu8_1_comb(conv8_1 + conv3_3_short)
        conv8_2 = self.relu8_2(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu8_3(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = self.relu9_1_comb(conv9_1 + conv2_2_short)
        conv9_2 = self.relu9_2(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = self.relu10_1_comb(conv10_1 + conv1_2_short)
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)          #bn c h w
        #del h_temp , w_temp , ref_ab_temp , feature_temp  ,atten_mask
        return torch.tanh(conv10_ab).view(b,n,2,h,w) * 128            #-128 ~ 128   


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer_encoder_parallel(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )    , Transformer_decoder_parallel(
        d_model=512,
        dropout=args.dropout,
        nhead=args.nheads,
        head_warp = args.nhead_warp,
        dim_feedforward=args.dim_feedforward ,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {activation}.")
