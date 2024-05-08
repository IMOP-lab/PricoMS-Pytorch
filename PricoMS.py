# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
import math

import torch
from torchvision import models
import torch.nn as nn
from Model.mae2 import mae_vit_base_patch16 
from Model.pvt2 import  pvt_tiny
from torch.nn import functional as F
import torchsummary
from torch.nn import init
import numpy as np
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torchvision.ops import DeformConv2d
from torch.autograd import Variable
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
torch_ver = torch.__version__[:3]

class DCnv2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super(DCnv2, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=k, stride=1, padding=p, groups=g, bias=False)
        deformable_groups = 1
        offset_channels = 18
        mask_channels = 9
        self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=k, stride=s, padding=p)
        self.conv2_mask = nn.Conv2d(c2, deformable_groups*mask_channels, kernel_size=k, stride=s, padding=p)
        # init_mask = torch.Tensor(np.zeros([mask_channels, 3, 3, 3])) + np.array([0.5])
        # self.conv2_mask.weight = torch.nn.Parameter(init_mask)
        self.conv2 = DeformConv2d(c2, c2, kernel_size=k, stride=s, padding=1, bias=True)
 
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        offset = self.conv2_offset(x)
        mask = torch.sigmoid(self.conv2_mask(x))
        x = self.act2(self.bn2(self.conv2(x, offset=offset, mask=mask)))
 
        return x

class AMFModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(AMFModule, self).__init__()
        # 定义各层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ) 
        self.in_channels=in_channels
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.deform_conv = nn.Sequential(
                           DCnv2(in_channels,out_channels),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(True)
                           )
        self.softmax=Softmax(dim=-1)

      
        # 其他必要的层和操作
    
    def forward(self, x1, x2):
     

        m, C, height, width = x1.size()
        a = self.deform_conv(x2)
        avg_pooled = self.avg_pool(a)
        max_pooled = self.max_pool(a)
        attention1 =self.softmax(torch.matmul(avg_pooled.view(m, C, 1),  max_pooled.view(m, C, 1).transpose(1, 2)))
        out1=torch.bmm(attention1, x1.view( m,C, -1))
        out1= out1.view(m, C, height, width)

        b=self.conv1(x1)
        s_h, s_w = self.pool_h(b), self.pool_w(b)  # .permute(0, 1, 3, 2)   
        attention2 =self.softmax(torch.matmul(s_h, s_w))
        out2=x2*attention2 

        x_fused =x1+x2+self.gamma*out1+out2
        
        return x_fused

class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        atten = self.sigmoid(avg_out + max_out )    # 计算得到的注意力
        return x * atten         # 将输入矩阵乘以对应的注意力
class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv1(atten)       # 计算得到的注意力
        atten = self.sigmoid(atten)      # 将输入矩阵乘以对应的注意力
        return x * atten       # 将输入矩阵乘以对应的注意力


class dprior_module(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
         super(dprior_module, self).__init__()
         self.channel_mapping = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels, 3,1,1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
         self.direc_reencode = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1)
                )
         self.scale=scale_factor
         self.gap = GlobalAvgPool2D()
         
    def forward(self, x):
        E_pre= self.channel_mapping(x)
        E_pre=F.interpolate(E_pre,scale_factor=self.scale,mode='bilinear',align_corners=True)
        E_pre=self.direc_reencode(E_pre)
        d_prior=self.gap(E_pre)
        return   d_prior


GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)


class gap(nn.Module):
    def __init__(self):
        super(gap, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool

class up_conv_r2u(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_r2u,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class HCS_module(nn.Module):
    def __init__(self, in_channels, kernel_size, out_planes):
         super(HCS_module, self).__init__()
         self.reencoder = nn.Sequential(
                        nn.Conv2d(out_planes, out_planes*8, 1),
                        nn.ReLU(True),
                        nn.Conv2d(out_planes*8, in_channels, 1))
         self.cam=CAM(in_channels)
         self.sam=SAM(kernel_size)
         self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, in_channels, 1))
         
    def forward(self, d_prior,x):
        d_prior= self.reencoder(d_prior)
        d = self.cam(x)
        d=self.sam(d)
        d=d*F.sigmoid(d_prior)
        d=self.final_conv (d)
        d=d+x
        return  d
    

class conv_block_r2u(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_r2u,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class PricoMS(nn.Module):
    def __init__(self,num_class=3):
        super(PricoMS,self).__init__()
        
        out_planes = num_class*8
        self.backbone1=mae_vit_base_patch16 (pretrained=True)
        self.backbone2=pvt_tiny (pretrained=True)
        
        self.dprior_module=dprior_module(512,out_planes,32)
        self.hcs_module=HCS_module(512,3,out_planes)
        self.cse4 = CSE(256,256,256)
        self.cse3 = CSE(128,128,128)
        self.cse2 = CSE(64,64,64)
        self.cse1 = CSE(32,32,32)
        self.gap = GlobalAvgPool2D()
        self.conv1 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.conv0 = nn.Conv2d(3, 24, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.amf=AMFModule(32,32)
        self.Up4 = up_conv_r2u(ch_in=512,ch_out=256)
        self.up_conv_r2u4 = conv_block_r2u(ch_in=512, ch_out=256)

        self.Up3 = up_conv_r2u(ch_in=256,ch_out=128)
        self.up_conv_r2u3 = conv_block_r2u(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv_r2u(ch_in=128,ch_out=64)
        self.up_conv_r2u2 = conv_block_r2u(ch_in=128, ch_out=64)
        
        self.Up1 = up_conv_r2u(ch_in=64,ch_out=32)
        self.up_conv_r2u1 = conv_block_r2u(ch_in=64, ch_out=32)

        self.final_decoder=Mul_fusion(in_channels=[32,64,128,256],out_channels=32,in_feat_output_strides=(4, 8, 16, 32),out_feat_output_stride=4,norm_fn=nn.BatchNorm2d,num_groups_gn=None)
       

        
        self.cls_pred_conv_2 = nn.Conv2d(32, out_planes , 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=2)
       

# 添加softmax层
        self.softmax = nn.Softmax(dim=1)  
    def forward(self, x):
        
        x = self.backbone1.forward_encoder(x)
        x1=self.conv0(x)
        x,c1 = self.backbone2.forward_features(x)
        c2 = x[0]#1/4   64
        c3 = x[1]#1/8   128
        c4 = x[2]#1/16   256
        c4=self.conv1(c4)
        c5 = x[3]#1/32   512
        
        d_prior = self.dprior_module(c5)
        c5=self.hcs_module( d_prior,c5)
        
        #多尺度融合
        d4 = self.Up4(c5)
        d4 = self.cse4(self.gap(d4),d4)
        d4 = torch.cat((c4,d4),dim=1)
        d4 = self.up_conv_r2u4(d4)
        
        d3 = self.Up3(d4)
        d3 = self.cse3(self.gap(d3),d3)
        d3 = torch.cat((c3,d3),dim=1)
        d3 = self.up_conv_r2u3(d3)

        d2 = self.Up2(d3)
        d2 = self.cse2(self.gap(d2),d2)
        d2 = torch.cat((c2,d2),dim=1)
        d2 = self.up_conv_r2u2(d2)

        d1 = self.Up1(d2)
        d1 = self.cse1(self.gap(d1),d1)
        d1 = torch.cat((c1,d1),dim=1)
        d1 = self.up_conv_r2u1(d1)

        feat_list = [c1,c2,c3,c4,c5]
        final_feat = self.final_decoder(feat_list)
        final_feat=self.amf(d1,final_feat)

        cls_pred = self.cls_pred_conv_2(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        cls_pred=cls_pred+x1
        cls_pred = self.softmax(cls_pred ) 
        return cls_pred
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
#        return F.logsigmoid(main_out,dim=1)
class CSE(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,
                 scale_aware_proj=False):
        super(CSE, self).__init__()
        self.in_channels=in_channels
        self.scale_aware_proj = scale_aware_proj
        t=int(abs((np.log2(self.in_channels)+1)/2))
        k=t if t%2 else t+1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.content_encoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ) 
        self.feature_reencoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        self.normalizer = nn.Sigmoid()
    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)
          
        scene_feat = self.scene_encoder(scene_feature)
     
        scene_feat=self.conv(scene_feat.squeeze(-1).transpose(-1, -2))
        scene_feat = scene_feat.transpose(-1, -2).unsqueeze(-1)
       
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))
        re_feats = self.feature_reencoders(features) 
        refined_feats = relations * re_feats 

        return refined_feats
    
    
class Mul_fusion(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(Mul_fusion, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        dec_level = 0
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels[dec_level] if idx ==0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))
            dec_level+=1

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


