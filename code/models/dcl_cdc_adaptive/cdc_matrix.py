'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


########################   Centeral-difference (second order, with 9 parameters and a const theta for 3x3 kernel) 2D Convolution   ##############################
## | a1 a2 a3 |   | w1 w2 w3 |
## | a4 a5 a6 | * | w4 w5 w6 | --> output = \sum_{i=1}^{9}(ai * wi) - \sum_{i=1}^{9}wi * a5 --> Conv2d (k=3) - Conv2d (k=1)
## | a7 a8 a9 |   | w7 w8 w9 |
##
##   --> output =
## | a1 a2 a3 |   |  w1  w2  w3 |
## | a4 a5 a6 | * |  w4  w5  w6 |  -  | a | * | w\_sum |     (kernel_size=1x1, padding=0)
## | a7 a8 a9 |   |  w7  w8  w9 |

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)



class Conv2d_cd_pixel_difference_matrix5x5_shared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, theta=0.7):

        super(Conv2d_cd_pixel_difference_matrix5x5_shared, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.theta = theta

        self.Pad = nn.ReflectionPad2d((1,1,1,1))

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        elif x.shape[1]<2 or x.shape[2]<2:
            return out_normal
        else:
            #pdb.set_trace()

            x_pad = self.Pad(x)

            [B,C,H,W] = x.shape

            out_normal_lt = self.conv(x_pad[:,:,0:H,0:W])
            out_normal_mt = self.conv(x_pad[:,:,1:(H+1),0:W])
            out_normal_rt = self.conv(x_pad[:,:,2:(H+2),0:W])
            out_normal_lm = self.conv(x_pad[:,:,0:H,1:(W+1)])
            out_normal_mr = self.conv(x_pad[:,:,2:(H+2),1:(W+1)])
            out_normal_ld = self.conv(x_pad[:,:,0:H,2:(W+2)])
            out_normal_md = self.conv(x_pad[:,:,1:(H+1),2:(W+2)])
            out_normal_rd = self.conv(x_pad[:,:,2:(H+2),2:(W+2)])



            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff_cent = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_lt = F.conv2d(input=x_pad[:,:,0:H,0:W], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_mt = F.conv2d(input=x_pad[:,:,1:(H+1),0:W], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_rt = F.conv2d(input=x_pad[:,:,2:(H+2),0:W], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_lm = F.conv2d(input=x_pad[:,:,0:H,1:(W+1)], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_mr = F.conv2d(input=x_pad[:,:,2:(H+2),1:(W+1)], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_ld = F.conv2d(input=x_pad[:,:,0:H,2:(W+2)], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_md = F.conv2d(input=x_pad[:,:,1:(H+1),2:(W+2)], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
            out_diff_rd = F.conv2d(input=x_pad[:,:,2:(H+2),2:(W+2)], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)


            return (out_normal+out_normal_lt+out_normal_mt+out_normal_rt+out_normal_lm+out_normal_mr+out_normal_ld+out_normal_md+out_normal_rd) - self.theta * (out_diff_cent+out_diff_lt+out_diff_mt+out_diff_rt+out_diff_lm+out_diff_mr+out_diff_ld+out_diff_md+out_diff_rd)



class Conv2d_cd_pixel_difference_matrix5x5_unshared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, theta=0.7):

        super(Conv2d_cd_pixel_difference_matrix5x5_unshared, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        self.conv_lt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_mt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_rt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_lm = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_mr = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_ld = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_md = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_rd = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.theta = theta

        self.Pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        elif x.shape[1]<2 or x.shape[2]<2:
            return out_normal
        else:
            # pdb.set_trace()

            x_pad = self.Pad(x)

            [B, C, H, W] = x.shape

            out_normal_lt = self.conv_lt(x_pad[:, :, 0:H, 0:W])
            out_normal_mt = self.conv_mt(x_pad[:, :, 1:(H + 1), 0:W])
            out_normal_rt = self.conv_rt(x_pad[:, :, 2:(H + 2), 0:W])
            out_normal_lm = self.conv_lm(x_pad[:, :, 0:H, 1:(W + 1)])
            out_normal_mr = self.conv_mr(x_pad[:, :, 2:(H + 2), 1:(W + 1)])
            out_normal_ld = self.conv_ld(x_pad[:, :, 0:H, 2:(W + 2)])
            out_normal_md = self.conv_md(x_pad[:, :, 1:(H + 1), 2:(W + 2)])
            out_normal_rd = self.conv_rd(x_pad[:, :, 2:(H + 2), 2:(W + 2)])

            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            kernel_diff_lt = self.conv_lt.weight.sum(2).sum(2)
            kernel_diff_lt = kernel_diff_lt[:, :, None, None]
            kernel_diff_mt = self.conv_mt.weight.sum(2).sum(2)
            kernel_diff_mt = kernel_diff_mt[:, :, None, None]
            kernel_diff_rt = self.conv_rt.weight.sum(2).sum(2)
            kernel_diff_rt = kernel_diff_rt[:, :, None, None]
            kernel_diff_lm = self.conv_lm.weight.sum(2).sum(2)
            kernel_diff_lm = kernel_diff_lm[:, :, None, None]
            kernel_diff_mr = self.conv_mr.weight.sum(2).sum(2)
            kernel_diff_mr = kernel_diff_mr[:, :, None, None]
            kernel_diff_ld = self.conv_ld.weight.sum(2).sum(2)
            kernel_diff_ld = kernel_diff_ld[:, :, None, None]
            kernel_diff_md = self.conv_md.weight.sum(2).sum(2)
            kernel_diff_md = kernel_diff_md[:, :, None, None]
            kernel_diff_rd = self.conv_rd.weight.sum(2).sum(2)
            kernel_diff_rd = kernel_diff_rd[:, :, None, None]

            out_diff_cent = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                     padding=0)
            out_diff_lt = F.conv2d(input=x_pad[:, :, 0:H, 0:W], weight=kernel_diff_lt, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_mt = F.conv2d(input=x_pad[:, :, 1:(H + 1), 0:W], weight=kernel_diff_mt, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_rt = F.conv2d(input=x_pad[:, :, 2:(H + 2), 0:W], weight=kernel_diff_rt, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_lm = F.conv2d(input=x_pad[:, :, 0:H, 1:(W + 1)], weight=kernel_diff_lm, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_mr = F.conv2d(input=x_pad[:, :, 2:(H + 2), 1:(W + 1)], weight=kernel_diff_mr, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_ld = F.conv2d(input=x_pad[:, :, 0:H, 2:(W + 2)], weight=kernel_diff_ld, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_md = F.conv2d(input=x_pad[:, :, 1:(H + 1), 2:(W + 2)], weight=kernel_diff_md, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_rd = F.conv2d(input=x_pad[:, :, 2:(H + 2), 2:(W + 2)], weight=kernel_diff_rd, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)

            return (
                               out_normal + out_normal_lt + out_normal_mt + out_normal_rt + out_normal_lm + out_normal_mr + out_normal_ld + out_normal_md + out_normal_rd) - self.theta * (
                               out_diff_cent + out_diff_lt + out_diff_mt + out_diff_rt + out_diff_lm + out_diff_mr + out_diff_ld + out_diff_md + out_diff_rd)


class Conv2d_cd_pixel_difference_matrix4x4_unshared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False, theta=0.7):

        super(Conv2d_cd_pixel_difference_matrix4x4_unshared, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        self.conv_lt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_mt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.conv_lm = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.theta = theta

        self.Pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal

        elif x.shape[1]<2 or x.shape[2]<2:
            return out_normal
        else:
            # pdb.set_trace()

            x_pad = self.Pad(x)

            [B, C, H, W] = x.shape

            out_normal_lt = self.conv_lt(x_pad[:, :, 0:H, 0:W])
            out_normal_mt = self.conv_mt(x_pad[:, :, 1:(H + 1), 0:W])
            out_normal_lm = self.conv_lm(x_pad[:, :, 0:H, 1:(W + 1)])

            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            kernel_diff_lt = self.conv_lt.weight.sum(2).sum(2)
            kernel_diff_lt = kernel_diff_lt[:, :, None, None]
            kernel_diff_mt = self.conv_mt.weight.sum(2).sum(2)
            kernel_diff_mt = kernel_diff_mt[:, :, None, None]
            kernel_diff_lm = self.conv_lm.weight.sum(2).sum(2)
            kernel_diff_lm = kernel_diff_lm[:, :, None, None]

            out_diff_cent = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                     padding=0)
            out_diff_lt = F.conv2d(input=x_pad[:, :, 0:H, 0:W], weight=kernel_diff_lt, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_mt = F.conv2d(input=x_pad[:, :, 1:(H + 1), 0:W], weight=kernel_diff_mt, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)
            out_diff_lm = F.conv2d(input=x_pad[:, :, 0:H, 1:(W + 1)], weight=kernel_diff_lm, bias=self.conv.bias,
                                   stride=self.conv.stride, padding=0)

            return (out_normal + out_normal_lt + out_normal_mt + out_normal_lm) - self.theta * (
                        out_diff_cent + out_diff_lt + out_diff_mt + out_diff_lm)


class CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd_pixel_difference_matrix5x5_unshared, theta=0.7):
        super(CDCN, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x):  # x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)  # x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)  # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)  # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 32, 32]

        # pdb.set_trace()

        x = self.lastconv1(x_concat)  # x [128, 32, 32]
        x = self.lastconv2(x)  # x [64, 32, 32]
        x = self.lastconv3(x)  # x [1, 32, 32]

        map_x = x.squeeze(1)

        return map_x, x_Block1, x_Block2


CDCN()