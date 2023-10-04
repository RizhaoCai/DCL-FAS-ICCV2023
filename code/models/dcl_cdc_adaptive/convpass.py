import torch
from torch import nn
import timm
import math
from torch.nn import functional as F

import logging

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x,dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, adaptive_type='learnable'):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.adaptive_type = adaptive_type
        if self.adaptive_type == 'learnable':
            self.theta = torch.nn.parameter.Parameter(torch.tensor(0.0,requires_grad=True))  # init: 0 - after sigmoid - 0.5

        elif self.adaptive_type == 'sample':
            self.z = torch.nn.parameter.Parameter(torch.rand(1, requires_grad=True))  # init: 0 - after sigmoid - 0.5

        elif self.adaptive_type == 'sample2':
            self.mu = torch.nn.parameter.Parameter(torch.tensor(0.0, requires_grad=True))  # init: 0 - after sigmoid - 0.5
            self.sigma = torch.nn.parameter.Parameter(
                torch.tensor(1.0, requires_grad=True))  # init: 0 - after sigmoid - 0.5

        elif self.adaptive_type == 'layer_attn':
            #self.down_scale = torch.nn.Linear(out_channels, out_channels//2)

            self.channel_pooling = torch.nn.Conv2d(out_channels, 1, kernel_size=(1,1), stride=1, padding=0)
            self.attn_layer1 = torch.nn.Linear(2, 4)
            self.attn_layer2 = torch.nn.Linear(4, 2)

        elif self.adaptive_type == 'spatial_attn':
            self.spatial_attention_layer = SpatialAttention()

        else:
            raise NotImplementedError

    def reparameterization_sampling(self, distribution_params):

        mean = 0
        if self.training:
            epison = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
            sigma = mean + distribution_params * epison
        else:
            sigma = mean + distribution_params

        return sigma

    def reparameterization_sampling_2(self, mu, sigma):

        if self.training:
            epison = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
            Theta = mu + sigma * epison
        else:
            Theta = mu

        return Theta

    def calculate_layer_attn_for_theta(self, conv_feat, cdc_feat):

        # conv_feat (b, c_out, h, w) ->  (b, c_out)

        conv_feat_d = torch.nn.functional.adaptive_avg_pool2d(conv_feat, (1,1)) # (b, 1)
        conv_feat_d = self.channel_pooling(conv_feat_d)
        cdc_feat_d = torch.nn.functional.adaptive_avg_pool2d(cdc_feat, (1,1)) # (b, 1)
        cdc_feat_d = self.channel_pooling(cdc_feat_d)
        cat_feat = torch.cat([conv_feat_d, cdc_feat_d], 1).squeeze(dim=3).squeeze(dim=2)

        atten_logit = self.attn_layer2(F.relu(self.attn_layer1(cat_feat)))
        theta_attention = torch.softmax(atten_logit,1)[:,1].view(-1,1,1,1)

        return theta_attention

    def calculate_spatial_attn_for_theta(self, conv_feat, cdc_feat):

        pass

    def forward(self, x):
        out_normal = self.conv(x)
        theta=0.5
        if self.adaptive_type == 'learnable':
            # theta = (1+torch.tanh(self.theta)) # constrain to 0~1
            theta = torch.sigmoid(self.theta)  # constrain to 0~1
        elif self.adaptive_type == 'sample':
            Z = self.reparameterization_sampling(self.z)
            theta = torch.sigmoid(Z).cuda()  # constrain to 0~1
        elif self.adaptive_type == 'sample2':
            Theta = self.reparameterization_sampling_2(self.mu, self.sigma )
            theta = torch.sigmoid(Theta).cuda()  # constrain to 0~1

        if math.fabs(theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            if self.adaptive_type == 'learnable' or self.adaptive_type == 'sample' or self.adaptive_type == 'sample2' :
                # theta = (1+torch.tanh(self.theta)) # constrain to 0~1
                return out_normal - theta * out_diff

            elif self.adaptive_type == 'layer_attn':
                layer_attn_theta = self.calculate_layer_attn_for_theta(out_normal, out_diff)
                #print(layer_attn_theta.size())
                #print(out_diff.size())
                return out_normal - layer_attn_theta * out_diff

            elif self.adaptive_type == 'spatial_attn':
                spatial_attn_theta = self.spatial_attention_layer(out_normal)
                output = out_normal - (1-spatial_attn_theta)*out_diff
                return output


            else:
                raise NotImplementedError

def forward_block(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.mlp(self.norm2(x))) + self.drop_path2(self.adapter_mlp(self.norm2(x))) * self.s
    return x

def forward_block_attn(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x))) + self.drop_path1(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False, conv_type='cdc', cdc_theta=0.7, adaptive_type='learnable'):
        super().__init__()

        logging.info('cdc_adaptive type={}'.format(adaptive_type))
        self.adapter_conv = Conv2d_cd(dim, dim, 3, 1, 1, adaptive_type=adaptive_type)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.conv.weight)
            self.adapter_conv.conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        #nn.init.zeros_(self.adapter_conv.conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)


        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim


    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)

        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


def set_Convpass(model, method, dim=8, s=1, xavier_init=False, conv_type='cdc', cdc_theta=0.7, adaptive_type='learnable'):

    if method == 'convpass':

        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta, adaptive_type=adaptive_type)
                _.adapter_mlp = Convpass(dim, xavier_init,conv_type=conv_type, cdc_theta=cdc_theta, adaptive_type=adaptive_type)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type, cdc_theta=cdc_theta, adaptive_type=adaptive_type)
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init, conv_type=conv_type)
                _.s = s
                bound_method = forward_block_attn.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init, conv_type=conv_type)
