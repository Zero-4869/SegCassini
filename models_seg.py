import torch
from torch import nn
import math
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, bias, sn, affine, track_running_stats, eps):
        super(convBlock, self).__init__()
        conv_block_ = list()
        for _ in range(2):
            if padding_type == "reflect":
                conv_block_.append(nn.ReflectionPad2d((1,1,1,1)))
            elif padding_type == "replicate":
                conv_block_.append(nn.ReplicationPad2d((1,1,1,1)))
            else:
                conv_block_.append(nn.ZeroPad2d((1,1,1,1)))
            if sn:
                conv_block_.append(SNConv2d(dim, dim, kernel_size=(3, 3), bias=bias))
            else:
                conv_block_.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), bias=bias))
            if norm_type == "instance":
                conv_block_.append(nn.InstanceNorm2d(dim, affine=affine, track_running_stats=track_running_stats, eps=eps))
            elif norm_type == "batch":
                conv_block_.append(nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats))
            elif norm_type == "layer":
                conv_block_.append(LayerNorm2d(dim))
            elif norm_type == "group":
                conv_block_.append(nn.GroupNorm(num_groups=dim//2, num_channels=dim))
            else:
                conv_block_.append(IdNorm())
            conv_block_.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*conv_block_[:-1])

    def forward(self, x):
        return self.conv_block(x)


class resBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, bias=True, sn=False, affine=False, track_running_stats=False, eps=1e-5):
        super(resBlock, self).__init__()
        self.conv_block = convBlock(dim, padding_type, norm_type, bias, sn, affine, track_running_stats, eps=eps)
        self.id = nn.Identity()

    def forward(self, x):
        return self.id(x) + self.conv_block(x)


class unet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, norm_type):
        super(unet, self).__init__()
        f = 7
        p = (f-1)//2
        padding_type = "reflect"
        # norm_type = "batch"
        padding_type2 = "zeros" #"zeros" reflect slows down and destroy the performance
        affine = True
        track_running_stats = True
        if norm_type == "batch":
            affine = True
            track_running_stats = True
        elif norm_type == "instance":
            affine = False
            track_running_stats = False           
        Bias = True
        eps = 1e-5

        def get_norm(norm_, dim):
            if norm_ == "batch":
                return nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats)
            elif norm_ == "instance":
                return nn.InstanceNorm2d(dim, affine=affine, track_running_stats=track_running_stats, eps=eps)
            elif norm_ == "layer":
                return LayerNorm2d(dim)
            elif norm_ == "group":
                return nn.GroupNorm(num_groups=dim//2, num_channels=dim)
            elif norm_ == "no_norm":
                return IdNorm()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=(f,f), padding=p, padding_mode=padding_type2),
            get_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resnet1 = resBlock(hidden_dim, padding_type2, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps)

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, 2*hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resnet2 = resBlock(2*hidden_dim, padding_type2, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, 4*hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resnet3 = resBlock(4*hidden_dim, padding_type2, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps)

        self.conv4 = nn.Sequential(
            nn.Conv2d(4*hidden_dim, 8*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, 8*hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resnet4 = resBlock(8*hidden_dim, padding_type2, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps)

        self.transconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(8 * hidden_dim, 4*hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
            get_norm(norm_type, 4 * hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*hidden_dim, 4*hidden_dim, kernel_size=(3,3), stride=(1,1), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, 4*hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.transconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(4 * hidden_dim, 2*hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
            get_norm(norm_type, 2 * hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(4*hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(1,1), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, 2*hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.transconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
            get_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=(3,3), stride=(1,1), padding=1, padding_mode=padding_type2),
            get_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), padding=p, padding_mode=padding_type, bias=Bias)
        # self.F = nn.Sigmoid()

    def forward(self, x):
        y1 = self.resnet1(self.conv1(x))
        y2 = self.resnet2(self.conv2(y1))
        y3 = self.resnet3(self.conv3(y2))
        y4 = self.resnet4(self.conv4(y3))

        z3 = self.transconv3(y4)
        z3 = self.conv5(torch.cat([y3, z3], dim=1))
        z2 = self.transconv2(z3)
        z2 = self.conv6(torch.cat([y2, z2], dim=1))
        z1 = self.transconv1(z2)
        z1 = self.conv7(torch.cat([y1, z1], dim=1))
        out = self.conv8(z1)
        # out = self.F(self.conv8(z1))
        return out

class cycle_D(nn.Module):
    '''default input to be 70 * 70'''
    def __init__(self, in_channel, hidden_dim, n_layers, use_sigmoid, use_mask, norm_type, slope, dropout=0.0):
        '''
        :param n_layers: 3 for basic D
        :param use_sigmoid: True if vanilla model, false if least square GAN
        :param dropout:
        '''
        super(cycle_D, self).__init__()
        model_list = list()
        kw = 4
        padw = math.ceil((kw-1)/2)
        padding_type = "zeros"  
        if norm_type == "batch":
            affine = True
            track_running_stats = True
        else:
            affine = False
            track_running_stats = False
        eps = 1e-5
        def get_norm(norm_, dim):
            if norm_ == "batch":
                return nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats)
            elif norm_ == "instance":
                return nn.InstanceNorm2d(dim, affine=affine, track_running_stats=track_running_stats, eps=eps)
            elif norm_ == "layer":
                return LayerNorm2d(dim)
            elif norm_ == "instance_mod":
                return InstanceNorm2d_modified(dim)
            elif norm_ == "group":
                return nn.GroupNorm(num_groups=dim//2, num_channels=dim)
            elif norm_ == "no_norm":
                return IdNorm()
            elif norm_ == "rebatch":
                return BatchRenorm2d(dim)

        Bias = True
        model_list.append(nn.Conv2d(in_channel, hidden_dim, kernel_size=(kw, kw), stride=(2,2), padding=(padw, padw), padding_mode=padding_type, bias=Bias))
        model_list.append(nn.LeakyReLU(slope, inplace=True))

        mul = 1
        for n in range(1, n_layers): 
            mul_prev = mul
            mul = min(2**n, 8)
            model_list.append(nn.Conv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw,kw), stride=(2,2),
                                        padding=(padw, padw), padding_mode=padding_type, bias=Bias))
            model_list.append(get_norm(norm_type, hidden_dim*mul))
            model_list.append(nn.Dropout(dropout))
            model_list.append(nn.LeakyReLU(slope, inplace=True))
        mul_prev = mul
        mul = min(2**n_layers, 8)
        model_list.append(nn.Conv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw, kw), stride=(1,1),
                                    padding=(padw, padw), padding_mode=padding_type, bias=Bias))
        model_list.append(get_norm(norm_type, hidden_dim*mul))
        model_list.append(nn.LeakyReLU(slope, inplace=True))
        model_list.append(nn.Conv2d(hidden_dim*mul, 1, kernel_size=(kw, kw), stride=(1,1),
                                    padding=(padw, padw), padding_mode=padding_type, bias=Bias))
        if use_sigmoid:
            model_list.append(nn.Sigmoid())
        self.use_mask = use_mask
        self.netD = nn.Sequential(*model_list)

    def forward(self, x, mask = 0):
        '''
        x : N * C * H * W
        mask: N * H * W: 1 stands for text detection areas
        '''
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)
        return self.netD(x)