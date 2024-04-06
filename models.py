import torch
from torch import nn
import math
from torch_pconv import PConv2d
from utils import compute_msv
from lpips import LPIPS
import torch.nn.functional as F
from batchrenorm import BatchRenorm2d

# from torchvision.models import VGG19_Weights, vgg19

'''@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}'''
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class InstanceNorm2d_modified(nn.Module):
    def __init__(self, channels):
        super(InstanceNorm2d_modified, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.postnorm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        N, C, H, W = x.size()
        mean = x.view(N, C, -1).mean(dim=-1, keepdim=False)[..., None, None]
        x_norm = self.norm(x) + mean
        x_norm = self.postnorm(x_norm)
        return x_norm

class IdNorm(nn.Module):
    def __init__(self):
        super(IdNorm, self).__init__()
    
    def forward(self, x):
        return x



class AdaIN(nn.Module):
    '''
    It implements an Adaptive instance normalization layer, applying solely on spatial dimensions
    '''

    def __init__(self, alpha):
        super(AdaIN, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean_x = self.get_mean(x)
        mean_y = self.get_mean(y)
        std_x = self.get_std(x)
        std_y = self.get_std(y)
        syn = std_y * (x - mean_x) / (std_x + 1e-5) + mean_y
        out = self.alpha * syn + (1 - self.alpha) * x
        return out

    def get_mean(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: N * C * H * W
        :return: mean N * C * 1 * 1
        '''
        assert len(x.shape) == 4
        x = torch.flatten(x, start_dim=2)
        return torch.mean(x, dim=2)[...,None,None]

    def get_std(self, x: torch.Tensor):
        assert len(x.shape) == 4
        x = torch.flatten(x, start_dim=2)
        return torch.std(x, dim=2)[...,None,None]


class style_decoder(nn.Module):
    def __init__(self, use_pretrain=True):
        super(style_decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )
        if use_pretrain:
            print("loading decoder")
            decoder_state_dict = torch.load("pretrain_weights/decoder.pth")
            self.model.load_state_dict(decoder_state_dict, strict=True)

    def forward(self, x):
        return self.model(x)


class style_encoder(nn.Module):
    def __init__(self, use_pretrain=True):
        super(style_encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
        )
        if use_pretrain:
            print("loading vgg normalised")
            encoder_state_dict = torch.load("pretrain_weights/vgg_normalised.pth")
            self.model.load_state_dict(encoder_state_dict, strict=True)

    def forward(self, x):
        return self.model(x)


class style(nn.Module):
    def __init__(self, alpha, encoder_use_pretrain=True, decoder_use_pretrain=True):
        super(style, self).__init__()
        raw_encoder = style_encoder(use_pretrain=encoder_use_pretrain)
        self.encoder = nn.Sequential(*list(raw_encoder.model.children())[:31])
        self.norm = AdaIN(alpha=alpha)
        self.decoder = style_decoder(use_pretrain=decoder_use_pretrain)

    def forward(self, x, y):
        '''
        x: N * C * H * W
        y: N * C * H * W
        '''
        x = self.encoder(x)
        y = self.encoder(y)
        out = self.norm(x, y)
        out = self.decoder(out)
        return out


# class convBlock(nn.Module):
#     def __init__(self, dim, padding_type, norm_type, bias, sn = False):
#         super(convBlock, self).__init__()
#         conv_block_ = list()
#         for _ in range(2):
#             if padding_type == "reflect":
#                 conv_block_.append(nn.ReflectionPad2d((1,1,1,1)))
#             elif padding_type == "replicate":
#                 conv_block_.append(nn.ReplicationPad2d((1,1,1,1)))
#             else:
#                 conv_block_.append(nn.ZeroPad2d((1,1,1,1)))
#             if sn:
#                 conv_block_.append(SNConv2d(dim, dim, kernel_size=(3, 3), bias=bias))
#             else:
#                 conv_block_.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), bias=bias))
#             if norm_type == "instance":
#                 conv_block_.append(nn.InstanceNorm2d(dim))
#             elif norm_type == "batch":
#                 conv_block_.append(nn.BatchNorm2d(dim))
#             conv_block_.append(nn.ReLU(inplace=True))
#         self.conv_block = nn.Sequential(*conv_block_[:-1])

#     def forward(self, x):
#         return self.conv_block(x)


# class resBlock(nn.Module):
#     def __init__(self, dim, padding_type, norm_type, bias, sn=False):
#         super(resBlock, self).__init__()
#         self.conv_block = convBlock(dim, padding_type, norm_type, bias, sn)
#         self.id = nn.Identity()

#     def forward(self, x):
#         return self.id(x) + self.conv_block(x)

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
                conv_block_.append(nn.GroupNorm(num_groups=dim//4, num_channels=dim))
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

class cycle_G(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, use_mask, norm_type, upsampling=False):
        super(cycle_G, self).__init__()
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
        # if use_mask:
            # Bias = False
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

        self.netG_enc = nn.Sequential(
            # nn.ReflectionPad2d((p,p,p,p)),
            # nn.Conv2d(in_channel, hidden_dim, kernel_size=(f,f), bias=Bias),
            nn.Conv2d(in_channel, hidden_dim, kernel_size=(f,f), padding=p, padding_mode=padding_type, bias=Bias),
            get_norm(norm_type, hidden_dim),
            # nn.BatchNorm2d(hidden_dim),
            # nn.InstanceNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
            get_norm(norm_type, 2*hidden_dim),
            # nn.BatchNorm2d(2*hidden_dim),
            # nn.InstanceNorm2d(2*hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
            get_norm(norm_type, 4*hidden_dim),
            # nn.BatchNorm2d(4*hidden_dim),
            # nn.InstanceNorm2d(4*hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
            nn.ReLU(inplace=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=Bias, affine=affine, track_running_stats=track_running_stats, eps=eps),
        )
        if upsampling:
            self.netG_dec = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                nn.Conv2d(4 * hidden_dim, 2*hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
                get_norm(norm_type, 2 * hidden_dim),
                # nn.BatchNorm2d(2*hidden_dim),
                # nn.InstanceNorm2d(2*hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=Bias),
                get_norm(norm_type, hidden_dim),
                # nn.BatchNorm2d(hidden_dim),
                # nn.InstanceNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
                nn.ReLU(inplace=True),
                # nn.ReflectionPad2d((p,p,p,p)),
                # nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), bias=Bias),
                nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), padding=p, padding_mode=padding_type, bias=Bias),
                nn.Tanh()
            )
        else:
            self.netG_dec = nn.Sequential(
                nn.ConvTranspose2d(4*hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=(1, 1), padding_mode="zeros", 
                                output_padding=(1,1), bias=Bias),
                get_norm(norm_type, 2 * hidden_dim),
                # nn.BatchNorm2d(2*hidden_dim),
                # nn.InstanceNorm2d(2*hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="zeros", 
                                output_padding=(1, 1), bias=Bias),
                get_norm(norm_type, hidden_dim),
                # nn.BatchNorm2d(hidden_dim),
                # nn.InstanceNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats, eps=eps),
                nn.ReLU(inplace=True),
                # nn.ReflectionPad2d((p,p,p,p)),
                # nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), bias=Bias),
                nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), padding=p, padding_mode=padding_type, bias=Bias),
                nn.Tanh()
            )
        self.use_mask = use_mask

    def forward(self, x, mask=0):
        # if self.use_mask:
        #     x = x.permute(1, 0, 2, 3)
        #     x = torch.mul(x, (1-mask))
        #     x = x.permute(1, 0, 2, 3)
        out = self.netG_enc(x)
        out = self.netG_dec(out)
        out = (out + 1) / 2 # (-1, 1)-> (0, 1) # wrongly use 2 * (out + 1) for random align
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
        padding_type = "zeros" #"zeros"
        # norm_type = "instance_mod" 
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
        # if use_mask:
        #     Bias = False
        model_list.append(nn.Conv2d(in_channel, hidden_dim, kernel_size=(kw, kw), stride=(2,2), padding=(padw, padw), padding_mode=padding_type, bias=Bias))
        model_list.append(nn.LeakyReLU(slope, inplace=True))

        mul = 1
        for n in range(1, n_layers):  ## 5
            mul_prev = mul
            mul = min(2**n, 8)
            model_list.append(nn.Conv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw,kw), stride=(2,2),
                                        padding=(padw, padw), padding_mode=padding_type, bias=Bias))
            model_list.append(get_norm(norm_type, hidden_dim*mul))
            # model_list.append(nn.BatchNorm2d(hidden_dim*mul))
            # model_list.append(nn.InstanceNorm2d(hidden_dim*mul, affine=affine, track_running_stats=track_running_stats, eps=eps))
            model_list.append(nn.Dropout(dropout))
            model_list.append(nn.LeakyReLU(slope, inplace=True))
        mul_prev = mul
        mul = min(2**n_layers, 8)
        model_list.append(nn.Conv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw, kw), stride=(1,1),
                                    padding=(padw, padw), padding_mode=padding_type, bias=Bias))
        model_list.append(get_norm(norm_type, hidden_dim*mul))
        # model_list.append(nn.BatchNorm2d(hidden_dim*mul))
        # model_list.append(nn.InstanceNorm2d(hidden_dim*mul, affine=affine, track_running_stats=track_running_stats, eps=eps))
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
        return self.netD(x), x

class cycle_D_multiscale_vanilla(nn.Module):
    def __init__(self, in_channel, hidden_dim, n_layers, use_sigmoid, use_mask, norm_type, slope, dropout=0.0):
        '''
        Vanilla multi-scale CNN-based discriminator. shared structure
        :param n_layers: shared layers, default to be 3
        :param use_sigmoid: True if vanilla model, false if least square GAN
        '''
        super(cycle_D_multiscale_vanilla, self).__init__()
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
            model_list.append(nn.Conv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw, kw), stride=(2,2),
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

        
        self.use_mask = use_mask
        self.encode = nn.Sequential(*model_list)
        self.head_l3 = nn.Conv2d(hidden_dim*mul, 1, kernel_size=(kw, kw), stride=(1,1),
                                 padding=(padw, padw), padding_mode=padding_type, bias=Bias)

        self.head_l4 = nn.Sequential(
            nn.Conv2d(hidden_dim*mul, hidden_dim*mul*2, kernel_size=(kw, kw), stride=(2,2),
                                    padding=(padw, padw), padding_mode=padding_type, bias=Bias),
            get_norm(norm_type, hidden_dim*mul*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(hidden_dim*mul*2, 1, kernel_size=(kw, kw), stride=(1,1),
                                 padding=(padw, padw), padding_mode=padding_type, bias=Bias)
        )
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.end = nn.Sigmoid()

    def forward(self, x, mask):
        '''
        x : N * C * H * W
        mask: N * H * W: 1 stands for text detection areas
        '''
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)
        feat = self.encode(x)
        y = self.head_l3(feat).flatten(start_dim=2)
        z = self.head_l4(feat).flatten(start_dim=2)
        if self.use_sigmoid:
            y = self.end(y)
            z = self.end(z)
        return torch.cat([y, z], dim=-1), x

class cycle_D_multiscale_vanilla2(nn.Module):
    '''default input to be 70 * 70'''
    def __init__(self, in_channel, hidden_dim, n_layers, use_sigmoid, use_mask, norm_type, slope, dropout=0.0):
        '''
        Vanilla multi-scale CNN-based discriminator. Structure not shared
        :param n_layers: shared layers, default to be 3
        :param use_sigmoid: True if vanilla model, false if least square GAN
        '''
        super(cycle_D_multiscale_vanilla2, self).__init__()

        # self.D_l2 = cycle_D(in_channel, hidden_dim, n_layers, use_sigmoid, False, norm_type, slope, dropout)
        # self.D_l3 = cycle_D(in_channel, hidden_dim, n_layers, use_sigmoid, False, norm_type, slope, dropout)
        self.D_l4 = cycle_D(in_channel, hidden_dim, n_layers, use_sigmoid, False, norm_type, slope, dropout)
        self.use_mask = use_mask
        
    def forward(self, x, mask):
        '''
        x : N * C * H * W
        mask: N * H * W: 1 stands for text detection areas
        '''
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)
        
        # x_upsample = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear")
        x_downsample = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        # w = self.D_l2(x_upsample, mask)[0].flatten(start_dim=2)
        # y = self.D_l3(x, mask)[0].flatten(start_dim=2)
        z = self.D_l4(x_downsample, mask)[0].flatten(start_dim=2)
        
        return torch.cat([z], dim=-1), x

class cycle_D_multiscale_transformer(nn.Module):
    '''implementation of transGAN decoder'''
    def __init__(self, embedding_dim:int, hidden_dim:int, num_blocks:int, num_layers:int, norm_type:str, num_heads:int, 
                    initial_patch_size:int, use_sigmoid: bool, pe_type:str, use_mask):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fRGB_1 = nn.Conv2d(3, embedding_dim//4, kernel_size=initial_patch_size*2, stride=initial_patch_size, padding = initial_patch_size//2)
        self.fRGB_2 = nn.Conv2d(3, embedding_dim//4, kernel_size=initial_patch_size*2, stride=initial_patch_size*2, padding = 0)
        self.fRGB_3 = nn.Conv2d(3, embedding_dim//2, kernel_size=initial_patch_size*4, stride=initial_patch_size*4, padding = 0)
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
            elif norm_ == "rebatch":
                return BatchRenorm2d(dim)
        self.transBlocks1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim//4, dim_feedforward=hidden_dim//4, nhead=num_heads), num_layers = num_layers)
        self.transBlocks2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim//2, dim_feedforward=hidden_dim//2, nhead=num_heads), num_layers = num_layers)
        self.transBlocks3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=hidden_dim, nhead=num_heads), num_layers = num_layers)
        self.transBlocks4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=hidden_dim, nhead=num_heads), num_layers = num_layers)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.cls_token_embed = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.initial_patch_size = initial_patch_size
        self.embedding_dim = embedding_dim
        self.pe_type = pe_type
        self.use_mask = use_mask
        
        if use_sigmoid:
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, 1)
            )
    def forward(self, x, mask):
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)

        B, C, H, W = x.shape
        x1 = self.fRGB_1(x); x2 = self.fRGB_2(x); x3 = self.fRGB_3(x) # embedding
        if self.pe_type == "vanilla":
            x1 = x1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2) # B * (H/patch * W/patch) * embed
            x2 = x2.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2) # B * (H/patch * W/patch) * embed
            x3 = x3.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2) # B * (H/patch * W/patch) * embed
            x1 = x1 + self.PE_vanilla(x1); x2 = x2 + self.PE_vanilla(x2); x3 = x3 + self.PE_vanilla(x3) # B * (H/patch * W/patch) * embed
        elif self.pe_type == "vanilla2d":
            x1 = x1.permute(0, 2, 3, 1) # B * H/patch * W/patch * embed
            x2 = x2.permute(0, 2, 3, 1) # B * H/patch * W/patch * embed
            x3 = x3.permute(0, 2, 3, 1)# B * H/patch * W/patch * embed
            x1 = x1 + self.PE_vanilla_2d(x1); x2 = x2 + self.PE_vanilla_2d(x2); x3 = x3 + self.PE_vanilla_2d(x3) # B * H/patch * W/patch * embed 
            x1 = x1.flatten(start_dim=1, end_dim=2); x2 = x2.flatten(start_dim=1, end_dim=2); x3 = x3.flatten(start_dim=1, end_dim=2)      
        
        x1 = self.transBlocks1(x1) # B * (H/patch * W/patch) * embed
        x1 = x1.permute(0, 2, 1).reshape(B, -1, H//self.initial_patch_size, W//self.initial_patch_size) # B * embed * H/patch * W/patch
        x1 = self._down_sampling(x1)
        x1 = x1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        y = torch.cat([x1, x2], dim=2)
        y = self.transBlocks2(y)
        y = y.permute(0, 2, 1).reshape(B, -1, H//(self.initial_patch_size*2), W//(self.initial_patch_size*2))
        y = self._down_sampling(y)
        y = y.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        z = torch.cat([y, x3], dim=2)
        z = self.transBlocks3(z) # B * (H/patch * W/patch) * embed

        cls_token = torch.zeros(z.shape[0], device=z.device, dtype=torch.long) # B * embed
        cls_token = self.cls_token_embed(cls_token)[:, None, :] # B * 1 * embed
        z = torch.cat([z, cls_token], dim=1)
        z = self.transBlocks4(z)
        out = z[:, -1]
        out = self.head(out)
        return out, x

    def PE_vanilla(self, x):
        '''1d sinusoidal PE
            x: B * N * C
        '''
        B, N, C = x.shape
        assert C%2 == 0, print("cannot do sinusoidal pe")
        pe = torch.empty((N, C), device=x.device)
        pos_x = torch.arange(0., N, device=x.device)
        pos_y = torch.arange(0., C//2, device=x.device)
        w = 1/torch.pow(10000*torch.ones((int)(C/2), dtype=x.dtype, device=x.device), 2*pos_y/C)
        pos = pos_x[None,:].T @ w[None,:] # N * d/2
        sin_pos = torch.sin(pos)
        cos_pos = torch.cos(pos)
        pe[:, 0::2] = sin_pos
        pe[:, 1::2] = cos_pos
        return pe.repeat(B, 1, 1)
    
    def PE_vanilla_2d(self, x):
        '''
        2d sinusoidal PE
        x: B * H * W * C
        deprecated
        '''
        B, H, W, C = x.shape
        assert C%4 == 0, print("cannot do 2d sinusoidal pe")
        pe = torch.empty((H, W, C), device=x.device)
        C = int(C / 2)
        pos = torch.arange(0., C//2, device=x.device)
        w = 1/torch.pow(10000*torch.ones((int)(C/2), dtype=x.dtype), 2*pos/C)
        pos_w = torch.arange(0., W).unsqueeze(1) # W * 1
        pos_h = torch.arange(0., H).unsqueeze(1) # H * 1

        pos1 = pos_w @ w[None] # W * C/4
        pos2 = pos_h @ w[None] # H * C/4
        pe[:, :, 0:C:2] = torch.sin(pos1).repeat(H, 1, 1)
        pe[:, :, 1:C:2] = torch.cos(pos1).repeat(H, 1, 1)
        pe[:, :, C + 0:C:2] = torch.sin(pos2).transpose(0, 1).repeat(1, W, 1)
        pe[:, :, C + 1:C:2] = torch.cos(pos2).transpose(0, 1).repeat(1, W, 1)
        return pe.repeat(B, 1, 1, 1).to(x.device)     

    def _down_sampling(self, x):
        '''x: N * C * H * W'''
        return self.pooling(x)
    
    # def window_partition(self, x, window_size):
    #     B, C, H, W = x.shape
    #     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    #     return windows
    
    # def window_concat(self, x, H, W):
    #     '''
    #     x: (num_windows * B, window_size, window_size, C)
    #     '''
    #     window_size = x.shape[1]
    #     num_window = H * W // (window_size ** 2)
    #     B = windows.shape[0] // num_window
    #     x = x.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    #     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    #     return x

class SNConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 ite = 1, padding_mode = "zeros", dilation = 1, groups = 1, bias = True):
        super(SNConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              padding_mode=padding_mode, dilation=dilation, groups=groups, bias=bias)
        self.u = nn.Parameter(torch.randn((1, out_channels), dtype = torch.float32), requires_grad=False)
        self.ite = ite

    def normalize_weight(self):
        W_mat = self.weight.reshape(self.weight.shape[0], -1)
        if self.u.device != W_mat.device:
            self.u = self.u.to(W_mat.device)
        sigma, _u = compute_msv(W_mat, self.u, self.ite)
        if self.training:
            self.u.data = _u
        with torch.no_grad():
            self.weight /= sigma

    def forward(self, x):
        return self._conv_forward(x, self.weight, self.bias)

class cycle_G_SN(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, use_mask, upsampling=False):
        super(cycle_G_SN, self).__init__()
        f = 7
        p = (f-1)//2
        padding_type = "reflect"
        norm_type = "instance"
        padding_type2 = "zeros"

        Bias = True
        # if use_mask:
            # Bias = False

        self.netG_enc = nn.Sequential(
            nn.ReflectionPad2d((p,p,p,p)),
            SNConv2d(in_channel, hidden_dim, kernel_size=(f,f), bias=Bias),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            SNConv2d(hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=(1, 1), padding_mode = padding_type2, bias=True),
            nn.InstanceNorm2d(2*hidden_dim),
            nn.ReLU(inplace=True),
            SNConv2d(2*hidden_dim, 4*hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode = padding_type2, bias=True),
            nn.InstanceNorm2d(4*hidden_dim),
            nn.ReLU(inplace=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
            resBlock(4 * hidden_dim, padding_type, norm_type, bias=True, sn=True),
        )
        if upsampling:
            self.netG_dec = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                SNConv2d(4 * hidden_dim, 2*hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode = padding_type2, bias=True),
                nn.InstanceNorm2d(2*hidden_dim),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                SNConv2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode = padding_type2, bias=True),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d((p,p,p,p)),
                SNConv2d(hidden_dim, out_channel, kernel_size=(f,f), bias=True),
                nn.Tanh()
            )
        else:
            self.netG_dec = nn.Sequential(
                nn.ConvTranspose2d(4*hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=(1, 1), padding_mode = "zeros", 
                                output_padding=(1,1), bias=True),
                nn.InstanceNorm2d(2*hidden_dim),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode = "zeros", 
                                output_padding=(1, 1), bias=True),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d((p,p,p,p)),
                SNConv2d(hidden_dim, out_channel, kernel_size=(f,f), bias=True),
                nn.Tanh()
            )
        self.use_mask = use_mask

    def forward(self, x, mask):
        # if self.use_mask:
        #     x = x.permute(1, 0, 2, 3)
        #     x = torch.mul(x, (1-mask))
        #     x = x.permute(1, 0, 2, 3)
        out = self.netG_enc(x)
        out = self.netG_dec(out)
        out = (out + 1) / 2 #(-1, 1) -> (0, 1)
        return out

class cycle_D_SN(nn.Module):
    '''default input to be 70 * 70'''
    def __init__(self, in_channel, hidden_dim, n_layers, use_sigmoid, use_mask, dropout=0.0):
        '''
        :param n_layers: 3 for basic D
        :param use_sigmoid: True if vanilla model, false if least square GAN
        :param dropout:
        '''
        super(cycle_D_SN, self).__init__()
        model_list = list()
        kw = 4
        padw = math.ceil((kw-1)/2)
        padding_type = "zeros"

        Bias = True
        if use_mask:
            Bias = False
        model_list.append(SNConv2d(in_channel, hidden_dim, kernel_size=(kw, kw), stride=(2,2), padding=(padw, padw), padding_mode = padding_type, bias=Bias))
        model_list.append(nn.LeakyReLU(0.2, inplace=True))

        mul = 1
        for n in range(1, n_layers):  ## 5
            mul_prev = mul
            mul = min(2**n, 8)
            model_list.append(SNConv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw,kw), stride=(2,2),
                                        padding=(padw, padw), padding_mode = padding_type, bias=Bias))
            model_list.append(nn.InstanceNorm2d(hidden_dim*mul))
            model_list.append(nn.Dropout(dropout))
            model_list.append(nn.LeakyReLU(0.2, inplace=True))
        mul_prev = mul
        mul = min(2**n_layers, 8)
        model_list.append(SNConv2d(hidden_dim*mul_prev, hidden_dim*mul, kernel_size=(kw, kw), stride=(1,1),
                                    padding=(padw, padw), padding_mode = padding_type, bias=Bias))
        model_list.append(nn.InstanceNorm2d(hidden_dim*mul))
        model_list.append(nn.LeakyReLU(0.2, inplace=True))
        model_list.append(SNConv2d(hidden_dim*mul, 1, kernel_size=(kw, kw), stride=(1,1),
                                    padding=(padw, padw), padding_mode = padding_type, bias=Bias))
        if use_sigmoid:
            model_list.append(nn.Sigmoid())
        self.use_mask = use_mask
        self.netD = nn.Sequential(*model_list)

    def normalize_weight(self):
        for m in self.netD.modules():
            if isinstance(m, SNConv2d):
                m.normalize_weight()

    def forward(self, x, mask):
        '''
        x : N * C * H * W
        mask: N * H * W: 1 stands for text detection areas
        '''
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)
        return self.netD(x)

class star_G(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, c_dim, repeat_num, use_mask, upsampling=False):
        super(star_G, self).__init__()
        f = 7
        p = (f - 1) // 2
        padding_type = "zero"
        norm_type = "instance"
        self.netG_enc = nn.Sequential(
            nn.Conv2d(in_channel + c_dim, hidden_dim, kernel_size=(f, f), padding=p, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(2 * hidden_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(4 * hidden_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *[resBlock(4 * hidden_dim, padding_type, norm_type, bias=False, affine=True, track_running_stats=True) for _ in range(repeat_num)],
        )

        if upsampling:
            self.netG_dec = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                nn.Conv2d(4 * hidden_dim, 2*hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=False),
                nn.InstanceNorm2d(2*hidden_dim, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2), mode="nearest"),
                nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=padding_type2, bias=False),
                nn.InstanceNorm2d(hidden_dim, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), padding=p, bias=False),
                nn.Tanh()
            )
        else:
            self.netG_dec = nn.Sequential(
                nn.ConvTranspose2d(4*hidden_dim, 2*hidden_dim, kernel_size=(3,3), stride=(2,2), padding=(1, 1), padding_mode="zeros", 
                                output_padding=(1,1), bias=False),
                nn.InstanceNorm2d(2*hidden_dim, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="zeros", 
                                output_padding=(1, 1), bias=False),
                nn.InstanceNorm2d(hidden_dim, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channel, kernel_size=(f,f), padding=p, bias=False),
                nn.Tanh()
            )
        self.use_mask = use_mask
                  

    def forward(self, x, c, mask):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        out = self.netG_enc(x)
        out = self.netG_dec(x)
        out = (out + 1)/2
        return out

class star_D(nn.Module):
    '''default input to be 70 * 70'''

    def __init__(self, image_size, in_channel, c_dim, hidden_dim, n_layers, use_mask):
        '''
        :param n_layers: 6 for basic D
        :param use_sigmoid: True if vanilla model, false if least square GAN
        :param dropout:
        '''
        super(star_D, self).__init__()
        model_list = list()
        kw = 4
        padw = math.ceil((kw - 1) / 2)
        model_list.append(nn.Conv2d(in_channel+c_dim, hidden_dim, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw)))
        model_list.append(nn.LeakyReLU(0.2, inplace=True))

        mul = 1
        for n in range(1, n_layers):
            mul_prev = mul
            mul = 2 ** n
            model_list.append(nn.Conv2d(hidden_dim * mul_prev, hidden_dim * mul, kernel_size=(kw, kw), stride=(2, 2),
                                        padding=(padw, padw)))
            model_list.append(nn.LeakyReLU(0.2, inplace=True))
        kernel_size = int(image_size/(2**n_layers))
        self.netD = nn.Sequential(*model_list)
        self.conv1 = nn.Conv2d(hidden_dim * mul, 1, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim * mul, c_dim, kernel_size=(kernel_size, kernel_size), bias=False)
        self.use_mask = use_mask

    def forward(self, x, mask):
        if self.use_mask:
            x = x.permute(1, 0, 2, 3)
            x = torch.mul(x, (1-mask))
            x = x.permute(1, 0, 2, 3)
        h = self.netD(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class LPIPS2(LPIPS):
    def __init__(self):
        super(LPIPS2, self).__init__()

    def forward(self, in0, retPerLayer = False, normalize = False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input = self.scaling_layer(in0) if self.version=='0.1' else (in0, in1)
        outs0 = self.net.forward(in0_input)
        feats0 = {}

        for kk in range(self.L):
            feats0[kk] = lpips.normalize_tensor(outs0[kk])

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](feats0[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](feats0[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(feats0[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(feats0[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]
        return res

class Affine(nn.Module):
    '''limited to -3 to 3'''
    def __init__(self):
        super(Affine, self).__init__()
        self.model = nn.Embedding(181, 2)
    def forward(self, x):
        '''x: city id, from 0 to 180, of type long'''
        out = self.model(x)
        out_x = out[:, 0]
        out_y = out[:, 1]
        return out_x, out_y

if __name__ == "__main__":
    encoder_state_dict = torch.load("pretrain_weights/vgg_normalised.pth")
    for name, param in encoder_state_dict.items():
        print(name, torch.min(param), torch.max(param))
    print("========================================================")
    decoder_state_dict = torch.load("pretrain_weights/decoder.pth")
    for name, param in decoder_state_dict.items():
        print(name, torch.min(param), torch.max(param))

    models = style_encoder()
    print(len(list(models.model.children())))
    print(list(models.model.children())[31])