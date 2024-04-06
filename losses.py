import torch
import torch.nn as nn
import lpips
from typing import Optional, Tuple, Union
import torchvision.transforms as T


class lsgan(nn.Module):
    def __init__(self):
        super(lsgan, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, pred, truth):
        '''
        :param gen: N, *
        :param real: N, *
        :return:
        '''
        return self.criterion(pred, truth)

class adversarial(nn.Module):
    def __init__(self):
        super(adversarial, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, truth):            
        return self.criterion(pred, truth)

class cycleloss(nn.Module):
    def __init__(self, gamma):
        super(cycleloss, self).__init__()
        self.gamma = gamma

    def forward(self, x1, x2, mask = 0):
        '''
        :param x1: M * 3 * H * W
        :param x2: F(G(x1))
        :return:
        '''
        x1 = x1.permute(1, 0, 2, 3)
        x2 = x2.permute(1, 0, 2, 3)
        C, M, H, W = x1.shape
        out = torch.mul(torch.abs(x1-x2), (1-mask))
        if isinstance(mask, int):
            N = M * H * W
        else:
            N = torch.sum(mask == 0)
        loss = torch.sum(out) / N
        return self.gamma * loss

class reconloss(nn.Module):
    def __init__(self, mu, sigma:Optional[Union[float, Tuple]] = None, kernel:int = 7):
        super(reconloss, self).__init__()
        self.mu = mu
        self.isBlur = False
        if sigma != None:
            self.filter = T.GaussianBlur(kernel, sigma)
            self.isBlur = True

    def forward(self, pred, gt, mask = 0):
        if self.isBlur:
            pred = self.filter(pred)
            gt = self.filter(gt)
        pred = pred.permute(1, 0, 2, 3)
        gt = gt.permute(1, 0, 2, 3)
        C, M, H, W = pred.shape
        out = torch.mul(torch.abs(pred - gt), (1-mask))
        if isinstance(mask, int):
            N = M * H * W
        else:
            N = torch.sum(mask == 0)
        loss = torch.sum(out) / N
        return self.mu * loss

class ploss(nn.Module):
    '''perceptual loss'''
    def __init__(self, net = "alex"):
        super(ploss, self).__init__()
        self.loss = lpips.LPIPS(net=net)
        for param in self.loss.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
    
    def forward(self, x1, x2, normalize = True):
        '''return N * 1'''
        d = self.loss(x1, x2, normalize = normalize)
        return torch.mean(d)

class FID_mdf(nn.Module):
    '''Assumption: feature space is a linear space with Euclidean metrics'''
    '''Modified FID score; use Gram matrix instead of mean & variance'''
    pass

if __name__ == "__main__":
    import torch
    
    loss_fn_alex = ploss()
    img0 = torch.ones(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1,3,64,64)
    d = loss_fn_alex(img0, img1)
    print(d)
    
