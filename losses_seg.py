import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Optional, Tuple, Union

class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss(reduction="mean", weight=weight)
    
    def forward(self, x, y):
        B = x.shape[0]
        l = self.loss(x.view(-1), y.view(-1))
        return l

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1).reshape(-1, 5)
        l = self.loss(x, y.reshape(-1))
        return l

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