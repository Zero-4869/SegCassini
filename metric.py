import scipy
import numpy as np

def IoU(pred, gt, label):
    overlap = len(np.where((pred==label)&(gt==label))[0])
    union = len(np.where((pred==label)|(gt==label))[0])
    return overlap/(union + 1e-8)

def precision(pred, gt, label):
    true = len(np.where((pred==label)&(gt==label))[0])
    total = len(np.where(pred==label)[0])
    return true/(total + 1e-8)

def recall(pred, gt, label):
    true = len(np.where((pred==label)&(gt==label))[0])
    total = len(np.where(gt==label)[0])
    return true/(total + 1e-8)

def f1(pred, gt, label):
    p = precision(pred, gt, label)
    r = recall(pred, gt, label)
    return 2*p*r/(p+r+1e-8)

def dilatedIoU(pred, gt, label, window = 3):
    assert len(pred.shape) == 2, print(pred.shape)
    pred_label = pred.copy()
    gt_label = gt.copy()
    pred_label[pred != label] = 0
    pred_label[pred == label] = 1
    gt_label[gt != label] = 0
    gt_label[gt == label] = 1

    filter = np.ones((window, window), dtype=float)
    pred_label_blur = scipy.signal.convolve2d(pred_label, filter, mode="same")
    gt_label_blur = scipy.signal.convolve2d(gt_label, filter, mode="same")
    intersection = len(np.where(((pred_label_blur > 0)&(gt_label == 1)) | ((gt_label_blur > 0)&(pred_label == 1)))[0])
    union = len(np.where((pred==label)|(gt==label))[0])
    return intersection / (union + 1e-8)