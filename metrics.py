import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import xlsxwriter
import surface_distance as surfdist
from scipy.spatial.distance import directed_hausdorff

def cal_base(y_true, y_pred, epsilon=1e-6):

    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)


    TP = (y_pred_flat * y_true_flat).sum()
    FP = (y_pred_flat * (1 - y_true_flat)).sum()
    FN = ((1 - y_pred_flat) *y_true_flat).sum()
    TN = ((1 - y_pred_flat) * (1 - y_true_flat)).sum()

    return TP, TN, FP, FN, epsilon


def PA(y_true, y_pred):
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + epsilon)
    return ACC.item()


def IoU(y_true, y_pred):
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    iou = TP / (TP + FP + FN + epsilon)
    return iou.item()


def Recall(y_true, y_pred):
    """ recall or sensitivity """
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    SE = TP / (TP + FN + epsilon)
    return SE.item()


def Precision(y_true, y_pred):
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    PC = TP / (TP + FP + epsilon)
    return PC.item()


def Specificity(y_true, y_pred):
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + epsilon)
    return SP.item()


def F1_socre(y_true, y_pred):
    SE = Recall(y_true, y_pred)
    PC = Precision(y_true, y_pred)
    epsilon = 1e-7
    F1 = 2 * SE * PC / (SE + PC + epsilon)
    return F1


def Dice(y_true, y_pred):
    TP, TN, FP, FN, epsilon = cal_base(y_true, y_pred)
    DC = (2 * TP) / (2 * TP + FP + FN + epsilon)
    return DC.item()
