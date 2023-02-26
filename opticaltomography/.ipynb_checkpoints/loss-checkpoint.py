import torch
import numpy as np
import torch.nn as nn

def mse_loss(predict, obj, device='cpu'):
    y, x= obj.size()
    mse = torch.pow(predict-obj, 2).sum()/(y*x)
    return mse
