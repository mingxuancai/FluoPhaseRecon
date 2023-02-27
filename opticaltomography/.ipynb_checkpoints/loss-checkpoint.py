import torch
import numpy as np
import torch.nn as nn



def mse_loss(predict, obj, device='cpu'):
    
    y, x = obj.size()
    mse = torch.pow(predict-obj, 2).sum()/(y*x)
    # print(mse.device)
    return mse


def sparsity_loss(obj, weight, device='cpu'):
    loss = nn.L1Loss()
    # print(obj.device)
    zero = torch.zeros(obj.size()).to(device)
    # print(zero.device)
    return loss(obj, zero)*weight

def total_variation_loss(obj, weight, device='cpu'):
    y, x, z = obj.size()
    tv_h = torch.abs(obj[1,:,:] - obj[-1,:,:]).sum()
    tv_w = torch.abs(obj[:,1,:] - obj[:,-1,:]).sum()
    tv_z = torch.abs(obj[:,:,1] - obj[:,:,-1]).sum()
    
    loss = weight*(tv_h+tv_w+tv_z)/(y*x*z)
    loss = loss.to(device)
    
    return loss


