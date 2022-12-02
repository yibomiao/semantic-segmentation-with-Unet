import torch
import torch.nn as nn

#act_fn is activate function
#in_dim=3(RGB), out_dim=64 ,expend the dimension of feature.
def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpooling():
    pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    return pool

def conv_block_2(in_dim, out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
    )
    return model

def conv_block_2(in_dim, out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
    )
    return model
