import os
import torch
import statistics
import numpy as np
import torch.nn as nn

def get_std_dev(arr):
    return statistics.stdev(arr)

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def conv_info_from_features(features):
    kernel_sizes, strides, paddings = [], [], []
    for layer in features:
        if isinstance(layer, nn.Conv2d):
            kernel_sizes.append(layer.kernel_size[0])
            strides.append(layer.stride[0])
            paddings.append(layer.padding[0])
        elif isinstance(layer, nn.MaxPool2d):
            ks = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            st = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            pd = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            kernel_sizes.append(ks)
            strides.append(st)
            paddings.append(pd)
    return kernel_sizes, strides, paddings
