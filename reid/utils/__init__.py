from __future__ import absolute_import

import torch
import os


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        #print('Successfully make dirs: {}'.format(dir))
    else:
        #print('Existed dirs: {}'.format(dir))
        pass