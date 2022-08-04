#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import os.path as ops
import numpy as np
import torch
import time
import re
import copy

from utils.logging_setup import logger
from utils.arg_parse import opt


class FloatCst:
    def __init__(self, val_prf, val_sfx):
        self.val_prf = val_prf
        self.val_sfx = val_sfx

    def __str__(self):
        return '%d-e%d' % (self.val_prf, self.val_sfx)

class LinearWarmUp():
    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.schedule(iteration - self.length)


class Lables:
    def __init__(self, path, nums=False):
        self.label2idx = {}
        self.idx2label = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                rr = re.match(r'(\d*)\s*([\w*|\s*|-|\d*|.]*)', line)
                if rr is not None:
                    idx = int(rr.group(1))
                    label = rr.group(2)
                    # idx, label = line.split()
                    # idx = int(idx)
                    if nums:
                        label = int(label)
                    self.label2idx[label] = idx
                    self.idx2label[idx] = label
        self._length = len(self.label2idx)
        assert len(self.label2idx) == len(self.idx2label)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.label2idx[item]
        if isinstance(item, int):
            return self.idx2label[item]

    def __len__(self):
        return self._length

class Meter(object):
    def __init__(self, mode='', name=''):
        self.mode = mode
        self.name = name
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        self.std = 0
        self._val_arr = []

    def log(self):
        logger.debug('%s %s: %f' % (self.mode.upper(), self.name, self.avg))

    def viz_dict(self):
        return {
            '%s/%s' % (self.name, self.mode.upper()): self.avg
        }

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        if n == 1:
            self._val_arr.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self._val_arr:
            self.std = np.std(self._val_arr) * 0.95 / np.sqrt(len(self._val_arr))

class MeterArray(object):
    def __init__(self, mapping=None):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.mapping = mapping

    def reset(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None

    def update(self, val, n=1):
        self.val = val
        self.sum += 1

def join_data(data1, data2, f):
    """Simple use of numpy functions vstack and hstack even if data not a tuple
    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy or str 'v'/'h'
    Returns:
        Joined data with provided method.
    """
    def _no_data(data):
        if data is None: return True
        if isinstance(data, np.ndarray):
            if data.shape[0] == 0: return True
        if data == []: return True
        return False

    if isinstance(f, str):
        if f == 'v': f = np.vstack
        elif f == 'h': f = np.hstack
        else: raise IOError('wrong input')

    if isinstance(data1, torch.Tensor):
        data1 = data1.numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.numpy()
    if isinstance(data2, tuple):
        data2 = f(data2)
    if _no_data(data2):
        data2 = data1
    elif not _no_data(data1):
        data2 = f((data1, data2))
    return data2


def adjust_lr(optimizer, old_lr, factor=0.1):
    """Decrease learning rate by 0.1 during training"""
    new_lr = old_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def load_model(name='mlp_text'):
    if opt.resume_str:
        resume_str = opt.resume_str
    else:
        resume_str = '%s.pth.tar' % opt.log_name
    opt.resume_str = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(ops.join(opt.store_root, 'models', name, resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(ops.join(opt.store_root, 'models', name, resume_str))
    checkpoint = checkpoint['state_dict']
    print('loaded attention: ' + ' %s' % resume_str)
    return checkpoint


def load_optimizer():
    if opt.resume_str:
        resume_str = opt.resume_str
    else:
        resume_str = '%s.pth.tar' % opt.log_name
    opt.resume_str = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(ops.join(opt.store_root, 'models', opt.model_name, resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(ops.join(opt.store_root, 'models', opt.model_name, resume_str))
    checkpoint = checkpoint['optimizer']
    print('loaded optimizer')
    return checkpoint


def timing(f):
    """Wrapper for functions to measure time"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
                     % (f, (time2-time1)*1000.0,
                        (time2-time1)/60.0,
                        (time2-time1)))
        return ret
    return wrap
