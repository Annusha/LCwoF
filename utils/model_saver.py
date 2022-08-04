#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


from collections import defaultdict
import os.path as ops
import torch
import os

from utils.arg_parse import opt


class ModelSaver:
    def __init__(self, k=4, path=''):
        '''
        Args:
            k: how many top performing models to save
            path: where to save
        '''
        self.k = k
        self.eval = defaultdict(dict)
        self.models = defaultdict(dict)
        self.worst_idx = defaultdict(lambda: -1)
        self.saved = defaultdict(dict)

        self.path = path
        os.makedirs(path, exist_ok=True)

    def check(self, val:dict):
        # key identify the type of the attention
        for key in val:
            # if the number of saved models at this moment less than k
            if len(self.eval[key]) < self.k: return True
            # if one of the new models is better then the worst out of stored
            if val[key] > self.eval[key][self.worst_idx[key]]: return True
        return False

    def update(self, val, model, epoch):
        for key in val:
            # add the current value to the storage
            self.eval[key][epoch] = val[key]
            self.models[key][epoch] = model
            # if the limit of how many models we can store is exceeded, remove the worst attention
            if len(self.eval[key]) > self.k:
                self.eval[key].pop(self.worst_idx[key])
                self.models[key].pop(self.worst_idx[key])
                self.saved[key].pop(self.worst_idx[key], 0)
            # assign as worst thing the current value
            self.worst_idx[key] = val[key]  # worst idx
            worst = val[key]  # worst val
            # let's find what is really the worst among all what we have now
            for epoch_worst, val_worst in self.eval[key].items():
                if val_worst <= worst:
                    worst = val_worst
                    self.worst_idx[key] = epoch_worst
            assert len(self.eval[key]) <= self.k

    def save(self):
        # save models on the disk
        # delete models from the disk if the were worse
        for key in self.eval:
            os.makedirs(ops.join(self.path, key), exist_ok=True)
            saved = list(self.saved[key].values())
            for filename in os.listdir(ops.join(self.path, key)):
                if ops.join(self.path, key, filename) not in saved:
                    # the attention has to be deleted since it's worse than the current the worst top-k attention
                    os.remove(ops.join(self.path, key, filename))
            for epoch, val in self.eval[key].items():
                # each attention name includes the evaluation value and epoch
                path = ops.join(self.path, key, '%s_%s_%s_v%.4f_ep%d.pth.tar' % (opt.model_name, opt.pfx, opt.sfx, val, epoch))
                if path not in saved:
                    self.saved[key][epoch] = path
                    torch.save(self.models[key][epoch], path)
