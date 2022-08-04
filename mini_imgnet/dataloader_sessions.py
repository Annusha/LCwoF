#!/usr/bin/env

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2020'


from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from collections import defaultdict
import torchvision.transforms as transforms
import os
import os.path as ops
import pandas as pd
import json
import numpy as np
from PIL import Image

from utils.arg_parse import opt
from utils.util_functions import Lables
from utils.logging_setup import logger


class MiniImageNetDataset_Sessions(Dataset):
    def __init__(self, mode='base_train'):
        super(MiniImageNetDataset_Sessions, self).__init__()
        logger.debug('init data %s' % mode)
        self.root = '/BS/kukleva2/work/data/mini-imagenet_pkl/fscil/data_splits/'
        self.mode = mode
        self.session_idx = 1
        self.blf = False
        self.base_train = None

        # self.ses_labels = [Lables(self.root + 'session2_classes.txt')]

        if mode == 'test':
            # self.base_labels = Lables(self.root + 'base_classes.txt')
            # self.base_test = pd.read_pickle(self.root + 'base_test.pickle')

            logger.debug('load base test')
            # self.ses_labels = [Lables(self.root + 'base_classes.txt'),
            #                    Lables(self.root + 'session2_classes.txt')]
            # self.ses_data = [pd.read_pickle(self.root + 'base_test.pickle'),
            #                  pd.read_pickle(self.root + 'test_session2.pickle')]
            self.ses_labels = [Lables(self.root + 'base_classes.txt')]
            self.ses_data = [pd.read_pickle(self.root + 'base_test.pickle')]

            self.label_bias = [0, len(self.ses_data[0]['labels'])]
            # self.label_bias = [0]
            self.gt_bias = [0, len(self.ses_labels[0])]

        if mode in ['train', 'iw_base']:
            # self.ses_labels = [Lables(self.root + 'session2_classes.txt')]
            # self.ses_data = [pd.read_pickle(self.root + 'train_session2.pickle')]
            self.ses_labels = []
            self.ses_data = []
            self.label_bias = [0, opt.n_base_classes * opt.k_shot]
            self.gt_bias = [0, opt.n_base_classes]

            if mode == 'iw_base':
                self.load_replay()
                self.label_bias = [0, len(self.base_train['labels'])]

        self.data_mean = np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
        self.data_std = np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        normalize = transforms.Normalize(mean=self.data_mean, std=self.data_std)

        if mode in ['test', 'iw_base']:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(92),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        if mode in ['train']:

            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(84),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )


        self.data = lambda idx: self.pkl['data'][idx]

    def __len__(self):
        if self.mode == 'train':
            if self.blf:
                l = 0
                for ses_idx in range(len(self.ses_data)):
                    l += len(self.ses_data[ses_idx]['labels'])
                return l + self.base_bias * opt.k_shot
            else:
                return len(self.ses_data[-1]['labels'])
        if self.mode == 'test':
            # l = len(self.base_test['labels'])
            l = 0
            for d in self.ses_data:
                l += len(d['labels'])
            return l
        if self.mode == 'iw_base':
            l = 0
            for ses_idx in range(len(self.ses_data)):
                l += len(self.ses_data[ses_idx]['labels'])
            return l + len(self.base_train['labels'])

    def load_session(self):
        self.session_idx += 1
        logger.debug(f'load session {self.session_idx} {self.mode}')
        self.ses_labels.append(Lables(self.root + 'session%d_classes.txt' % self.session_idx))
        gt_bias_update = self.gt_bias[-1] + len(self.ses_labels[-1])
        self.gt_bias.append(gt_bias_update)
        if self.mode == 'test':
            self.ses_data.append(pd.read_pickle(self.root + 'test_session_%d.pickle' % self.session_idx))
            self.label_bias.append(len(self.ses_data[-1]['labels']))
        if self.mode in ['train', 'iw_base']:
            self.ses_data.append(pd.read_pickle(self.root + 'train_session_%d.pickle' % self.session_idx))
            self.label_bias.append(len(self.ses_data[-1]['labels']))

    def load_replay(self):
        # change to pick replay
        if self.base_train is None:
            logger.debug('load base train')
            self.base_train = pd.read_pickle(self.root + 'base_train.pickle')
            self.base_labels = Lables(self.root + 'base_classes.txt')
            self.base_bias = len(self.base_labels)
            self.labels_per_class = defaultdict(list)


    def __getitem__(self, idx):
        output = {
            'video_idx': idx,
        }
        if self.mode in ['train', 'iw_base']:
            if self.blf:
                if self.mode == 'iw_base' and idx < len(self.base_train['labels']):
                    label_txt = self.base_train['labels'][idx]
                    label = self.base_labels[label_txt]
                    data = self.base_train['data'][idx]

                elif idx < self.base_bias * opt.k_shot:
                    label = idx % self.base_bias
                    label_txt = self.base_labels[label]
                    n_sample = idx % opt.k_shot
                    # sample_idx = np.where(np.array(self.base_train['labels']) == label_txt)[0][n_sample]
                    if len(self.labels_per_class[label_txt]) < opt.k_shot:
                        sample_idx = np.random.choice(np.where(np.array(self.base_train['labels']) == label_txt)[0])
                        self.labels_per_class[label_txt].append(sample_idx)
                    else:
                        sample_idx = self.labels_per_class[label_txt][n_sample]
                    data = self.base_train['data'][sample_idx]

                else:
                    ses_idx = 0
                    bias = self.label_bias[ses_idx + 1]
                    while idx >= bias:
                        ses_idx += 1
                        idx -= bias
                        bias = self.label_bias[ses_idx + 1]
                    # ses_idx - 1 because base data is separately
                    data = self.ses_data[ses_idx-1]['data'][idx]
                    label_txt = self.ses_data[ses_idx-1]['labels'][idx]
                    label = self.ses_labels[ses_idx-1][label_txt] + self.gt_bias[ses_idx]
                    #
                    # sample_idx = idx - self.base_bias * opt.k_shot
                    # data = self.ses_data[-1]['data'][sample_idx]
                    # label_txt = self.ses_data[-1]['labels'][sample_idx]
                    # label = self.ses_labels[-1][label_txt] + self.base_bias  # only for the second session, won't work further
            else:
                data = self.ses_data[-1]['data'][idx]
                label_txt = self.ses_data[-1]['labels'][idx]
                if opt.gce:
                    label = self.ses_labels[-1][label_txt] + self.gt_bias[self.session_idx-1]
                else:
                    label = self.ses_labels[-1][label_txt]


        if self.mode == 'test':
            ses_idx = 0
            bias = self.label_bias[ses_idx+1]
            while idx >= bias:
                ses_idx += 1
                idx -= bias
                bias = self.label_bias[ses_idx+1]
            data = self.ses_data[ses_idx]['data'][idx]
            label_txt = self.ses_data[ses_idx]['labels'][idx]
            label = self.ses_labels[ses_idx][label_txt] + self.gt_bias[ses_idx]

        data = Image.fromarray(data).convert('RGB')
        data = self.transform(data)

        output.update({
            'label': label,
            'data': data
        })

        return output


