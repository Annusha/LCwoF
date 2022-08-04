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

class Features:
    def __init__(self):
        self.training = True
        self.features = {}

        self.data_mean = np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
        self.data_std = np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        normalize = transforms.Normalize(mean=self.data_mean, std=self.data_std)

        self.transform_test = transforms.Compose(
            [
                # lambda x: np.asarray(x),
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                normalize
            ]
        )

        self.transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(84, padding=8),
                transforms.RandomResizedCrop(84),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(84),
                # transforms.ColorJitter(brightness=63. / 255.),
                # transforms.ColorJitter(contrast=(0.2, 1.8)),
                # transforms.ColorJitter(saturation=(0.5, 1.5)),
                # transforms.RandomRotation(5),
                # lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,
                # transforms.RandomErasing(),
            ]
        )

    def custom_collate(self, batch):
        batch = default_collate(batch)
        # if opt.extend_distill and -1 in batch['label']:
        #     batch['distill_data'] = batch['data'][batch['label'] == -1]
        #     batch['data'] = batch['data'][batch['label'] != -1]
        #     batch['label'] = batch['label'][batch['label'] != -1]

        return batch


class Episode(Features, Dataset):
    def __init__(self, **kwargs):
        super(Episode, self).__init__()
        self._data = kwargs['data']
        self.data = lambda idx: self._data[idx]
        self.labels = kwargs['labels']
        self.l2idx = defaultdict(list)
        self.base_data = defaultdict(list)
        for key, val in self.labels.items():
            self.l2idx[val].append(key)
        # self.features = {}
        self.training = False
        self.epoch = -1
        self.crt = False
        try:
            self.base_labels = kwargs['base_labels']
            self.base_csv = kwargs['base_csv']
            self.n_classes = kwargs['n_classes']
            self._init_base()
        except KeyError:
            self.base_labels = []
            self.base_csv = []
            self.n_classes = 0
        # if opt.extend_distill:
        #     try:
        #         self.distill_set = kwargs['distill_set']
        #         self._init_distill()
        #     except KeyError:
        #         self.distill_set = []
        #         self.distill_idxs = []

    def _init_base(self):
        # for idx in range(opt.k_shot * len(self.base_labels)):
        for idx in self.base_labels.label2idx:
            # label_name = idx % opt.n_base_classes
            label_name = idx
            # label_name = self.base_labels.idx2label[base_idx]
            base_idx = np.random.choice(np.where(np.array(self.base_csv['labels']) == label_name)[0])

            self.base_data[label_name].append(base_idx)

    def _init_distill(self):
        self.distill_idxs = np.random.choice(np.arange(self.distill_set['data'].shape[0]), opt.extend_distill_n)


    def __len__(self):
        # if opt.n_snip:
        #     return len(self.labels) * opt.n_snip
        # if self.crt and self.epoch < opt.crt_epoch:
        if self.crt and self.epoch >= opt.crt_epoch:
            # data_len = len(self.labels) + opt.n_base_classes * opt.k_shot
            data_len = len(self.labels) + opt.n_base_classes
        else:
            data_len = len(self.labels)
            # if opt.extend_distill:
            #     data_len += len(self.distill_idxs)
        return data_len

    def __getitem__(self, idx):
        output = {}
        target_idx = 0

        if idx >= len(self.labels):
            # if opt.extend_distill and idx >= len(self.labels) and not (self.crt and self.epoch >= opt.crt_epoch):
            #     image_idx = idx - len(self.labels)
            #     output['label'] = -1
            #     image = self.distill_set['data'][self.distill_idxs[image_idx]]
            #     image = Image.fromarray(image).convert('RGB')
            #     image = self.transform_train(image)
            #
            #     output['data'] = image
            # else:
                # crt training
            if opt.n_base_classes in [5,10]:
                label_name = self.base_labels.idx2label[idx % opt.n_base_classes]
                base_idx = idx % opt.n_base_classes
            else:
                label_name = idx % opt.n_base_classes
            # label_name = int(self.base_labels.idx2label[base_idx])
            if opt.base_sampl_fix:
                tg = (idx - len(self.labels)) // opt.n_base_classes
                image_idx = self.base_data[label_name][tg]
            else:
                image_idx = np.random.choice(np.where(np.array(self.base_csv['labels']) == label_name)[0])
            image = self.base_csv['data'][image_idx]
            image = Image.fromarray(image).convert('RGB')
            if self.training:
                image = self.transform_train(image)
            else:
                image = self.transform_test(image)

            output['data'] = image

            if opt.n_base_classes in [5, 10]:
                output['label'] = base_idx + self.n_classes
            else:
                output['label'] = label_name + self.n_classes

        else:
            image = self.data(idx)
            image = Image.fromarray(image).convert('RGB')
            if self.training:
                image = self.transform_train(image)
            else:
                image = self.transform_test(image)

            output['data'] = image

            output['label'] = self.labels[idx]

        output.update({
            'video_idx': idx,
            'target_idx': target_idx
        })

        return output


class MiniImageNetDataset(Features, Dataset):
    def __init__(self, mode='base_train'):
        super(MiniImageNetDataset, self).__init__()
        logger.debug('init data %s' % mode)
        if opt.fscil:
            root = '/BS/kukleva2/work/data/mini-imagenet_pkl/fscil/data_splits/'
        else:
            root = '/BS/kukleva2/work/data/mini-imagenet_pkl/'
        self.mode = mode
        # TODO: either rename, or put name correspondences to the original ones
        if mode in ['base_train', 'base_val', 'base_test']:
            # if opt.n_base_classes == 5:
            #     classes_path = root + 'base_classes5.txt'
            # if opt.n_base_classes == 10:
            #     classes_path = root + 'base_classes10.txt'
            if opt.n_base_classes == 64:
                classes_path = root + 'base_classes.txt'
            if opt.fscil:
                pkl_file = root + '%s.pickle' % mode
            else:
                pkl_file = root + 'miniImageNet_category_split_%s.pickle' % mode
            episodes_file = None

        if mode in ['nov_val', 'nov_test']:
            classes_path = root + '%s_classes.txt' % mode
            pkl_file = root + 'miniImageNet_category_split_%s.pickle' % mode
            # episodes_file = root + '%s_episodes_%dshot_%dway_15_test.json' % (mode, opt.k_shot, opt.n_way)
            episodes_file = root + f'{mode}_episodes_{opt.k_shot}shot_{opt.n_way}way_15_test_iccv21_update.json'


        if opt.fscil:
            self.labels = Lables(classes_path)
        else:
            self.labels = Lables(classes_path, nums=True)
        self.pkl = pd.read_pickle(pkl_file)
        if opt.n_base_classes in [5, 10]:
            self.pkl['mask'] = np.zeros(len(self.pkl['labels']), dtype=bool)
            for ein_label in self.labels.label2idx:
                self.pkl['mask'] += (int(ein_label) == np.array(self.pkl['labels']))

        # self.data_mean = np.mean(self.pkl['data'], axis=(0,1,2))
        # self.data_std = np.std(self.pkl['data'], axis=(0,1,2))
        # self.data_mean = np.array([0.485, 0.456, 0.406])
        # self.data_std = np.array([0.229, 0.224, 0.225])
        self.data_mean = np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
        self.data_std = np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        normalize = transforms.Normalize(mean=self.data_mean, std=self.data_std)

        if mode in ['base_test', 'base_val']:
            self.transform = transforms.Compose(
                [
                    # lambda x: np.asarray(x),
                    transforms.Resize(92),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        if mode in ['base_train']:

            self.transform = transforms.Compose(
                [
                    # transforms.RandomCrop(84, padding=8),
                    transforms.RandomResizedCrop(84),
                    # transforms.ColorJitter(brightness=63. / 255.),
                    # transforms.ColorJitter(contrast=(0.2, 1.8)),
                    # transforms.ColorJitter(saturation=(0.5, 1.5)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(15),
                    # lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize,
                    # transforms.RandomErasing(),
                ]
            )

        if episodes_file is not None:
            with open(episodes_file, 'r') as f:
                self.episodes = json.load(f)
        else:
            self.episodes = None

        self.n_classes = len(self.labels)

        self.data = lambda idx: self.pkl['data'][idx]

        if opt.gfsv:
            logger.debug('read base train / test data')
            self.pkl_base_test = pd.read_pickle(root + 'miniImageNet_category_split_base_test.pickle')
            self.pkl_base_train = pd.read_pickle(root + 'miniImageNet_category_split_base_train.pickle')
            # if opt.n_base_classes == 5:
            #     classes_path_base = root + 'base_classes5.txt'
            # if opt.n_base_classes == 10:
            #     classes_path_base = root + 'base_classes10.txt'
            if opt.n_base_classes == 64:
                classes_path_base = root + 'base_classes.txt'
            self.labels_base = Lables(classes_path_base, nums=True)

            if opt.n_base_classes in [5, 10]:
                self.pkl_base_test['mask'] = np.zeros(len(self.pkl['labels']), dtype=bool)
                self.pkl_base_train['mask'] = np.zeros(len(self.pkl['labels']), dtype=bool)
                for ein_label in self.labels_base.label2idx:
                    self.pkl_base_test['mask'] += (int(ein_label) == np.array(self.pkl['labels']))
                    self.pkl_base_train['mask'] += (int(ein_label) == np.array(self.pkl['labels']))
        # if opt.extend_distill:
        #     logger.debug('read val set')
        #     self.distill_set = pd.read_pickle(root + 'miniImageNet_category_split_nov_val.pickle')

    def __len__(self):
        if opt.n_base_classes in [5,10]:
            return sum(self.pkl['mask'])
        return len(self.pkl['labels'])

    def get_episodes(self):
        init = opt.init_ep
        output = {}
        for idx in range(init, init+opt.n_episodes):
            key = 'ep%05d' % idx

            # mapping
            mapping_txt = self.episodes[key]['mapping_gt2train']
            label_mapping = {}
            for line in mapping_txt:
                line = line.split(' ')  # gt label <-> mapped label
                label_mapping[int(line[0])] = int(line[1])

            train_features, train_labels = {}, {}
            episode = self.episodes[key]['train']
            # augmentation options
            for file_idx, data_idx in enumerate(episode['data']):
                # try:
                data_idx = int(data_idx)

                train_features[file_idx] = self.pkl['data'][data_idx]
                # train_labels[file_idx] = self.labels.label2idx[self.pkl['labels'][data_idx]]
                train_labels[file_idx] = self.pkl['labels'][data_idx]

            # label_mapping = {}
            # for lidx, label_uniq in enumerate(np.unique(list(train_labels.values()))):
            #     label_mapping[label_uniq] = lidx

            mapped_train_labels = {}
            for key_ml, val_ml in train_labels.items():
                mapped_train_labels[key_ml] = label_mapping[val_ml]

            # train_dict = {'data':train_features, 'labels':train_labels}
            train_dict = {'data':train_features, 'labels':mapped_train_labels}

            test_features, test_labels = {}, {}
            episode = self.episodes[key]['test']

            for file_idx, data_idx in enumerate(episode['data']):
                # try:
                data_idx = int(data_idx)
                test_features[file_idx] = self.pkl['data'][data_idx]
                # test_labels[file_idx] = self.labels.label2idx[self.pkl['labels'][data_idx]]
                test_labels[file_idx] = self.pkl['labels'][data_idx]

            mapped_test_labels = {}
            for key_ml, val_ml in test_labels.items():
                mapped_test_labels[key_ml] = label_mapping[val_ml]

            # test_dict = {'data':test_features, 'labels':test_labels}
            test_dict = {'data':test_features, 'labels':mapped_test_labels}



            if opt.gfsv:
                base_features, base_labels = {}, {}
                file_idx = 0

                if opt.n_base_classes in [5, 10]:
                    choice_labels_base = list(self.labels_base.label2idx.keys())
                else:
                    choice_labels_base = list(range(opt.n_base_classes))

                for base_label in choice_labels_base:
                    base_idxs = np.random.choice(np.where(np.array(self.pkl_base_test['labels']) == base_label)[0], 15, replace=False)

                    for base_idx in base_idxs:
                        base_features[file_idx] = self.pkl_base_test['data'][base_idx]
                        # base_labels[file_idx] = self.labels_base[self.csv_base['label'][base_idx]] + len(self.labels)
                        if opt.n_base_classes in [5, 10]:
                            base_labels[file_idx] = self.labels_base.label2idx[self.pkl_base_test['labels'][base_idx]] + opt.n_way
                        else:
                            base_labels[file_idx] = self.pkl_base_test['labels'][base_idx] + opt.n_way
                        file_idx += 1

                output['base'] = Episode(**{'data':base_features, 'labels':base_labels})

                if opt.crt or opt.knn:
                    train_dict['base_labels'] = self.labels_base
                    train_dict['base_csv'] = self.pkl_base_train
                    # train_dict['n_classes'] = len(self.labels)
                    train_dict['n_classes'] = opt.n_way

            # if opt.extend_distill:
            #     train_dict['distill_set'] = self.distill_set

            output.update({'train': Episode(**train_dict),
                           'test': Episode(**test_dict),
                           'idx': idx,
                           'label_mapping': label_mapping,
                           })
            yield output


    def __getitem__(self, idx):
        output = {}
        if opt.n_base_classes in [5, 10]:
            idx = np.where(self.pkl['mask'])[0][idx]
        data = self.pkl['data'][idx]
        data = Image.fromarray(data).convert('RGB')
        data = self.transform(data)

        if opt.fscil:
            label = self.labels[self.pkl['labels'][idx]]
        else:
            label = int(self.pkl['labels'][idx])
            if opt.n_base_classes in [5, 10]:
                label = self.labels.label2idx[label]

        output.update({
            # 'label': self.labels.label2idx[self.pkl['labels'][idx]],
            'label': label,
            'video_idx': idx,
            'data': data
        })

        return output


