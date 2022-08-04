#!/usr/bin/env python

""" Create episodes for ucf dataset
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2020'


from collections import defaultdict
import os.path as ops
import pandas as pd
import numpy as np
import json
# import csv
# import os

from utils.arg_parse import opt

# splits_root = '/BS/kukleva/work/data/ucf/data_splits'
splits_root = '/BS/kukleva2/work/data/mini-imagenet_pkl/'

def get_classes(idx):
    subname = {0: 'nov_test',
               1: 'val'}
    classes_path = ops.join(splits_root, '%s_classes.txt' % subname[idx])
    classes_list = []
    classes_idxs = []
    lab2idx, idx2lab = {}, {}
    with open(classes_path, 'r') as f:
        for line in f:
            line_idx = line.strip().split()[0]
            line = line.strip().split()[1]
            classes_list.append(int(line))
            classes_idxs.append(int(line_idx))
            lab2idx[line] = int(line_idx)
            idx2lab[int(line_idx)] = int(line)
    return {
        'classes_list': classes_list,
        'classes_idxs': classes_idxs,
        'lab2idx': lab2idx,
        'idx2lab': idx2lab,

    }
    # return classes_list

def create_episodes(idx):
    # set for episods creations
    subname = {0: 'nov_test',
               1: 'nov_val'}
    # form list of classes
    classes_d = get_classes(idx)

    classes_list = classes_d['classes_list']
    classes_idxs = classes_d['classes_idxs']
    lab2idx = classes_d['lab2idx']

    assert ops.exists(ops.join(splits_root, 'miniImageNet_category_split_%s.pickle' % subname[idx]))
    data_csv = pd.read_pickle(ops.join(splits_root, 'miniImageNet_category_split_%s.pickle' % subname[idx]))

    # create dict by class
    data_dict = defaultdict(list)
    for idx_file in range(len(data_csv['labels'])):
        data_dict[data_csv['labels'][idx_file]].append(str(idx_file))

    episodes = {}
    for ep_n in range(opt.n_episodes):
        ep_labels = np.random.choice(classes_list, opt.n_way, replace=False)
        fs_train_data = {'data': [], 'label': []}
        fs_test_data = {'data': [], 'label': []}
        map_gt2train = []
        for idx_train, gt_lab in enumerate(ep_labels):
            map_gt2train.append('%d %d' % (gt_lab, idx_train))
        for ep_lab in ep_labels:
            samples = np.random.choice(data_dict[ep_lab], opt.k_shot + 15, replace=False)
            fs_train_data['data'].extend(samples[:opt.k_shot])
            fs_train_data['label'].extend([str(ep_lab)] * opt.k_shot)

            fs_test_data['data'].extend(samples[opt.k_shot:])
            fs_test_data['label'].extend([str(ep_lab)] * 15)
            # fs_test_data['data'].append(samples[-1])
            # fs_test_data['label'].append(ep_lab)
        episodes['ep%05d' % ep_n] = {'train': fs_train_data,
                                     'test': fs_test_data,
                                     'mapping_gt2train': map_gt2train}

    p = ops.join(splits_root, '%s_episodes_%dshot_%dway_15_test.json' % (subname[idx], opt.k_shot, opt.n_way))
    with open(p, 'w') as f:
        json.dump(episodes, f, sort_keys=True, indent=4, separators=(',', ': '))

    print('%s episodes are written' % subname[idx])
    print(p)


if __name__ == '__main__':
    # train_val_files(4)
    # join_extract_features()
    opt.n_way = 20
    opt.k_shot = 1
    opt.n_episodes = 2000
    create_episodes(0)