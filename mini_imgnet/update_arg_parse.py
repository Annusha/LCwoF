#!/usr/bin/env python

""" Arguments specific for the particular dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2020'

import json
import torch

from utils.arg_parse import opt
from utils.logging_setup import setup_logger_path


def load_few_shot(path):
    with open(path, 'r') as f:
        splits = json.load(f)
    train = splits['train']
    few_shot = splits['few_shot']
    fs_train_files = splits['few_shot_train']

    return train, few_shot, fs_train_files


def update():
    # list all parameters which should be in the log name
    # map long names of parameters to short abbreviations for the log name
    args_map = {
        'dropout': 'dp%s',
        'weight_decay': 'wd%s',
        'optim': '%s',
        'lr': 'lr%s',
        'batch_size': 'bs%s',
        'feat_type':'%s',

    }

    if not opt.few_shot:
        args_map.update({'epochs': 'ep%s'})

    opt_d = vars(opt)


    opt_d['dataset'] = 'mini.imgnet'

    if not opt.pretrain:
        opt.one_plus2_model = ''
    else:
        assert opt.one_plus2_model.endswith('pth.tar')

    opt_d['viz_env'] = '%s.%s%s_%s.' % (opt.model_name, opt_d['dataset'], opt.env_pref, opt.sfx)
    if torch.cuda.is_available():
        opt_d['device'] = 'cuda'
    else: opt_d['device'] = 'cpu'

    if opt.feat_type in ['raw']:
        opt_d['i_dim'] = 512

    if opt.feat_type == 'raw':
        opt_d['feat_dir'] = ''


    if opt.few_shot:
        if opt.fscil:
            args_map.update({'epochs': 'ep%s',
                             'blf_epoch': '%s+',
                             'k_shot': 'ks%s',
                             })
        else:
            args_map.update({'few_shot':'fs',
                             'epoch_episodes': 'fsep%s',
                             'crt_epoch': '%s+',

                             })

    log_name = ''
    log_name_attr = sorted(args_map)
    for attr_name in log_name_attr:
        attr = getattr(opt, attr_name)
        arg = args_map[attr_name]
        if isinstance(attr, bool):
            attr = arg if attr else '!' + arg
        else:
            attr = arg % str(attr)
        log_name += '%s.' % attr

    if opt.pfx:
        opt.log_name = '%s.' % opt.pfx + log_name
    else:
        opt.log_name = log_name

    logger = setup_logger_path()

    # print in the log file all the set parameters
    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))
    return logger