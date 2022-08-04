#!/usr/bin/env

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2020'

import sys
from datetime import datetime

import pathlib
dir_path = pathlib.Path(__file__).parent.parent.resolve()
print(dir_path)
sys.path.append(dir_path)

from utils.arg_parse import opt
from mini_imgnet.dataloader_classification import MiniImageNetDataset
import mini_imgnet.update_arg_parse
import  mini_imgnet.train_base, mini_imgnet.test

import mini_imgnet.resnet12
from utils.util_functions import timing
from utils.logging_setup import viz


def catch_inner(logger):

    model, loss, optimizer = mini_imgnet.resnet12.create_model()

    train_dataset = MiniImageNetDataset(mode='base_train')
    val_dataset = MiniImageNetDataset(mode='base_test')

    test_choice = mini_imgnet.test.testing

    mini_imgnet.train_base.training(train_dataset=train_dataset,
                                    model=model,
                                    loss=loss,
                                    optimizer=optimizer,
                                    val_dataset=val_dataset,
                                    testing=test_choice)




@timing
def pipeline():
    # update data-specific parameters
    logger = mini_imgnet.update_arg_parse.update()
    catch_inner(logger)
    if opt.viz:
        viz.env.save([opt.viz_env])




def run_base():

    # path to save pretrained model
    opt.storage = '/BS/kukleva2/work/storage'
    opt.data_root = '/BS/kukleva2/work/data/mini-imagenet_pkl'


    opt.few_shot = False
    opt.n_way = 5
    opt.n_base_classes = 64
    opt.fscil = False
    opt.train_fc_during_novel = False

    opt.model_name = f''
    opt.resnet = 12
    opt.mlp_ly = 1
    opt.sfx = str('res%d+%d.base.cl' % (opt.resnet, opt.mlp_ly))
    opt.viz = False

    opt.test = False
    opt.feat_type = 'raw'

    opt.train_weights = True
    opt.num_workers = 5
    opt.pretrain = False

    opt.epochs = 500
    opt.batch_size = 16
    opt.test_batch_size = 16
    opt.optim = 'sgd'

    opt.lr = 1e-3
    opt.weight_decay = 5e-4

    opt.extend_distill = False

    opt.save_model = 1
    opt.gfsv = False
    opt.test_freq = 1


    pipeline()


if __name__ == '__main__':
    run_base()