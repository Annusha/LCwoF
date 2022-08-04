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
import mini_imgnet.resnet12
from utils.util_functions import timing
from utils.logging_setup import viz
import mini_imgnet.episode_training_clean




def catch_inner_fs(logger):
    train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    n_classes = []
    fs_val_dataset = MiniImageNetDataset(mode='nov_test')

    n_classes.append(fs_val_dataset.n_classes)


    fs_train_dict = {
        'train_dataset': fs_val_dataset,
        'train_start_time': train_start_time,
        'n_classes': n_classes
    }


    train = mini_imgnet.episode_training_clean.Training()
    train.training(**fs_train_dict)



@timing
def pipeline_fs():
    # update data-specific parameters
    logger = mini_imgnet.update_arg_parse.update()
    catch_inner_fs(logger)
    if opt.viz:
        viz.env.save([opt.viz_env])






def run_novel(ce=None, marg=None, tripl=None, consist=None, ep_plus=1):

    ########################################################################
    ##### START parameters to adjust  #####

    # pretrained model on base classes
    opt.one_plus2_model = '/BS/kukleva/work/models/mini.imgnet/v13_no.bias_res12+1.base.cl_v0.7994_ep490.pth.tar'
    # data root
    opt.data_root = '/BS/kukleva2/work/data/mini-imagenet_pkl'

    # how many episodes to test; in paper 600 episodes
    opt.n_episodes = 1

    # if to have all stat saved in files
    opt.write_in_file = True
    opt.write_2_phase = True   # during second phase as well
    # to save additional files with all episodes
    opt.file_root = '/BS/kukleva2/work/storage/stat2/'
    opt.log_save_dir = '/BS/kukleva2/nobackup/logs'

    # available released options 5w1s, 5w5s
    opt.n_way = 5
    opt.k_shot = 1

    # how often to test
    opt.test_freq = 10   # during the 2nd phase
    opt.test_freq_crt = 2  # during the 3rd phase

    # to turn on visdom visualization if you set up your server
    # will work only for the first episode, next episdos won't be recorderd by visdom
    # setup visdom server and port in /utils/logging_setup.py
    opt.viz = False


    ##### END parameters to adjust  #####
    ########################################################################

    opt.n_base_classes = 64
    opt.few_shot = True
    opt.fscil = False  # for incremental learning, but not added here
    opt.pfx = ''
    opt.sfx = ''
    opt.init_ep = 1

    opt.viz_env = ''

    lr = 1e-2
    opt.lr = lr
    opt.lr_stem = lr*0.1
    opt.optim = 'sgd'
    opt.new_optim = True

    opt.resnet = 12
    opt.mlp_ly = 1

    opt.viz_meta = True

    opt.weight_decay = 1e-4
    opt.dropout = 0.1
    opt.feat_type = 'raw'

    opt.num_workers = 0
    opt.batch_size = 16
    opt.test_batch_size = 256
    opt.gfsv = True
    opt.train_weights = True
    opt.freeze_layer_from = -1

    opt.train_fc_during_novel = True

    opt.l2_params = True
    if not opt.train_weights:
        opt.l2_params = False

    if opt.k_shot == 1:
        opt.l2_weight = '5e+2'  # 5w1s
    else:
        opt.l2_weight = '5e+3'  # 5w5s

    # True to FIX base sampling
    # false for random base sampling
    opt.base_sampl_fix = True

    opt.clip = 100

    a = True
    opt.crt = a
    opt.crt_weight_training = opt.train_weights
    opt.consistency = a
    opt.crt_lr = 1e-3
    alpha = 0.1
    opt.crt_lr_stem = opt.crt_lr*alpha


    opt.save_scores = False

    if opt.n_episodes > 1:
        opt.viz_meta = False
    else:
        opt.write_in_file = False


    # 2nd phase
    opt.crt_epoch = 150
    # 3rd phase
    ep_plus = 200

    opt.model_name = 'repr.rr26.res%d.%dw%ds.%d' % (opt.resnet, opt.n_way, opt.k_shot, opt.n_episodes)


    if opt.crt:
        opt.epoch_episodes = opt.crt_epoch + ep_plus
        opt.gfsv = True
    else:
        opt.epoch_episodes = opt.crt_epoch

    opt.blf_epochs = ep_plus

    if not opt.write_in_file:
        opt.write_2_phase = False


    opt.pretrain = True

    opt.ce_ft3 = True
    opt.ce_ft3_lymbda = 0.1


    ######### update names based on parameters

    if opt.train_fc_during_novel:
        opt.pfx += '.gce'

    if opt.l2_params:
        opt.pfx += '.l2-%s' % opt.l2_weight
        opt.l2_weight = float(opt.l2_weight)

    if opt.train_weights:
        opt.pfx += f'.crtlr{alpha}'

    if opt.crt:
        opt.pfx += f'.clr{opt.crt_lr}'

    opt.sfx = 'fs.cl'
    if opt.gfsv:
        opt.pfx += str('.gfsv.%dw%ds' % (opt.n_way, opt.k_shot))
    else:
        opt.pfx += str('.%dw%ds' % (opt.n_way, opt.k_shot))
    opt.pre_f = 'max'
    opt.data_sfx = opt.pre_f

    if opt.train_weights:
        opt.pfx = '.ft2' + opt.pfx + '.slr%.3f' % opt.lr_stem
        opt.sfx += '.ft2'
    elif opt.crt_weight_training:
        opt.pfx += '.slr%s' % str(opt.lr_stem)

    if opt.gfsv:
        opt.sfx += '.gfsv'
        if opt.crt:
            if opt.crt_weight_training:
                opt.sfx += '.ft3'

    if not opt.crt:
        opt.crt_epoch = -1

    pipeline_fs()


if __name__ == '__main__':

    run_novel()