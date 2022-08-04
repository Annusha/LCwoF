#!/usr/bin/env python

""" Main parameters for the project. Other parameters are dataset specific.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--device', default='cuda',
                    help='cuda | cpu')
parser.add_argument('--momentum', default=0.9)

#######################################################################################
### VISDOM
parser.add_argument('--viz', default=True, type=bool)
parser.add_argument('--viz_env', default='main')
parser.add_argument('--env_pref', default='')

#######################################################################################
### LOGS
parser.add_argument('--log_mode', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL')
parser.add_argument('--log_save_dir', default='/BS/kukleva2/nobackup/logs')
parser.add_argument('--debug_freq', default=1, type=int)
parser.add_argument('--sfx', default='')
parser.add_argument('--pfx', default='')

# ### STORAGE
# where store models for base pretraining
parser.add_argument('--storage', default='/BS/kukleva2/work/storage')

# ### IMAGES
parser.add_argument('--im_dim', default=84, type=int,
                    help='cropping size for images (now only  for cub dataset, mini and tiered ae always 84x84')
parser.add_argument('--dropout', default=0.1, type=float)


opt = parser.parse_args()