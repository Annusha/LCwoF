#!/usr/bin/env python

""" File with the training loop.
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2020'

import torch.backends.cudnn as cudnn
from datetime import datetime
from os.path import join
import numpy as np
import random
import torch
import time
import copy
import os

from utils.util_functions import adjust_lr, Meter
from utils.model_saver import ModelSaver
from utils.logging_setup import logger
from utils.arg_parse import opt
import mini_imgnet.resnet12


def training(**kwargs):
    try:
        train_start_time = kwargs['train_start_time']
    except KeyError:
        train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger.debug('MLP model, training. Start training time is: %s' % train_start_time)

    model = kwargs['model']
    optimizer = kwargs['optimizer']

    loss = kwargs['loss']
    testing = kwargs['testing']

    batch_time = Meter()
    data_time = Meter()
    loss_meter = Meter()

    train_dataset = kwargs['train_dataset']

    # do not modify opt.lr, use everywhere here adjustible_lr
    adjustable_lr = opt.lr
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   drop_last=True,
                                                   pin_memory=True)

    logger.debug('Starting training for %d epochs:' % opt.epochs)

    model_saver_val = ModelSaver(path=join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))
    # model_saver_test = ModelSaver(path=join(opt.storage, 'models', kwargs['name'], opt.log_name, 'test'))
    # if you need test set validation, add corresponding functions below similarly to val set

    for epoch in range(opt.epochs):
        logger.debug('Epoch # %d' % epoch)
        train_dataset.epoch = epoch

        # adjust learning rate if necessary
        # if epoch and epoch in [30, 60]:  # adam, lr 1e-3
        if opt.optim == 'adam':
            if epoch and epoch in [100, 300, 500]:  # adam, lr 1e-3
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
        else:
            if epoch and epoch in [75, 150, 300]:  # sgd, lr 1e-3
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)

        if epoch == 0:
            testing(**{'test_dataset':kwargs['val_dataset'], 'model': model, 'loss':loss, 'epoch':-1, 'mode':'val', 'time_id':train_start_time})

        end = time.time()

        model.to(opt.device)
        model.train()
        try:
            train_dataset.training = True
        except: pass

        n_train_samples = 0
        for i, input in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            labels = input['label']

            if len(labels) == 1: raise EnvironmentError('LAZY ANNA')

            output = model(input)
            loss_values = loss(output, input)
            loss_meter.update(loss_values.item(), len(labels))

            optimizer.zero_grad()

            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            n_train_samples += len(labels)

            if i % opt.debug_freq == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_dataloader), batch_time=batch_time, data_time=data_time, loss=loss_meter))

        logger.debug('Number of training sampler within one epoch: %d' % n_train_samples)
        logger.debug('Loss: %f' % loss_meter.avg)
        loss_meter.reset()

        torch.cuda.empty_cache()

        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0 and (epoch >= 1 or epoch == 0):
            if opt.test_val:
                check_val = testing(**{'test_dataset': kwargs['val_dataset'], 'model': model, 'loss': loss, 'epoch': epoch,
                                       'mode': 'val', 'time_id': train_start_time})


            if opt.save_model and epoch % opt.save_model == 0:
                if model_saver_val.check(check_val):
                    save_dict = {'epoch': epoch,
                                 'state_dict': copy.deepcopy(model.state_dict()),
                                 'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                    model_saver_val.update(check_val, save_dict, epoch)

                logger.debug(opt.log_name)

        if opt.save_model:
            model_saver_val.save()

    # save the last checkpoint of the training
    if opt.save_model:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        os.makedirs(join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name), exist_ok=True)
        torch.save(save_dict,
                   join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'last_%d.pth.tar' % epoch))

    return {'state_dict': model.state_dict(), 'train_start_time': train_start_time, 'epochs': epoch+1}