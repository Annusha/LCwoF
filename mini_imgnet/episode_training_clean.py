#!/usr/bin/env

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2020'

import csv
import os
import time
# import mlp.model
from collections import defaultdict
from datetime import datetime

import torch

import mini_imgnet.resnet12
from utils.arg_parse import opt
from utils.eval import Precision
from utils.logging_setup import logger
from utils.logging_setup import viz
from utils.plotting_utils import visdom_plot_losses, visdom_scatter_update
from utils.util_functions import adjust_lr, Meter


def hm(a,b):
    return ((2 * a * b) / (a + b))

def am(a,b):
    return (a+b) / 2

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class Training:
    def __init__(self):
        self.model = None
        self.loss = mini_imgnet.resnet12.CrossEntropyLoss()
        self.train_dataloader = None
        self.test_dataloader = None
        self.test_base_dataloder = None


    def training(self, **kwargs):
        try:
            train_start_time = kwargs['train_start_time']
        except KeyError:
            train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.debug('MLP model, training. Start training time is: %s' % train_start_time)


        loss = mini_imgnet.resnet12.CrossEntropyLoss()

        episodes_meter = Meter()
        masked_novel = Meter()

        if opt.gfsv:
            episodes_meter_base = Meter()
            harmonic_meter = Meter()
            arithmetic_meter = Meter()
            masked_base = Meter()

        if opt.write_in_file:

            total_nov = defaultdict(Meter)
            total_nov_mask = defaultdict(Meter)
            total_base = defaultdict(Meter)
            total_base_mask = defaultdict(Meter)
            total_hm = defaultdict(Meter)
            total_am = defaultdict(Meter)


        train_dataset = kwargs['train_dataset']

        logger.debug('Starting training for %d episodes:' % opt.n_episodes)

        if opt.one_plus2_model.endswith('pth.tar'):
            if opt.device == 'cpu':
                checkpoint = torch.load(opt.one_plus2_model, map_location='cpu')['state_dict']
            else:
                checkpoint = torch.load(opt.one_plus2_model)['state_dict']

            if 'fc.bias' in checkpoint and not opt.bias_classifier :
                del checkpoint['fc.bias']

        train_weights_init = opt.train_weights
        test_fr = opt.test_freq
        l2_weight = opt.l2_weight
        optim = opt.optim

        for episode in train_dataset.get_episodes():
            self.model = None
            opt.test_freq = test_fr
            if episode['idx'] > opt.init_ep:
                opt.viz_meta = False
                if not opt.write_2_phase:
                    opt.test_freq = 1000
                if opt.write_in_file:
                    opt.viz = False
            opt.train_weights = train_weights_init
            opt.l2_weight = l2_weight
            opt.optim = optim

            if opt.l2_params:
                # save deep copy of the parameters of init point
                self.model = mini_imgnet.resnet12.MLP(checkpoint_init=checkpoint)
            else:
                self.model = mini_imgnet.resnet12.MLP()

            if opt.pretrain:
                checkpoint_load = self.model.load_state_dict(checkpoint, strict=True)
                logger.debug(checkpoint_load)
            self.model.fs_init()

            self.model.to(opt.device)

            if opt.crt_epoch == 1:
                episode_ce_lymbda = 0
            else:
                episode_ce_lymbda = 1
            batch_time = Meter()
            data_time = Meter()
            loss_meter = Meter()

            episode['train'].training = True
            self.train_dataloader = torch.utils.data.DataLoader(episode['train'],
                                                           batch_size=opt.batch_size,
                                                           shuffle=True,
                                                           num_workers=opt.num_workers,
                                                           pin_memory=True,
                                                           collate_fn=train_dataset.custom_collate)

            self.test_dataloader = torch.utils.data.DataLoader(episode['test'],
                                                      batch_size=opt.test_batch_size,
                                                      shuffle=False,
                                                      num_workers=opt.num_workers,
                                                      drop_last=False
                                                      )
            if opt.gfsv or opt.crt:
                self.test_base_dataloder = torch.utils.data.DataLoader(episode['base'],
                                                          batch_size=opt.test_batch_size,
                                                          shuffle=False,
                                                          num_workers=opt.num_workers,
                                                          drop_last=False
                                                          )

            logger.debug('Starting training for %d EPISODE' % episode['idx'])

            if opt.optim == 'sgd':
                if opt.train_weights:
                    optimizer = torch.optim.SGD(
                        [
                            {'params': self.model.stem_param(), 'lr': opt.lr_stem},
                            {'params': self.model.novel_cl_param(), 'lr': opt.lr}
                        ],
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay,
                        # nesterov=True,
                    )
                else:
                    optimizer = torch.optim.SGD(
                        [
                            {'params': self.model.novel_cl_param(), 'lr': opt.lr}
                        ],
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)

            adjustable_lr = opt.lr
            adjustable_lr_stem = opt.lr_stem

            blf_epoch = -1
            for epoch in range(opt.epoch_episodes):
                logger.debug('Epoch # %d' % epoch)
                episode['train'].epoch = epoch
                if blf_epoch > -1 or opt.write_2_phase:
                    blf_epoch += 1

                if opt.optim == 'sgd' and opt.resnet == 4:
                    if epoch in [150]:  # sgd, lr 1e-1
                        adjustable_lr = adjust_lr(optimizer, adjustable_lr)



                ############################
                #  decide on protocol: second or third phase
                if opt.crt and episode['train'].epoch == opt.crt_epoch:
                    # change of parameters for the third phase if it's already time
                    if not opt.write_2_phase:
                        blf_epoch += 1
                    opt.gfsv = True
                    if opt.viz_meta or opt.write_in_file:
                        opt.test_freq = opt.test_freq_crt
                    opt.optim = opt.optim if opt.new_optim else ''
                    episode['train'].training = True
                    episode['train'].crt = True

                    # change dataloader to have novel and base samples at the same time
                    self.train_dataloader = torch.utils.data.DataLoader(episode['train'],
                                                                        batch_size=opt.batch_size,
                                                                        shuffle=True,
                                                                        num_workers=0,
                                                                        pin_memory=True,
                                                                        collate_fn=train_dataset.custom_collate)
                    phase2_train_w = opt.train_weights
                    opt.train_weights = opt.crt_weight_training

                    for param in self.model.stem_param():
                        param.requires_grad = (opt.train_weights or opt.cross_c or opt.triplet_loss)
                        if not opt.train_weights:
                            param.grad = None

                    if opt.optim == 'sgd':

                        if opt.train_weights:
                            optimizer = torch.optim.SGD(
                                [
                                    {'params': self.model.stem_param(), 'lr': opt.crt_lr_stem},
                                    {'params': self.model.fc_param(), 'lr': opt.crt_lr},
                                ],
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                # nesterov=True,
                            )
                        else:
                            optimizer = torch.optim.SGD(
                                [
                                    {'params': self.model.fc_param(), 'lr': opt.crt_lr}
                                ],
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

                    else:
                        # pass

                        optimizer.add_param_group({'params':self.model.base_cl_param(), 'lr': opt.crt_lr })
                        if opt.train_weights and not phase2_train_w:
                            optimizer.add_param_group({'params': self.model.stem_param(), 'lr': opt.crt_lr_stem})

                    episode_ce_lymbda = opt.ce_ft3_lymbda


                elif episode['train'].epoch < opt.crt_epoch:
                    opt.gfsv = False

                self.model.train()
                self.model.apply(set_bn_eval)
                #

                n_train_samples = 0
                end = time.time()
                for i, input in enumerate(self.train_dataloader):

                    data_time.update(time.time() - end)
                    labels = input['label']

                    l2_loss = 0
                    output = self.model(input)

                    if opt.gfsv and episode['train'].crt or opt.train_fc_during_novel:
                        # get logits of base classifier
                        self.model.gfsv = True
                        output_gfsv = self.model(input)
                        self.model.gfsv = False
                        output['probs'] = torch.cat((output['probs'], output_gfsv['probs']), dim=1)
                    loss_values = loss(output, input, lymbda=episode_ce_lymbda)

                    if opt.l2_params:
                        l2_loss = self.model.L2_weight_loss()



                    optimizer.zero_grad()
                    loss_meter.update(loss_values.item(), len(labels))
                    loss_values.backward(retain_graph=True)
                    if l2_loss:
                        l2_loss.backward(retain_graph=True)
                    if opt.clip != 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()


                    batch_time.update(time.time() - end)
                    end = time.time()

                    n_train_samples += len(labels)

                    if i % opt.debug_freq == 0:
                        logger.debug('Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f}) {l2:.8f}\t'.format(
                            epoch, i, len(self.train_dataloader), batch_time=batch_time, data_time=data_time, loss=loss_meter, l2=l2_loss))

                logger.debug('Number of training sampler within one epoch: %d' % n_train_samples)
                logger.debug('Loss: %f' % loss_meter.avg)
                loss_meter.reset()


                if opt.test_freq and epoch % opt.test_freq == 0 or (epoch == (opt.epoch_episodes-1)):
                    self.test(mode='fs_train%d' % episode['idx'], epoch=epoch, time_id=train_start_time, episode_idx=episode['idx'])

                    test_output = self.test(mode='fs_val%d' % episode['idx'], epoch=epoch, time_id=train_start_time, episode_idx=episode['idx'])
                    if opt.gfsv or opt.crt:
                        test_output_base = self.test(mode='fs_base%d' % episode['idx'], epoch=epoch, time_id=train_start_time, episode_idx=episode['idx'])
                        if opt.viz and epoch % opt.viz_freq == 0 and opt.viz_meta:
                            dd = {'hm': hm(test_output['top1'], test_output_base['top1']),
                                  'am': am(test_output['top1'], test_output_base['top1'])}
                            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + train_start_time, epoch,
                                               xylabel=('epoch', 'prec'), **dd)

                        if opt.write_in_file and blf_epoch > -1:
                            total_nov[blf_epoch].update(test_output['top1'], 1)
                            total_nov_mask[blf_epoch].update(test_output['top1_masked'], 1)
                            total_base[blf_epoch].update(test_output_base['top1'], 1)
                            total_base_mask[blf_epoch].update(test_output_base['top1_masked'], 1)
                            total_hm[blf_epoch].update(hm(test_output['top1'], test_output_base['top1']), 1)
                            total_am[blf_epoch].update(am(test_output['top1'], test_output_base['top1']), 1)


                    logger.debug(opt.model_name)
                    logger.debug('%s %s' % (opt.sfx, opt.dataset))
                    logger.debug(opt.log_name)

            episodes_meter.update(test_output['top1'], 1)
            masked_novel.update(test_output['top1_masked'], 1)
            logger.debug('FewS Ep %d: %.3f | (%.3f)\t(avg %.3f, std %.3f)\n' % (episode['idx'], test_output['top1'], masked_novel.avg, episodes_meter.avg, episodes_meter.std))
            if opt.gfsv or opt.crt:
                episodes_meter_base.update(test_output_base['top1'], 1)
                masked_base.update(test_output_base['top1_masked'], 1)
                logger.debug('BASE Ep %d: %.3f | (%.3f)\t(avg %.3f, std %.3f)\n' % (episode['idx'], test_output_base['top1'], masked_base.avg, episodes_meter_base.avg,
                                                                           episodes_meter_base.std))
                harmonic_meter.update(hm(test_output['top1'], test_output_base['top1']), 1)
                arithmetic_meter.update(am(test_output['top1'], test_output_base['top1']), 1)
                logger.debug('HM Ep %d: %.3f\t(avg %.3f, std %.3f)\n' % (episode['idx'], harmonic_meter.val, harmonic_meter.avg,
                                                                         harmonic_meter.std))
                logger.debug('AM Ep %d: %.3f\t(avg %.3f, std %.3f)\n' % (episode['idx'], arithmetic_meter.val, arithmetic_meter.avg,
                                                                         arithmetic_meter.std))





            if opt.viz:
                viz_dict = {'nov': test_output['top1'], 'avg_nov': episodes_meter.avg, 'avg_nov_m': masked_novel.avg}
                if opt.gfsv:
                    viz_dict.update({'base': test_output_base['top1'], 'avg_base': episodes_meter_base.avg, 'avg_base_m': masked_base.avg,
                                     'hm': harmonic_meter.avg, 'am': arithmetic_meter.avg})
                visdom_scatter_update(viz.env, 'text-%s-%s' % (opt.log_name, train_start_time), x=episode['idx'],
                                      xylabel=('epis#', 'prec'), **viz_dict)


            if opt.write_in_file:
                file_root = opt.file_root
                file_path_hm = file_root + '/%s/res%d/%s/%s_%s_hm.am.csv' % (opt.dataset, opt.resnet, opt.model_name, opt.sfx, opt.log_name)
                file_path_hm_var = file_root + '/%s/res%d/%s/%s_%s_hm.am.mean.var.csv' % (opt.dataset, opt.resnet, opt.model_name, opt.sfx, opt.log_name)
                file_path_gen = file_root + '/%s/res%d/%s/%s_%s_gen.csv' % (opt.dataset, opt.resnet, opt.model_name, opt.sfx, opt.log_name)
                file_path_mask = file_root + '/%s/res%d/%s/%s_%s_mask.csv' % (opt.dataset, opt.resnet, opt.model_name, opt.sfx, opt.log_name)
                file_path_clean = file_root + '/%s/res%d/%s/%s_%s_clean.csv' % (opt.dataset, opt.resnet, opt.model_name, opt.sfx, opt.log_name)
                os.makedirs(file_root + '/%s/res%d/%s' % (opt.dataset, opt.resnet, opt.model_name), exist_ok=True)

                # WRITE MEAN OF HM AND AM
                existed = os.path.exists(file_path_hm)
                with open(file_path_hm, 'a+') as f_csv:
                    wr = csv.writer(f_csv)
                    if not existed:
                        wr.writerow(["sep=,"])
                        line = [elem + str(idx) for idx in total_am.keys() for elem in ('am', 'hm', '-')]
                        # line = [i + '%d' % (idx // 3) for idx, i in enumerate(['am', 'hm', '-'] * opt.blf_epochs)]
                        wr.writerows([line])

                    vals = []
                    for idx in total_am.keys():
                        vals.append(total_am[idx].avg)
                        vals.append(total_hm[idx].avg)
                        vals.append(' ')
                    wr.writerows([vals])

                # WRITE CONFIDENCE INTERVALS OF HM AND AM
                existed = os.path.exists(file_path_hm_var)
                with open(file_path_hm_var, 'a+') as f_csv:
                    wr = csv.writer(f_csv)
                    if not existed:
                        wr.writerow(["sep=,"])
                        line = [elem + str(idx) for idx in total_am.keys() for elem in ('am', 'hm', '-')]
                        # line = [i + '%d' % (idx // 3) for idx, i in enumerate(['am', 'hm', '-'] * opt.blf_epochs)]
                        wr.writerows([line])

                    vals = []
                    for idx in total_am.keys():
                        vals.append(total_am[idx].std)
                        vals.append(total_hm[idx].std)
                        vals.append(' ')
                    wr.writerows([vals])

                # WRITE GENERALIZED BASE AND NOVEL ACCURACY
                existed = os.path.exists(file_path_gen)
                with open(file_path_gen, 'a+') as f_csv:
                    wr = csv.writer(f_csv)
                    if not existed:
                        wr.writerow(["sep=,"])
                        line = [elem + str(idx) for idx in total_base.keys() for elem in ('gB', 'gN', '-')]
                        # line = [i + '%d' % (idx // 3) for idx, i in enumerate(['GB', 'GN', '-'] * opt.blf_epochs)]
                        wr.writerows([line])

                    vals = []
                    for idx in total_base.keys():
                        vals.append(total_base[idx].avg)
                        vals.append(total_nov[idx].avg)
                        vals.append(' ')
                    wr.writerows([vals])

                # WRITE MASKED BASE AND NOVEL ACCURACY
                existed = os.path.exists(file_path_mask)
                with open(file_path_mask, 'a+') as f_csv:
                    wr = csv.writer(f_csv)
                    if not existed:
                        wr.writerow(["sep=,"])
                        line = [elem + str(idx) for idx in total_base_mask.keys() for elem in ('B', 'N', '-')]
                        # line = [i + '%d' % (idx // 3) for idx, i in enumerate(['B', 'N', '-'] * opt.blf_epochs)]
                        wr.writerows([line])

                    vals = []
                    for idx in total_base_mask.keys():
                        vals.append(total_base_mask[idx].avg)
                        vals.append(total_nov_mask[idx].avg)
                        vals.append(' ')
                    wr.writerows([vals])

                # WRITE accuracy per episode generalized
                existed = os.path.exists(file_path_clean)
                with open(file_path_clean, 'a+') as f_csv:
                    wr = csv.writer(f_csv)
                    if not existed:
                        wr.writerow(["sep=,"])
                        line = [elem + str(idx) for idx in total_base.keys() for elem in
                                ('clB', 'clN', '-')]
                        # line = [i + '%d' % (idx // 3) for idx, i in enumerate(['B', 'N', '-'] * opt.blf_epochs)]
                        wr.writerows([line])

                    vals = []
                    for idx in total_base.keys():
                        vals.append(total_base[idx].val)
                        vals.append(total_nov[idx].val)
                        vals.append(' ')
                    wr.writerows([vals])




    def test(self, mode, epoch=1, time_id='', episode_idx=0):

        if 'train' in mode:
            test_loader = self.train_dataloader
        if 'val' in mode:
            test_loader = self.test_dataloader
        if 'base' in mode:
            test_loader = self.test_base_dataloder

        meter = Meter(mode=mode, name='loss')
        self.model.eval()
        prec = Precision(mode)
        with torch.no_grad():
            for idx, input in enumerate(test_loader):
                labels = input['label']
                # if len(labels) == 1 and bs != 1: raise EnvironmentError('LAZY ANNA')
                output = self.model(input)
                # if opt.gfsv or opt.train_fc_during_novel:
                    # if opt.gfsv:
                self.model.gfsv = True
                output_gfsv = self.model(input)
                self.model.gfsv = False
                output['probs'] = torch.cat((output['probs'], output_gfsv['probs']), dim=1)
                loss_values = self.loss(output, input)
                meter.update(loss_values.item(), len(labels))
                prec.update_probs_segm_arb(pr_probs=output['probs'], gt=labels, video_idx=input['video_idx'])

        # torch.cuda.empty_cache()
        if opt.save_scores and 'train' not in mode:
            prec.finalizer_probs_segm_arb(epoch=epoch, save=True, episode_idx=episode_idx)
        else:
            prec.finalizer_probs_segm_arb()
        meter.log()
        prec.log()

        # del test_loader
        # gc.collect()

        if opt.viz and epoch % opt.viz_freq == 0 and opt.viz_meta:
            # visdom_plot_losses(viz.env, opt.log_name + '-loss-' + time_id, epoch,
            #                    xylabel=('epoch', 'loss'), **meter.viz_dict())
            visdom_plot_losses(viz.env, opt.log_name + '-prec-' + time_id, epoch,
                               xylabel=('epoch', 'prec'), **prec.viz_dict())

        output_dict = {'top1': prec.top1(), 'top1_masked': prec.top1_masked(),
                       'gt_arr': prec.gt_arr, 'pred_arr': prec.predictions}
        # self.model.train()
        return output_dict