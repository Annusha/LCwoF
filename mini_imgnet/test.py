#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2020'

import torch
import gc

from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter
from utils.logging_setup import viz
from utils.eval import Precision
from utils.arg_parse import opt


# def testing(test_dataset, model, loss, epoch=1, mode='val', time_id=''):
def testing(**kwargs):
    test_dataset = kwargs['test_dataset']
    try:
        base_dataset = kwargs['base_dataset']
        # base_dataset = None
    except KeyError:
        base_dataset = None
    model = kwargs['model']
    loss = kwargs['loss']
    mode = kwargs['mode']
    try:
        epoch = kwargs['epoch']
    except KeyError:
        epoch=1
    try:
        time_id = kwargs['time_id']
    except KeyError:
        time_id = ''

    # torch.cuda.empty_cache()
    bs = opt.batch_size
    try:
        test_dataset.training = False
        bs = opt.test_batch_size
        # test_dataset.crt = False
        # if not test_dataset.training:
        #     bs = opt.test_batch_size
        # else:
    except: pass
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=bs,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              drop_last=False
                                              )


    meter = Meter(mode=mode, name='loss')
    model.eval()
    prec = Precision(mode)
    with torch.no_grad():
        for idx, input in enumerate(test_loader):
            labels = input['label']
            # if len(labels) == 1 and bs != 1: raise EnvironmentError('LAZY ANNA')
            output = model(input)
            if opt.gfsv or opt.train_fc_during_novel:
            # if opt.gfsv:
                model.gfsv = True
                output_gfsv = model(input)
                model.gfsv = False
                output['probs'] = torch.cat((output['probs'], output_gfsv['probs']), dim=1)
            loss_values = loss(output, input)
            meter.update(loss_values.item(), len(labels))
            prec.update_probs_segm_arb(pr_probs=output['probs'], gt=labels, video_idx=input['video_idx'])

    # torch.cuda.empty_cache()

    prec.finalizer_probs_segm_arb()
    meter.log()
    prec.log()

    del test_loader
    gc.collect()

    if opt.viz and epoch % opt.viz_freq == 0 and opt.viz_meta:
        visdom_plot_losses(viz.env, opt.log_name + '-loss-' + time_id, epoch,
                           xylabel=('epoch', 'loss'), **meter.viz_dict())
        visdom_plot_losses(viz.env, opt.log_name + '-prec-' + time_id, epoch,
                           xylabel=('epoch', 'prec'), **prec.viz_dict())

    output_dict = {'top1': prec.top1(), 'top1_masked': prec.top1_masked(), 'loss': meter.avg}
    return output_dict

    # if base_dataset is not None:
    #     base_loader = torch.utils.data.DataLoader(base_dataset,
    #                                               batch_size=bs,
    #                                               shuffle=False,
    #                                               num_workers=opt.num_workers,
    #                                               drop_last=False
    #                                               )
    #
    #
    #     meter = Meter(mode='base', name='loss')
    #     model.eval()
    #     prec = Precision('base')
    #     with torch.no_grad():
    #         for idx, input in enumerate(base_loader):
    #             labels = input['label']
    #             # if len(labels) == 1 and bs != 1: raise EnvironmentError('LAZY ANNA')
    #             output = model(input)
    #             if opt.gfsv or opt.train_fc_during_novel:
    #             # if opt.gfsv:
    #                 model.gfsv = True
    #                 output_gfsv = model(input)
    #                 model.gfsv = False
    #                 output['probs'] = torch.cat((output['probs'], output_gfsv['probs']), dim=1)
    #             loss_values = loss(output, input)
    #             meter.update(loss_values.item(), len(labels))
    #             prec.update_probs_segm_arb(pr_probs=output['probs'], gt=labels, video_idx=input['video_idx'])
    #
    #     torch.cuda.empty_cache()
    #
    #     prec.finalizer_probs_segm_arb()
    #     meter.log()
    #     prec.log()
    #
    #     del base_loader
    #     gc.collect()
    #
    #     if opt.viz and epoch % opt.viz_freq == 0 and opt.viz_meta:
    #         visdom_plot_losses(viz.env, opt.log_name + '-loss-' + time_id, epoch,
    #                            xylabel=('epoch', 'loss'), **meter.viz_dict())
    #         visdom_plot_losses(viz.env, opt.log_name + '-prec-' + time_id, epoch,
    #                            xylabel=('epoch', 'prec'), **prec.viz_dict())
    #
    #     try:
    #         test_dataset.training = True
    #     except: pass
    #
    # output_dict.update({'top1_base': prec.top1(), 'top1_masked_base': prec.top1_masked()})
    # return output_dict