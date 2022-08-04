#!/usr/bin/env python

""" Classes for metric calculations.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


from collections import defaultdict
import numpy as np
import warnings
import torch.nn as nn
import torch
from scipy.stats import hmean
import os

from utils.logging_setup import logger
from utils.arg_parse import opt



class Precision(object):
    def __init__(self, mode='', ses=[]):
        self.mode = mode
        self.ses = ses
        self._top1 = 0
        self._top5 = 0
        self.total = 0

        self._top1_masked = 0
        self._top1_masked_ses = defaultdict(int)
        self._top1_ses = defaultdict(int)
        self._ses_total = defaultdict(int)
        self._top1_novel_ses = 0
        self._top1_novel_ses_masked = 0
        self._top1_novel_ses_total = 0

        self._top1_masked_novel_together = 0

        # segment prediction
        self.videos = defaultdict(dict)

        self.preds = []
        self.output = dict()

    def log(self):
        logger.debug('total items during test: %d' % self.total)
        logger.debug('%s pr@1: %f' % (self.mode.upper(), self.top1()))
        logger.debug('%s pr@5: %f' % (self.mode.upper(), self.top5()))
        logger.debug('%s masked pr@1: %f' % (self.mode.upper(), self.top1_masked()))

    def viz_dict(self):
        return {
            'pr@1/%s' % self.mode.upper(): self.top1(),
            'pr_m@1/%s' % self.mode.upper(): self.top1_masked(),
            # 'pr@5/%s' % self.mode.upper(): self.top5(),
        }

    def viz_dict_fscil(self):
        dd = {
            '%s/pr@1' % self.mode.upper(): self.top1(),
        }
        if 'train' not in self.mode:
            scores = []
            for i in self._top1_masked_ses.keys():
                scores.append(self._top1_ses[i] / self._ses_total[i])
                dd.update({
                    '%s/pr_s%d' % (self.mode.upper(), i): self._top1_ses[i] / self._ses_total[i],
                    '%s/pr(m)_s%d' % (self.mode.upper(), i): self._top1_masked_ses[i] / self._ses_total[i],
                })
            hm_ses = hmean(scores)
            dd.update({'hm_ses': hm_ses})
            two_scores = []
            two_scores.append(self._top1_ses[0] / self._ses_total[0])
            two_scores.append(self._top1_novel_ses / self._top1_novel_ses_total)
            hm_two = hmean(two_scores)
            dd.update({'hm': hm_two})
            dd.update({
                'NOVEL_pr%s' % (self.mode.upper()): self._top1_novel_ses / self._top1_novel_ses_total,
                'NOVEL_pr(m)_%s' % (self.mode.upper()): self._top1_novel_ses_masked / self._top1_novel_ses_total,
                f'NOV.GL_pr(m)_{self.mode.upper()}' : self._top1_masked_novel_together / self._top1_novel_ses_total,
            })

        return dd

    def log_fscil(self):

        logger.debug('total items during test: %d' % self.total)
        logger.debug('%s pr@1: %f' % (self.mode.upper(), self.top1()))
        logger.debug('%s pr@5: %f' % (self.mode.upper(), self.top5()))
        for i in self._top1_masked_ses.keys():
            logger.debug('pr_s%d/%s: %f' % (i, self.mode.upper(), self._top1_ses[i] / self._ses_total[i]))
            logger.debug('pr(m)_s%d/%s: %f' % (i, self.mode.upper(), self._top1_masked_ses[i] / self._ses_total[i]))
        logger.debug('pr_NOVEL %s: %f' % (self.mode.upper(), self._top1_novel_ses / self._top1_novel_ses_total))
        logger.debug('pr(m)_NOVEL %s: %f' % (self.mode.upper(), self._top1_novel_ses_masked / self._top1_novel_ses_total))
        logger.debug(f'pr(GL.m)_NOVEL {self.mode.upper()}, {self._top1_masked_novel_together / self._top1_novel_ses_total}')

    def _to_numpy(self, a):
        torch_types = []
        torch_types.append(torch.Tensor)
        torch_types.append(torch.nn.Parameter)
        if isinstance(a, list):
            return np.array(a)
        if len(torch_types) > 0:
            if isinstance(a, torch.autograd.Variable):
                # For PyTorch < 0.4 comptability.
                warnings.warn(
                    "Support for versions of PyTorch less than 0.4 is deprecated "
                    "and will eventually be removed.", DeprecationWarning)
                a = a.data
        for kind in torch_types:
            if isinstance(a, kind):
                # For PyTorch < 0.4 comptability, where non-Variable
                # tensors do not have a 'detach' method. Will be removed.
                if hasattr(a, 'detach'):
                    a = a.detach()
                return a.cpu().numpy()
        return a

    def update_probs(self, pr_probs=None, gt=None, pr_classes=None, **kwargs):
        '''
        Args:
            pr_probs: None or matrix (batch_size, N_classes) with probabilities (confidence) of the attention to correspond
                sample to one of the classes
            gt: vector (batch_size) with ground truth assignments
            pr_classes: if as input there is a sorted matrix (batch_size, N_classes) where the first column correspond to
             the most probable class for the relative sample
            **kwargs: whatever else can be included as additional parameter

        Returns: all additional return values should be written into self.output['key']

        '''
        self.total += len(gt)
        gt = self._to_numpy(gt)
        if opt.repr_learning and opt.model_name == 'mlp' and not opt.base_cl:
            gt[gt < 0] = 0
            cos = nn.CosineSimilarity()
            a = cos(pr_probs[:, 0], pr_probs[:, 1])
            pr_probs = np.zeros((a.shape[0], 2))
            pr_probs[a.cpu().numpy() < 0.5, 0] = 1
            pr_probs[a.cpu().numpy() > 0.5, 1] = 1
        if len(gt) > 1:
            gt = gt.squeeze()  # make sure that it's vector now, and not the matrix
        if pr_classes is None:
            assert len(pr_probs) == len(gt)
            pr_probs = self._to_numpy(pr_probs)
            pr_classes = np.argsort(-pr_probs, axis=1)
        else:
            pr_classes = self._to_numpy(pr_classes)
            assert len(pr_classes) == len(gt)

        self._top1 += np.sum((pr_classes[:, 0] == gt))
        self._top5 += np.sum([1 for top5_classes, gt_class in zip(pr_classes[:, :5], gt) if gt_class in top5_classes])

        # to have an output confusion matrix
        try:
            conf_mat = kwargs['conf_mat']
            for gt_label, pr_label in zip(gt, pr_classes[:, 0]):
                conf_mat[gt_label, pr_label] += 1
            self.output['conf_mat'] = conf_mat
        except KeyError: pass

    def update_probs_segments(self, pr_probs=None, gt=None, **kwargs):
        ''' Accumulate(average) score for separate segments of the video into one score.
        Args:
            pr_probs: None or matrix (batch_size, N_classes) with probabilities (confidence) of the attention to correspond
                sample to one of the classes
            gt: vector (batch_size) with ground truth assignments
            pr_classes: if as input there is a sorted matrix (batch_size, N_classes) where the first column correspond to
             the most probable class for the relative sample
            **kwargs: whatever else can be included as additional parameter

        Returns: all additional return values should be written into self.output['key']

        '''
        self.total += 1
        gt = self._to_numpy(gt).squeeze()  # make sure that it's vector now, and not the matrix
        assert np.equal(gt, gt[0]).all()
        gt = gt[0]
        pr_probs = self._to_numpy(pr_probs).mean(0)
        pr_classes = np.argsort(-pr_probs)

        self._top1 += np.sum((pr_classes[0] == gt))
        self._top5 += 1 if gt in pr_classes[:5] else 0

        # to have an output confusion matrix
        try:
            conf_mat = kwargs['conf_mat']
            for gt_label, pr_label in zip(gt, pr_classes[:, 0]):
                conf_mat[gt_label, pr_label] += 1
            self.output['conf_mat'] = conf_mat
        except KeyError: pass

    def update_probs_segm_arb(self, pr_probs=None, gt=None, **kwargs):
        video_idxs = kwargs['video_idx']
        uniq_video_idxs = np.unique(video_idxs)
        gt = self._to_numpy(gt).squeeze()
        pr_probs = self._to_numpy(pr_probs)
        if video_idxs.size(0) == 1:
            video_idx = video_idxs[0]
            if video_idxs[0] in self.videos:
                self.videos[video_idx]['probs'] += pr_probs.squeeze()
                self.videos[video_idx]['counter'] += 1
            else:
                self.videos[video_idx]['probs'] = pr_probs.squeeze()
                self.videos[video_idx]['counter'] = 1
                self.videos[video_idx]['gt'] = gt
        else:
            for video_idx in uniq_video_idxs:
                mask = (video_idxs == video_idx).squeeze()
                if video_idx in self.videos:
                    self.videos[video_idx]['probs'] += pr_probs[mask].sum(0)
                    self.videos[video_idx]['counter'] += mask.sum()
                else:
                    self.videos[video_idx]['probs'] = pr_probs[mask].sum(0)
                    self.videos[video_idx]['counter'] = mask.sum()
                    self.videos[video_idx]['gt'] = gt[np.argmax(mask)]

    def finalizer_probs_segm_arb(self, epoch=0, save=False, episode_idx=0):
        self.total = len(self.videos)
        self.predictions = np.zeros(opt.n_base_classes + opt.n_way)
        self.gt_arr = np.zeros(opt.n_base_classes + opt.n_way)
        if save:
            scores = np.zeros((self.total, 2+len(self.videos[0]['probs'])))
        for video_idx in self.videos:
            pr_classes = np.argsort(-self.videos[video_idx]['probs'])
            self._top1 += np.sum((pr_classes[0] == self.videos[video_idx]['gt']))
            self._top5 += 1 if self.videos[video_idx]['gt'] in pr_classes[:5] else 0

            self.predictions[pr_classes[0]] += 1
            self.gt_arr[self.videos[video_idx]['gt']] += 1

            if save:
                scores[video_idx][0] = self.videos[video_idx]['gt']
                scores[video_idx][1] = pr_classes[0]
                scores[video_idx][2:] = self.videos[video_idx]['probs']

            if opt.few_shot:
                vi = list(self.videos.keys())[0]
                mask = np.zeros_like(self.videos[vi]['probs'], dtype=bool)
                # probs = self.videos[video_idx]['probs']
                # if probs[:opt.n_way].mean() > probs[opt.n_way:].mean():
                if self.videos[video_idx]['gt'] < opt.n_way:
                    mask[list(range(opt.n_way))] = True
                else:
                    mask[list(np.arange(opt.n_base_classes) + opt.n_way)] = True
                self.videos[video_idx]['probs'][~mask] = -float('inf')
                pr_classes_masked = np.argsort(-self.videos[video_idx]['probs'])

                self._top1_masked += np.sum((pr_classes_masked[0] == self.videos[video_idx]['gt']))
            else:
                self._top1_masked = 0

        if save:
            save_folder = '/BS/kukleva/work/iccv21/stat/scores/%s/res%d/%s/' % (opt.dataset, opt.resnet, opt.model_name)
            save_file = save_folder + '%s_%s_episode%d_ep%d' % (opt.sfx, opt.log_name, episode_idx, epoch)
            os.makedirs(save_folder, exist_ok=True)

            np.savez(save_file, scores)



    def finalizer_probs_segm_arb_ses(self):
        self.total = len(self.videos)
        if (len(self.ses)-1):
            global_mask_novel = np.zeros_like(self.videos[0]['probs'], dtype=bool)
            for ses_idx_glob_mask in range(1, len(self.ses) - 1):
                global_mask_novel[list(range(self.ses[ses_idx_glob_mask], self.ses[ses_idx_glob_mask + 1]))] = True

        for video_idx in self.videos:
            pr_classes = np.argsort(-self.videos[video_idx]['probs'])
            self._top1 += np.sum((pr_classes[0] == self.videos[video_idx]['gt']))
            self._top5 += 1 if self.videos[video_idx]['gt'] in pr_classes[:5] else 0

            if opt.gce:
                vi = list(self.videos.keys())[0]
                mask = np.zeros_like(self.videos[vi]['probs'], dtype=bool)

                ses_idx = 0
                for ses_idx in range(len(self.ses)-1):
                    if self.videos[video_idx]['gt'] in range(self.ses[ses_idx], self.ses[ses_idx+1]):
                        mask[list(range(self.ses[ses_idx], self.ses[ses_idx+1]))] = True
                        break

                # compute global masking
                if ses_idx != 0:
                    self.videos[video_idx]['probs'][~global_mask_novel] = -float('inf')
                    pr_classes_masked_globally_novel = np.argsort(-self.videos[video_idx]['probs'])
                    self._top1_masked_novel_together += np.sum((pr_classes_masked_globally_novel[0] == self.videos[video_idx]['gt']))

                self.videos[video_idx]['probs'][~mask] = -float('inf')
                pr_classes_masked = np.argsort(-self.videos[video_idx]['probs'])

                self._top1_ses[ses_idx] += np.sum((pr_classes[0] == self.videos[video_idx]['gt']))
                self._top1_masked_ses[ses_idx] += np.sum((pr_classes_masked[0] == self.videos[video_idx]['gt']))
                self._ses_total[ses_idx] += 1

                if ses_idx != 0:
                    self._top1_novel_ses += np.sum((pr_classes[0] == self.videos[video_idx]['gt']))
                    self._top1_novel_ses_masked += np.sum((pr_classes_masked[0] == self.videos[video_idx]['gt']))
                    self._top1_novel_ses_total += 1




    def update_probs_segm_arb_cl2(self, pr_probs=None, cl2_probs=None, gt=None, gt_cl2=None, **kwargs):
        mask_cl2 = np.ones((2, opt.n_way + 64))
        mask_cl2[0, :opt.n_way] = 0
        mask_cl2[1, opt.n_way:] = 0
        video_idxs = kwargs['video_idx']

        uniq_video_idxs = np.unique(video_idxs)
        gt = self._to_numpy(gt).squeeze()
        pr_probs = self._to_numpy(pr_probs)
        if video_idxs.size(0) == 1:
            if cl2_probs[0,0] > cl2_probs[0, 1]:
                mask_idx = 1
            else:
                mask_idx = 0
            video_idx = video_idxs[0]
            if video_idxs[0] in self.videos:
                self.videos[video_idx]['probs'] += pr_probs.squeeze() * mask_cl2[mask_idx]
                self.videos[video_idx]['counter'] += 1
            else:
                self.videos[video_idx]['probs'] = pr_probs.squeeze() * mask_cl2[mask_idx]
                self.videos[video_idx]['counter'] = 1
                self.videos[video_idx]['gt'] = gt
        else:
            for video_idx in uniq_video_idxs:
                mask = (video_idxs == video_idx).squeeze()
                pr_probs_curr = np.zeros_like(pr_probs[0].squeeze())
                for i_idx, i in enumerate(mask):
                    if not i: continue
                    if cl2_probs[i_idx, 0] > cl2_probs[i_idx, 1]:
                        pr_probs_curr += pr_probs[i_idx] * mask_cl2[1]
                    else:
                        pr_probs_curr += pr_probs[i_idx] * mask_cl2[0]
                if video_idx in self.videos:
                    self.videos[video_idx]['probs'] += pr_probs_curr
                    self.videos[video_idx]['counter'] += mask.sum()
                else:
                    self.videos[video_idx]['probs'] = pr_probs_curr
                    self.videos[video_idx]['counter'] = mask.sum()
                    self.videos[video_idx]['gt'] = gt[np.argmax(mask)]



    def top1(self):
        return self._top1 / self.total

    def top5(self):
        return self._top5 / self.total

    def top1_masked(self):
        return self._top1_masked / self.total
