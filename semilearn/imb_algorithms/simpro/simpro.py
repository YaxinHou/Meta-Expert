# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@IMB_ALGORITHMS.register('simpro')
class SimPro(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(SimPro, self).__init__(args, net_builder, tb_log, logger)

        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(lb_class_dist / lb_class_dist.sum())
        
        self.scale_ratio = args.ulb_dest_len / args.lb_dest_len
        
        self.ema_u = args.ema_u
        
        self.tau = args.tau

        self.py_con = self.lb_class_dist.cuda(args.gpu)
        self.py_uni = torch.ones(self.num_classes).cuda(args.gpu) / self.num_classes
        self.py_rev = torch.flip(self.py_con, dims=[0])

        self.adapt_dis = self.py_con
        self.estimate_dis = self.py_uni

        self.adapt_dis_adjustment = torch.log(self.adapt_dis ** self.tau + 1e-12)
        self.estimate_dis_adjustment = torch.log(self.estimate_dis ** self.tau + 1e-12)


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")
            
            self.adapt_dis_adjustment = torch.log(self.adapt_dis ** self.tau + 1e-12)
            self.estimate_dis_adjustment = torch.log(self.estimate_dis ** self.tau + 1e-12)
            
            self.count_labeled_dataset = torch.zeros(self.num_classes).cuda(self.args.gpu)
            self.dis_unlabeled_dataset = torch.zeros(self.num_classes).cuda(self.args.gpu)

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb
        num_lb = lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x = outputs['logits'][:num_lb]
                logits_u_w, logits_u_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_u_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_u_w = outs_x_ulb_w['logits']

            pseudo_label = torch.softmax(logits_u_w.detach() + self.estimate_dis_adjustment, dim=-1)
            
            max_probs, max_indexs = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.p_cutoff)
            
            self.count_labeled_dataset += torch.bincount(lb, minlength=self.num_classes)
            self.dis_unlabeled_dataset += torch.sum(pseudo_label[mask], dim=0)
            
            Lx = (F.cross_entropy(logits_x + self.adapt_dis_adjustment, lb, reduction="mean")) / self.scale_ratio

            Lu = (F.cross_entropy(logits_u_s + self.adapt_dis_adjustment, pseudo_label, reduction="none") * mask.float()).mean()
            
            estimate_dis = self.dis_unlabeled_dataset / (self.dis_unlabeled_dataset.sum() + 1)

            self.estimate_dis = self.estimate_dis * self.ema_u + (estimate_dis) * (1 - self.ema_u)

            count_forward = self.count_labeled_dataset + self.dis_unlabeled_dataset

            self.adapt_dis = self.adapt_dis * self.ema_u + (count_forward / count_forward.sum()) * (1 - self.ema_u)
            
            total_loss = Lx + Lu
            
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=Lx.item(), 
                                         unsup_loss=Lu.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())

        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tau', float, 1),
            SSL_Argument('--ema_u', float, 0.9),
        ]
