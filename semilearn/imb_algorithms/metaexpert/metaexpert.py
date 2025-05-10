# Copyright (c) Microsoft Corporation.yonov19v23+v2.py
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn as nn
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from sklearn.metrics import precision_score, recall_score
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

class MetaExpertNet(nn.Module):
    def __init__(self, backbone, num_classes, p_hat_lb, tau_lb1, tau_lb2, tau_lb3, cut1, cut2):
        super().__init__()
        self.cut1 = cut1
        self.cut2 = cut2
        
        self.hat = p_hat_lb
        
        self.backbone = backbone
        self.channels = backbone.channels

        self.tau1 = torch.from_numpy((np.array(tau_lb1)).astype(np.float32))
        self.tau2 = torch.from_numpy((np.array(tau_lb2)).astype(np.float32))
        self.tau3 = torch.from_numpy((np.array(tau_lb3)).astype(np.float32))

        # feat
        self.lin1 = nn.Sequential(nn.Linear(self.channels[0], self.channels[1]), nn.SiLU())
        self.lin2 = nn.Sequential(nn.Linear(self.channels[1], self.channels[2]), nn.SiLU())
        self.lin3 = nn.Sequential(nn.Linear(self.channels[2], self.channels[3]), nn.SiLU())
        self.lin4 = nn.Sequential(nn.Linear(self.channels[3], 2 * self.channels[3]), nn.SiLU())

        self.lin5 = nn.Sequential(nn.Linear(self.channels[1], self.channels[2]), nn.SiLU())
        self.lin6 = nn.Sequential(nn.Linear(self.channels[2], self.channels[3]), nn.SiLU())
        self.lin7 = nn.Sequential(nn.Linear(self.channels[3], 2 * self.channels[3]), nn.SiLU())
        self.lin8 = nn.Sequential(nn.Linear(2 * self.channels[3], 2 * self.channels[3]), nn.SiLU())
        
        # ensemble head
        self.predict = nn.Sequential(nn.Linear(2 * self.channels[3] + 3 * num_classes, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU(),
                                     nn.Linear(64, 3), nn.SiLU()
                                     )
        self.fuse_softmax = nn.Softmax(dim=1)
        
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)

        feat1 = results_dict['feat_for_fuse']['feat1']
        feat2 = results_dict['feat_for_fuse']['feat2']
        feat3 = results_dict['feat_for_fuse']['feat3']
        feat4 = results_dict['feat_for_fuse']['feat4']

        c_logit_1 = results_dict['logits']
        c_logit_2 = results_dict['aux_logits1']
        c_logit_3 = results_dict['aux_logits2']
        
        cp_logit_1 = self.fuse_softmax(c_logit_1)
        cp_logit_2 = self.fuse_softmax(c_logit_2)
        cp_logit_3 = self.fuse_softmax(c_logit_3)
        
        c_logit_x_H_1 = results_dict['logitsH']
        c_logit_x_M_1 = results_dict['logitsM']
        c_logit_x_T_1 = results_dict['logitsT']
        c_logit_x_H_2 = results_dict['aux_logitsH1']
        c_logit_x_M_2 = results_dict['aux_logitsM1']
        c_logit_x_T_2 = results_dict['aux_logitsT1']
        c_logit_x_H_3 = results_dict['aux_logitsH2']
        c_logit_x_M_3 = results_dict['aux_logitsM2']
        c_logit_x_T_3 = results_dict['aux_logitsT2']
        
        min_c_logit_x_H_1 = torch.min(c_logit_x_H_1, dim=1, keepdim=True).values
        min_c_logit_x_M_1 = torch.min(c_logit_x_M_1, dim=1, keepdim=True).values
        min_c_logit_x_T_1 = torch.min(c_logit_x_T_1, dim=1, keepdim=True).values
        min_c_logit_x_H_2 = torch.min(c_logit_x_H_2, dim=1, keepdim=True).values
        min_c_logit_x_M_2 = torch.min(c_logit_x_M_2, dim=1, keepdim=True).values
        min_c_logit_x_T_2 = torch.min(c_logit_x_T_2, dim=1, keepdim=True).values
        min_c_logit_x_H_3 = torch.min(c_logit_x_H_3, dim=1, keepdim=True).values
        min_c_logit_x_M_3 = torch.min(c_logit_x_M_3, dim=1, keepdim=True).values
        min_c_logit_x_T_3 = torch.min(c_logit_x_T_3, dim=1, keepdim=True).values

        lnn_c_logit_x_H_1 = c_logit_x_H_1 - min_c_logit_x_H_1
        lnn_c_logit_x_M_1 = c_logit_x_M_1 - min_c_logit_x_M_1
        lnn_c_logit_x_T_1 = c_logit_x_T_1 - min_c_logit_x_T_1
        lnn_c_logit_x_H_2 = c_logit_x_H_2 - min_c_logit_x_H_2
        lnn_c_logit_x_M_2 = c_logit_x_M_2 - min_c_logit_x_M_2
        lnn_c_logit_x_T_2 = c_logit_x_T_2 - min_c_logit_x_T_2
        lnn_c_logit_x_H_3 = c_logit_x_H_3 - min_c_logit_x_H_3
        lnn_c_logit_x_M_3 = c_logit_x_M_3 - min_c_logit_x_M_3
        lnn_c_logit_x_T_3 = c_logit_x_T_3 - min_c_logit_x_T_3

        l_logit_1 = c_logit_1 + self.tau1 * torch.log(self.hat)
        l_logit_2 = c_logit_2 + self.tau2 * torch.log(self.hat)
        l_logit_3 = c_logit_3 + self.tau3 * torch.log(self.hat)
        
        lp_logit_1 = self.fuse_softmax(l_logit_1)
        lp_logit_2 = self.fuse_softmax(l_logit_2)
        lp_logit_3 = self.fuse_softmax(l_logit_3)

        min_l_logit_1 = torch.min(l_logit_1, dim=1, keepdim=True).values
        min_l_logit_2 = torch.min(l_logit_2, dim=1, keepdim=True).values
        min_l_logit_3 = torch.min(l_logit_3, dim=1, keepdim=True).values

        lnn_l_logit_1 = l_logit_1 - min_l_logit_1
        lnn_l_logit_2 = l_logit_2 - min_l_logit_2
        lnn_l_logit_3 = l_logit_3 - min_l_logit_3

        feat11 = self.lin1(feat1)
        feat22 = self.lin2(feat2 + feat11)
        feat33 = self.lin3(feat3 + feat22)
        feat44 = self.lin4(feat4 + feat33)
        
        feat111 = self.lin5(feat11)
        feat222 = self.lin6(feat22 + feat111)
        feat333 = self.lin7(feat33 + feat222)
        feat444 = self.lin8(feat44 + feat333)
        
        fuse_out = torch.cat([feat444, lp_logit_1, lp_logit_2, lp_logit_3], dim=1)
        out_for_attention = self.predict(fuse_out)

        w1, w2, w3 = self.fuse_softmax(out_for_attention).chunk(3, dim=1)
        p1, p2, p3 = torch.max(cp_logit_1[:,:self.cut1], dim=1, keepdim=True).values, torch.max(cp_logit_2[:,self.cut1:self.cut2], dim=1, keepdim=True).values, torch.max(cp_logit_3[:,self.cut2:], dim=1, keepdim=True).values
        new_w1, new_w2, new_w3 = self.fuse_softmax(torch.cat([p1 * w1, p2 * w2, p3 * w3], dim=1)).chunk(3, dim=1)

        results_dict['fuse_w_logit'] = out_for_attention
        results_dict['w1'], results_dict['w2'], results_dict['w3'] = new_w1, new_w2, new_w3
        results_dict['fuse_logit_l'] = results_dict['w1'] * lnn_l_logit_1 + results_dict['w2'] * lnn_l_logit_2 + results_dict['w3'] * lnn_l_logit_3

        results_dict['fuse_logit_HMT_c_1'] = results_dict['w1'] * lnn_c_logit_x_H_1 + results_dict['w2'] * lnn_c_logit_x_M_1 + results_dict['w3'] * lnn_c_logit_x_T_1
        results_dict['fuse_logit_HMT_c_2'] = results_dict['w1'] * lnn_c_logit_x_H_2 + results_dict['w2'] * lnn_c_logit_x_M_2 + results_dict['w3'] * lnn_c_logit_x_T_2
        results_dict['fuse_logit_HMT_c_3'] = results_dict['w1'] * lnn_c_logit_x_H_3 + results_dict['w2'] * lnn_c_logit_x_M_3 + results_dict['w3'] * lnn_c_logit_x_T_3

        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@IMB_ALGORITHMS.register('metaexpert')
class MetaExpert(ImbAlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(MetaExpert, self).__init__(args, net_builder, tb_log, logger)

        self.head = 3
        
        self.cut1 = args.cut1
        self.cut2 = args.cut2
        
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        head_lb_class_dist = [0 for _ in range(self.head)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
                if c < self.cut1:
                    head_lb_class_dist[0] += 1
                elif c < self.cut2:
                    head_lb_class_dist[1] += 1
                else:
                    head_lb_class_dist[2] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
                if c < self.cut1:
                    head_lb_class_dist[0] += 1
                elif c < self.cut2:
                    head_lb_class_dist[1] += 1
                else:
                    head_lb_class_dist[2] += 1
        lb_class_dist = np.array(lb_class_dist)
        head_lb_class_dist = np.array(head_lb_class_dist)
        
        self.p_hat_lb = torch.from_numpy((lb_class_dist / lb_class_dist.sum()).astype(np.float32)).cuda(args.gpu)
        self.head_p_hat_lb = torch.from_numpy((head_lb_class_dist / head_lb_class_dist.sum()).astype(np.float32)).cuda(args.gpu)
        
        self.class_dist_con = torch.from_numpy((lb_class_dist / lb_class_dist.sum()).astype(np.float32)).cuda(args.gpu)
        self.class_dist_uni = (torch.ones(self.num_classes) / self.num_classes).cuda(args.gpu)
        self.class_dist_rev = torch.flip(self.class_dist_con, dims=[0]).cuda(args.gpu)
        
        self.current_mask = torch.zeros(self.num_classes).cuda(self.args.gpu)
        self.est_class_dist = (torch.ones(self.num_classes) / self.num_classes).cuda(args.gpu)
        
        self.current_est_step = 0
        
        self.count_kl = torch.zeros(self.head).cuda(args.gpu)
        self.weight_kl = (torch.ones(self.head) / self.head).cuda(args.gpu)
        
        self.kl_div = nn.KLDivLoss(reduction='sum')
        
        self.weight1 = (1.0 - self.beta1) / (1.0 - np.power(self.beta1, lb_class_dist))
        self.weight1 = torch.from_numpy((self.weight1 / np.sum(self.weight1) * self.num_classes).astype(np.float32)).cuda(args.gpu)

        self.weight2 = (1.0 - self.beta2) / (1.0 - np.power(self.beta2, head_lb_class_dist))
        self.weight2 = torch.from_numpy((self.weight2 / np.sum(self.weight2) * self.head).astype(np.float32)).cuda(args.gpu)

        self.tau_lb1 = args.la_tau_lb1
        self.tau_lb2 = args.la_tau_lb2
        self.tau_lb3 = args.la_tau_lb3
        
        self.model = MetaExpertNet(self.model, num_classes=self.num_classes, p_hat_lb=self.p_hat_lb, tau_lb1=self.tau_lb1, tau_lb2=self.tau_lb2, tau_lb3=self.tau_lb3, cut1=self.cut1, cut2=self.cut2)
        self.ema_model = MetaExpertNet(self.ema_model, num_classes=self.num_classes, p_hat_lb=self.p_hat_lb, tau_lb1=self.tau_lb1, tau_lb2=self.tau_lb2, tau_lb3=self.tau_lb3, cut1=self.cut1, cut2=self.cut2)
        self.ema_model.load_state_dict(self.model.state_dict())

        self.est_epoch = args.est_epoch
        
        self.current_epoch = args.est_epoch
        
        self.est_step = args.num_eval_iter

        self.ema_u = args.ema_u

    def update_w(self, w1, w2, w3, lb):
        for i in range(len(lb)):
            self.record_w1[lb[i]].append(w1[i].item())
            self.record_w2[lb[i]].append(w2[i].item())
            self.record_w3[lb[i]].append(w3[i].item())

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        if self.epoch > self.est_epoch and self.epoch > self.current_epoch:
            self.current_epoch += 1
            if self.epoch == self.est_epoch + 2:
                if self.current_mask.sum() > 0:
                    self.current_mask = self.current_mask / self.current_mask.sum()
                    self.est_class_dist = self.current_mask
                    self.weight_kl[2] = (((torch.max(self.est_class_dist[int(self.num_classes / 2):]) / torch.min(self.est_class_dist[:int(self.num_classes / 2)])) > 2.0) & ((torch.max(self.est_class_dist[:int(self.num_classes / 2)]) / torch.min(self.est_class_dist[int(self.num_classes / 2):])) < 2.0)).float()
                    self.weight_kl[1] = ((torch.max(self.est_class_dist) / torch.min(self.est_class_dist)) < 2.0).float()
                    self.weight_kl[0] = 1.0 - (self.weight_kl[2] + self.weight_kl[1]).eq(1.0).float()
                    self.current_mask = torch.zeros(self.num_classes).cuda(self.args.gpu)

        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb
        num_lb = lb.shape[0]

        if self.it % self.num_eval_iter == 0:
            self.record_w1 = [[] for _ in range(self.num_classes)]
            self.record_w2 = [[] for _ in range(self.num_classes)]
            self.record_w3 = [[] for _ in range(self.num_classes)]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)

                logits_x_lb1 = outputs['logits'][:num_lb]
                logits_x_ulb_w1, logits_x_ulb_s1 = outputs['logits'][num_lb:].chunk(2)
                _, logits_x_ulb_sH1 = outputs['logitsH'][num_lb:].chunk(2)
                _, logits_x_ulb_sM1 = outputs['logitsM'][num_lb:].chunk(2)
                _, logits_x_ulb_sT1 = outputs['logitsT'][num_lb:].chunk(2)

                logits_x_lb2 = outputs['aux_logits1'][:num_lb]
                logits_x_ulb_w2, logits_x_ulb_s2 = outputs['aux_logits1'][num_lb:].chunk(2)
                _, logits_x_ulb_sH2 = outputs['aux_logitsH1'][num_lb:].chunk(2)
                _, logits_x_ulb_sM2 = outputs['aux_logitsM1'][num_lb:].chunk(2)
                _, logits_x_ulb_sT2 = outputs['aux_logitsT1'][num_lb:].chunk(2)

                logits_x_lb3 = outputs['aux_logits2'][:num_lb]
                logits_x_ulb_w3, logits_x_ulb_s3 = outputs['aux_logits2'][num_lb:].chunk(2)
                _, logits_x_ulb_sH3 = outputs['aux_logitsH2'][num_lb:].chunk(2)
                _, logits_x_ulb_sM3 = outputs['aux_logitsM2'][num_lb:].chunk(2)
                _, logits_x_ulb_sT3 = outputs['aux_logitsT2'][num_lb:].chunk(2)
                
                fuse_l_logits_x_lb = outputs['fuse_logit_l'][:num_lb]
                fuse_l_logits_x_ulb_w, fuse_l_logits_x_ulb_s = outputs['fuse_logit_l'][num_lb:].chunk(2)
                
                fuse_c_logits_HMT_x_lb_1 = outputs['fuse_logit_HMT_c_1'][:num_lb]
                fuse_c_logits_HMT_x_ulb_w_1, fuse_c_logits_HMT_x_ulb_s_1 = outputs['fuse_logit_HMT_c_1'][num_lb:].chunk(2)
                
                fuse_c_logits_HMT_x_lb_2 = outputs['fuse_logit_HMT_c_2'][:num_lb]
                fuse_c_logits_HMT_x_ulb_w_2, fuse_c_logits_HMT_x_ulb_s_2 = outputs['fuse_logit_HMT_c_2'][num_lb:].chunk(2)
                
                fuse_c_logits_HMT_x_lb_3 = outputs['fuse_logit_HMT_c_3'][:num_lb]
                fuse_c_logits_HMT_x_ulb_w_3, fuse_c_logits_HMT_x_ulb_s_3 = outputs['fuse_logit_HMT_c_3'][num_lb:].chunk(2)
                
                fuse_logits_w_lb = outputs['fuse_w_logit'][:num_lb]
                fuse_logits_w_ulb_w, fuse_logits_w_ulb_s = outputs['fuse_w_logit'][num_lb:].chunk(2)
            else:
                pass
            
            feat_dict = {}

            # First Head: FixMatch w/ tau1 * Logit Adjustment
            sup_loss1 = self.ce_loss(logits_x_lb1 + self.tau_lb1 * torch.log(self.p_hat_lb), lb, reduction='mean')
            probs_x_ulb_w1 = self.compute_prob(logits_x_ulb_w1.detach())
            mask1 = probs_x_ulb_w1.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label1 = probs_x_ulb_w1.argmax(dim=-1)
            pseudo_label1H = F.one_hot(pseudo_label1, self.num_classes).sum(dim=1) * mask1.float()
            pseudo_label1M = F.one_hot(pseudo_label1, self.num_classes)[:, self.cut1:].sum(dim=1) * mask1.float()
            pseudo_label1T = F.one_hot(pseudo_label1, self.num_classes)[:, self.cut2:].sum(dim=1) * mask1.float()
            unsup_loss1 = (self.ce_loss(logits_x_ulb_sH1, pseudo_label1, reduction='none') * pseudo_label1H).sum()
            unsup_loss1 += (self.ce_loss(logits_x_ulb_sM1, pseudo_label1, reduction='none') * pseudo_label1M).sum()
            unsup_loss1 += (self.ce_loss(logits_x_ulb_sT1, pseudo_label1, reduction='none') * pseudo_label1T).sum()
            unsup_loss1 /= (pseudo_label1H.sum() + pseudo_label1M.sum() + pseudo_label1T.sum() + 1e-12)

            # Second Head: FixMatch w/ tau2 * Logit Adjustment
            sup_loss2 = self.ce_loss(logits_x_lb2 + self.tau_lb2 * torch.log(self.p_hat_lb), lb, reduction='mean')
            probs_x_ulb_w2 = self.compute_prob(logits_x_ulb_w2.detach())
            mask2 = probs_x_ulb_w2.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label2 = probs_x_ulb_w2.argmax(dim=-1)
            pseudo_label2H = F.one_hot(pseudo_label2, self.num_classes).sum(dim=1) * mask2.float()
            pseudo_label2M = F.one_hot(pseudo_label2, self.num_classes)[:, self.cut1:].sum(dim=1) * mask2.float()
            pseudo_label2T = F.one_hot(pseudo_label2, self.num_classes)[:, self.cut2:].sum(dim=1) * mask2.float()
            unsup_loss2 = (self.ce_loss(logits_x_ulb_sH2, pseudo_label2, reduction='none') * pseudo_label2H).sum()
            unsup_loss2 += (self.ce_loss(logits_x_ulb_sM2, pseudo_label2, reduction='none') * pseudo_label2M).sum()
            unsup_loss2 += (self.ce_loss(logits_x_ulb_sT2, pseudo_label2, reduction='none') * pseudo_label2T).sum()
            unsup_loss2 /= (pseudo_label2H.sum() + pseudo_label2M.sum() + pseudo_label2T.sum() + 1e-12)

            # Third Head: FixMatch w/ tau3 * Logit Adjustment
            sup_loss3 = self.ce_loss(logits_x_lb3 + self.tau_lb3 * torch.log(self.p_hat_lb), lb, reduction='mean')
            probs_x_ulb_w3 = self.compute_prob(logits_x_ulb_w3.detach())
            mask3 = probs_x_ulb_w3.amax(dim=-1).ge(self.p_cutoff)
            pseudo_label3 = probs_x_ulb_w3.argmax(dim=-1)
            pseudo_label3H = F.one_hot(pseudo_label3, self.num_classes).sum(dim=1) * mask3.float()
            pseudo_label3M = F.one_hot(pseudo_label3, self.num_classes)[:, self.cut1:].sum(dim=1) * mask3.float()
            pseudo_label3T = F.one_hot(pseudo_label3, self.num_classes)[:, self.cut2:].sum(dim=1) * mask3.float()
            unsup_loss3 = (self.ce_loss(logits_x_ulb_sH3, pseudo_label3, reduction='none') * pseudo_label3H).sum()
            unsup_loss3 += (self.ce_loss(logits_x_ulb_sM3, pseudo_label3, reduction='none') * pseudo_label3M).sum()
            unsup_loss3 += (self.ce_loss(logits_x_ulb_sT3, pseudo_label3, reduction='none') * pseudo_label3T).sum()
            unsup_loss3 /= (pseudo_label3H.sum() + pseudo_label3M.sum() + pseudo_label3T.sum() + 1e-12)
            
            if self.epoch > self.est_epoch and self.epoch == self.current_epoch:
                self.current_mask[pseudo_label2] += mask2.float()

            # fuse loss
            sup_fuse_loss1 = F.cross_entropy(fuse_l_logits_x_lb, lb, self.weight1)
            fuse_probs_x_ulb_w_1 = self.compute_prob(fuse_c_logits_HMT_x_ulb_w_1.detach())
            fuse_mask11 = fuse_probs_x_ulb_w_1.amax(dim=-1).ge(self.p_cutoff).float()
            fuse_pseudo_label11 = fuse_probs_x_ulb_w_1.argmax(dim=-1)
            fuse_probs_x_ulb_w_2 = self.compute_prob(fuse_c_logits_HMT_x_ulb_w_2.detach())
            fuse_mask12 = fuse_probs_x_ulb_w_2.amax(dim=-1).ge(self.p_cutoff).float()
            fuse_pseudo_label12 = fuse_probs_x_ulb_w_2.argmax(dim=-1)
            fuse_probs_x_ulb_w_3 = self.compute_prob(fuse_c_logits_HMT_x_ulb_w_3.detach())
            fuse_mask13 = fuse_probs_x_ulb_w_3.amax(dim=-1).ge(self.p_cutoff).float()
            fuse_pseudo_label13 = fuse_probs_x_ulb_w_3.argmax(dim=-1)
            unsup_fuse_loss1 = self.weight_kl[0] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_1, fuse_pseudo_label11)
            unsup_fuse_loss1 += self.weight_kl[1] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_2, fuse_pseudo_label12)
            unsup_fuse_loss1 += self.weight_kl[2] * F.cross_entropy(fuse_c_logits_HMT_x_ulb_s_3, fuse_pseudo_label13)
            
            lb_w1 = torch.where(lb < self.cut1, torch.ones_like(lb), torch.zeros_like(lb))
            lb_w2 = torch.where((self.cut1 <= lb) & (lb < self.cut2), torch.ones_like(lb), torch.zeros_like(lb))
            lb_w3 = torch.where(self.cut2 <= lb, torch.ones_like(lb), torch.zeros_like(lb))
            lb_w = 0 * lb_w1 + 1 * lb_w2 + 2 * lb_w3
            sup_fuse_loss2 = F.cross_entropy(fuse_logits_w_lb, lb_w, self.weight2)
            fuse_probs_w_ulb_w = self.compute_prob(fuse_logits_w_ulb_w.detach())
            fuse_mask2 = fuse_probs_w_ulb_w.amax(dim=-1).ge(self.p_cutoff).float()
            fuse_pseudo_label2 = fuse_probs_w_ulb_w.argmax(dim=-1)
            unsup_fuse_loss2 = F.cross_entropy(fuse_logits_w_ulb_s, fuse_pseudo_label2)

            sup_fuse_loss = sup_fuse_loss1 + sup_fuse_loss2
            unsup_fuse_loss = unsup_fuse_loss1 + unsup_fuse_loss2

            # To compute frequencies, precision, and recalls
            self.update_w(outputs['w1'][:num_lb], outputs['w2'][:num_lb], outputs['w3'][:num_lb], lb)
            if self.epoch <= self.est_epoch+1:
                sup_fuse_loss = torch.tensor(0.0)
                unsup_fuse_loss = torch.tensor(0.0)
            sup_loss = sup_loss1 + sup_loss2 + sup_loss3 + sup_fuse_loss
            unsup_loss = self.lambda_u * unsup_loss1 + self.lambda_u * unsup_loss2 + self.lambda_u * unsup_loss3 + self.lambda_u * unsup_fuse_loss
            total_loss = sup_loss + unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss2.item(),
                                         unsup_loss=unsup_loss2.item(),
                                         total_loss=(sup_loss2 + self.lambda_u * unsup_loss2).item(),
                                         util_ratio=mask2.float().mean().item(),
                                         total_fuse_loss=(sup_fuse_loss + self.lambda_u * unsup_fuse_loss).item())

        if self.it % (self.num_eval_iter) == self.num_eval_iter - 1:
            log_dict['train/w1'] = [np.mean(item) if len(item) else np.nan for item in self.record_w1]
            log_dict['train/w2'] = [np.mean(item) if len(item) else np.nan for item in self.record_w2]
            log_dict['train/w3'] = [np.mean(item) if len(item) else np.nan for item in self.record_w3]

        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                w1 = self.model(x)['w1'].reshape(-1, 1)
                w2 = self.model(x)['w2'].reshape(-1, 1)
                w3 = self.model(x)['w3'].reshape(-1, 1)

                logit1 = self.model(x)['logits']
                logit2 = self.model(x)['aux_logits1']
                logit3 = self.model(x)['aux_logits2']

                w = torch.cat([w1, w2, w3], dim=1)

                one_hot_w = torch.zeros_like(w)
                argmax_indices = torch.max(w, dim=1)[1]
                one_hot_w.scatter_(1, argmax_indices.unsqueeze(1), 1)
                
                one_hot_w1 = torch.zeros_like(w)
                _, top_two = torch.topk(w, 2, dim=1)
                one_hot_w1.scatter_(1, top_two, 1)

                logits = w[:, 0].reshape(-1, 1) * logit1 + w[:, 1].reshape(-1, 1) * logit2 + w[:, 2].reshape(-1, 1) * logit3

                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1,
                     eval_dest + '/top-5-acc': top5, eval_dest + '/balanced_acc': balanced_top1,
                     eval_dest + '/precision': precision, eval_dest + '/recall': recall, eval_dest + '/F1': F1}
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--la_tau_lb1', float, 0.0),
            SSL_Argument('--la_tau_lb2', float, 2.0),
            SSL_Argument('--la_tau_lb3', float, 4.0),
            SSL_Argument('--est_epoch', int, 0),
            SSL_Argument('--ema_u', float, 0.9),
            SSL_Argument('--cut1', float, 2),
            SSL_Argument('--cut2', float, 4),
            SSL_Argument('--beta1', float, 0.99),
            SSL_Argument('--beta2', float, 0.99),
        ]