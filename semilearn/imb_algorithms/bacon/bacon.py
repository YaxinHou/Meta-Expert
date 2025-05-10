# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('bacon')
class BaCon(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(BaCon, self).__init__(args, net_builder, tb_log, logger)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)

        self.abc_p_cutoff = args.abc_p_cutoff
        self.abc_loss_ratio = args.abc_loss_ratio

        self.lb_dest_len = args.lb_dest_len
        self.ulb_dest_len = args.ulb_dest_len

        self.selected_label = (torch.ones((self.lb_dest_len + self.ulb_dest_len,), dtype=torch.long) * -1).cuda(args.gpu)
        self.cls_freq = (torch.ones((self.num_classes,))).cuda(args.gpu)
        self.feat_list = (torch.ones((self.lb_dest_len + self.ulb_dest_len, 32))).cuda(args.gpu)
        self.class_feat_center = (torch.ones((self.num_classes, 32))).cuda(args.gpu)


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, idx_lb, idx_ulb, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
                abc_logits_x_lb = outputs['aux_logits'][:num_lb]
                abc_logits_x_ulb_w, abc_logits_x_ulb_s = outputs['aux_logits'][num_lb:].chunk(2)
                proj_lb = outputs['pro_out'][:num_lb]
                proj_ulb_w, proj_ulb_s = outputs['pro_out'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                abc_logits_x_lb = outs_x_lb['aux_logits']
                proj_lb = outs_x_lb['pro_out']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                abc_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                proj_ulb_s = outs_x_ulb_s['pro_out']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
                    abc_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    proj_ulb_w = outs_x_ulb_w['pro_out']

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label, T=self.T, softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            abc_loss, mask_ulb = self.compute_abc_loss(logits_x_lb=abc_logits_x_lb, y_lb=lb,
                                                       logits_x_ulb_w=abc_logits_x_ulb_w,
                                                       logits_x_ulb_s=abc_logits_x_ulb_s)

            # update class count
            abc_max_probs, abc_max_idx = torch.max(abc_logits_x_ulb_w.softmax(-1), dim=-1)
            select = abc_max_probs.ge(0.95)
            if idx_ulb[select == 1].nelement() != 0:
                self.selected_label[self.lb_dest_len + idx_ulb[select == 1]] = abc_max_idx[select == 1]
                self.selected_label[idx_lb] = lb
            else:
                self.selected_label[idx_lb] = lb
            for i in range(self.num_classes):
                self.cls_freq[i] = torch.sum(self.selected_label == i)

            select_lb = (torch.max(abc_logits_x_lb.softmax(-1), dim=-1)[0]).ge(0.98)
            select_ulb = (torch.max(abc_logits_x_ulb_w.softmax(-1), dim=-1)[0]).ge(0.98)
            select_all = torch.cat((select_lb, select_ulb), dim=0)

            contra_loss = torch.tensor(0)
            if self.it > 262144 / 3:
                contra_loss = self.contrastive_loss(anchors=self.class_feat_center,
                                                    feats=torch.cat((proj_lb, proj_ulb_w), dim=0), y_lb=lb,
                                                    top_ulb=abc_logits_x_ulb_w.topk(3, dim=-1)[1], select=select_all)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.abc_loss_ratio * abc_loss + contra_loss

        # update feature space
        self.feat_list[idx_lb[select_lb == 1]] = proj_lb[select_lb == 1].clone().detach()
        self.feat_list[(idx_ulb + self.lb_dest_len)[select_ulb == 1]] = proj_ulb_w[select_ulb == 1].clone().detach()
        for i in range(self.num_classes):
            self.class_feat_center[i] = torch.mean(self.feat_list[self.selected_label == i], 0)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         abc_loss=abc_loss.item(),
                                         contra_loss=contra_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask_ulb.float().mean().item(),
                                         select_for_contra=select_all.sum().item())

        return out_dict, log_dict

    def contrastive_loss(self, anchors, feats, y_lb, top_ulb, select):
        contra_loss = 0
        y = torch.cat((y_lb, top_ulb[:, 0]), dim=0)
        for i in range(self.num_classes):
            to = 0.1
            et = 0.005
            to_c = to * (1 - (1 - self.it/262144)**2 * et * torch.sqrt(self.cls_freq[i]/torch.max(self.cls_freq)))
            temp = top_ulb - i
            idx = torch.nonzero(temp == 0)[:, 0]
            neg_idx = torch.ones((top_ulb.shape[0],)).to(y_lb.device)
            neg_idx[idx] = 0
            neg_idx = torch.cat((y_lb[:] != i, neg_idx), dim=0).to(torch.long)
            neg_samples = feats[neg_idx == 1]
            cosine_similarity_1 = torch.cosine_similarity(feats[y == i], anchors[y][y == i], dim=-1)
            cosine_similarity_2 = torch.cosine_similarity(feats[y == i].unsqueeze(1).repeat(1, neg_samples.shape[0], 1),
                                                          neg_samples.unsqueeze(0).repeat(feats[y == i].shape[0], 1, 1),
                                                          dim=-1)
            pos = torch.exp(cosine_similarity_1 / to_c)
            neg = torch.exp(cosine_similarity_2 / to)
            loss = pos / (pos + 64 * neg.mean() + 1e-8)
            contra_loss += (-1 * torch.log(loss) * select[y == i]).sum()
        return contra_loss / (select.sum() + 1e-8)

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

    def compute_abc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]

        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        abc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(self.abc_p_cutoff).to(logits_x_ulb_w.dtype)
            ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)
            mask_ulb_2 = self.bernouli_mask(ulb_class_dist[y_ulb])
            mask_ulb = mask_ulb_1 * mask_ulb_2

        abc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            abc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()

        abc_loss = abc_lb_loss + abc_ulb_loss
        return abc_loss, mask_ulb

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='aux_logits', return_logits=return_logits)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--abc_p_cutoff', float, 0.95),
            SSL_Argument('--abc_loss_ratio', float, 1.0),
        ]
