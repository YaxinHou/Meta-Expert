# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .utils import DebiasPLConsistencyLoss, DebiasPLHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@IMB_ALGORITHMS.register('debiaspl')
class DebiasPL(ImbAlgorithmBase):
    def __init__(self, args, **kwargs):
        self.imb_init(args.debiaspl_tau, args.debiaspl_ema_p)
        super().__init__(args, **kwargs)
        assert args.algorithm not in ['mixmatch', 'meanteacher', 'pimodel'], "DebiasPL not supports {} as the base algorithm.".format(args.algorithm)

        self.p_hat = torch.ones((self.num_classes, )).to(self.gpu) / self.num_classes
        self.consistency_loss = DebiasPLConsistencyLoss(tau=self.tau)

    def imb_init(self, tau=0.4, ema_p=0.999):
        self.tau = tau 
        self.ema_p = ema_p

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(DebiasPLHook(), "NORMAL")

    def compute_prob(self, logits):
        # update p_hat
        probs = super().compute_prob(logits)
        delta_p = probs.mean(dim=0)
        self.p_hat = self.ema_m * self.p_hat + (1 - self.ema_p) * delta_p
        return super().compute_prob(logits - self.tau * torch.log(self.p_hat))

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s):
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
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
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

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--debiaspl_tau', float, 0.4),
            SSL_Argument('--debiaspl_ema_p', float, 0.999),
        ]
