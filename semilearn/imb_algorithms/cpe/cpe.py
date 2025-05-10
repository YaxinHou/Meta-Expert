# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from sklearn.metrics import precision_score, recall_score
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

@IMB_ALGORITHMS.register('cpe')
class CPE(ImbAlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(CPE, self).__init__(args, net_builder, tb_log, logger)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.p_hat_lb = (torch.from_numpy(lb_class_dist / lb_class_dist.sum())).cuda(args.gpu)

        self.tau_lb1 = args.la_tau_lb1
        self.tau_lb2 = args.la_tau_lb2
        self.tau_lb3 = args.la_tau_lb3

        self.est_epoch = args.est_epoch

        self.ema_u = args.ema_u

        self.cut1 = args.cut1
        self.cut2 = args.cut2

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

                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                pass

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

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

            sup_loss = sup_loss1 + sup_loss2 + sup_loss3
            unsup_loss = self.lambda_u * unsup_loss1 + self.lambda_u * unsup_loss2 + self.lambda_u * unsup_loss3
            total_loss = sup_loss + unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss2.item(),
                                         unsup_loss=unsup_loss2.item(),
                                         total_loss=(sup_loss2 + self.lambda_u * unsup_loss2).item(),
                                         util_ratio=mask2.float().mean().item())

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

                logits = self.model(x)['aux_logits1']

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
        ]