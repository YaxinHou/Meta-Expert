# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import numpy as np

from semilearn.core import ImbAlgorithmBase
from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import get_data_loader, IMB_ALGORITHMS
from .utils import AdaptiveThresholdingHook
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

@IMB_ALGORITHMS.register('adsh')
class Adsh(ImbAlgorithmBase):
    def __init__(self, args, **kwargs):
        self.imb_init(tau_1=args.adsh_tau_1)
        super().__init__(args, **kwargs)
        assert args.algorithm == 'fixmatch', "Adsh only supports FixMatch as the base algorithm."

    def imb_init(self, tau_1):
        self.tau_1 = tau_1

    def set_dataset(self):
        dataset_dict, lb_count_message = super().set_dataset()
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False
        return dataset_dict, lb_count_message

    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        
        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    def set_hooks(self):
        super().set_hooks()
        # reset hooks
        self.register_hook(AdaptiveThresholdingHook(self.num_classes, self.tau_1), "MaskingHook")

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

                logits = self.model(x)[out_key]

                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        
        if eval_dest == 'eval':
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
        eval_dict = {}
        if eval_dest == 'eval':
            eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1,
                         eval_dest + '/top-5-acc': top5, eval_dest + '/balanced_acc': balanced_top1,
                         eval_dest + '/precision': precision, eval_dest + '/recall': recall, eval_dest + '/F1': F1}
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict
        
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--adsh_tau_1', float, 0.95),
        ]
