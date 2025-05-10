# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
from io import BytesIO
import torch.distributed as dist
from torch.utils.data import sampler, DataLoader

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_labeled_unlabeled_data(args, data, targets, num_classes, lb_num_labels, ulb_num_labels=None,
                                 lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0, noise_ratio=0.1,
                                 noise_per_class=False, lb_imb_type='exp', ulb_imb_type='exp', num_steps=5,
                                 lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=False):
    """
    data & target is splitted into labeled and unlabeled data.

    Args
        data: data to be split to labeled and unlabeled
        targets: targets to be split to labeled and unlabeled
        num_classes: number of total classes
        lb_num_labels: number of labeled samples.
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        ulb_num_labels: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes,
                                                                                lb_num_labels, ulb_num_labels,
                                                                                lb_imbalance_ratio, ulb_imbalance_ratio,
                                                                                noise_ratio, noise_per_class,
                                                                                lb_imb_type, ulb_imb_type, num_steps,
                                                                                load_exist)

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)

    return lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx


def sample_labeled_unlabeled_data(args, data, target, num_classes, lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0, noise_ratio=0.1,
                                  noise_per_class=True, lb_imb_type='exp', ulb_imb_type='exp', num_steps=5,
                                  load_exist=True):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join('./data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir,
                                f'lb_labels_{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir,
                                 f'ulb_labels_{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy')
    lb_clean_dump_path = os.path.join(dump_dir,
                                      f'lb_clean_labels_{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy')
    lb_noise_dump_path = os.path.join(dump_dir,
                                      f'lb_noise_labels_{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy')

    lb_idx = []
    lb_clean_idx = []
    lb_noise_idx = []
    ulb_idx = []
    flag = os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and os.path.exists(
        lb_clean_dump_path) and os.path.exists(lb_noise_dump_path)
    if flag and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        lb_clean_idx = np.load(lb_clean_dump_path)
        lb_noise_idx = np.load(lb_noise_dump_path)
        return lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx

    if -1 in target:
        # get samples per class
        if lb_imbalance_ratio == 1.0:
            # balanced setting, lb_num_labels is total number of labels for labeled data
            assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
            lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
        else:
            # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
            lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio, lb_imb_type, num_steps)
        
        if noise_per_class:
            lb_noise_samples_per_class = [int(noise_ratio * item) for item in lb_samples_per_class]
        
        for c in range(-1, num_classes):
            if c != -1:
                idx = np.where(target == c)[0]
                np.random.shuffle(idx)
                all_lab_idx = idx[:lb_samples_per_class[c]]
                lb_idx.extend(all_lab_idx)
                np.random.shuffle(all_lab_idx)
                if noise_per_class:
                    lb_noise_idx.extend(all_lab_idx[:lb_noise_samples_per_class[c]])
                    lb_clean_idx.extend(all_lab_idx[lb_noise_samples_per_class[c]:])
            else:
                idx = np.where(target == c)[0]
                np.random.shuffle(idx)
                ulb_idx.extend(idx)
        
        if not noise_per_class:
            np.random.shuffle(lb_idx)
            num_of_noise_labels = int(noise_ratio * len(lb_idx))

            # r = 1 - sum([(item / sum(lb_samples_per_class)) ** 2 for item in lb_samples_per_class])
            # num_of_noise_labels = int(noise_ratio / r * sum(lb_samples_per_class))

            lb_noise_idx = lb_idx[:num_of_noise_labels]
            lb_clean_idx = lb_idx[num_of_noise_labels:]
    else:
        # get samples per class
        if lb_imbalance_ratio == 1.0:
            # balanced setting, lb_num_labels is total number of labels for labeled data
            assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
            lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
        else:
            # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
            lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio, lb_imb_type, num_steps)

        if ulb_imbalance_ratio == 1.0:
            # balanced setting
            if ulb_num_labels is None or ulb_num_labels == 'None':
                ulb_samples_per_class = [int(len(data) / num_classes) - lb_samples_per_class[c] for c in range(num_classes)]
                # [int(len(data) / num_classes) - int(lb_num_labels / num_classes)] * num_classes
            else:
                assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
                ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
        else:
            # imbalanced setting
            assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
            ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio, ulb_imb_type, num_steps)

        if noise_per_class:
            lb_noise_samples_per_class = [int(noise_ratio * item) for item in lb_samples_per_class]

        for c in range(num_classes):
            idx = np.where(target == c)[0]
            np.random.shuffle(idx)
            all_lab_idx = idx[:lb_samples_per_class[c]]
            lb_idx.extend(all_lab_idx)
            np.random.shuffle(all_lab_idx)
            if noise_per_class:
                lb_noise_idx.extend(all_lab_idx[:lb_noise_samples_per_class[c]])
                lb_clean_idx.extend(all_lab_idx[lb_noise_samples_per_class[c]:])

            if ulb_num_labels is None or ulb_num_labels == 'None':
                ulb_idx.extend(idx[lb_samples_per_class[c]:])
            else:
                ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c] + ulb_samples_per_class[c]])

        if not noise_per_class:
            np.random.shuffle(lb_idx)
            num_of_noise_labels = int(noise_ratio * len(lb_idx))

            # r = 1 - sum([(item / sum(lb_samples_per_class)) ** 2 for item in lb_samples_per_class])
            # num_of_noise_labels = int(noise_ratio / r * sum(lb_samples_per_class))

            lb_noise_idx = lb_idx[:num_of_noise_labels]
            lb_clean_idx = lb_idx[num_of_noise_labels:]

    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)

    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    if isinstance(lb_clean_idx, list):
        lb_clean_idx = np.asarray(lb_clean_idx)

    if isinstance(lb_noise_idx, list):
        lb_noise_idx = np.asarray(lb_noise_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)

    np.save(lb_clean_dump_path, lb_clean_idx)
    np.save(lb_noise_dump_path, lb_noise_idx)

    return lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx


def make_imbalance_data(max_num_labels, num_classes, gamma, imb_type, num_steps):
    """
    calculate samplers per class for imbalanced data
    """
    if imb_type == 'exp':
        samples_per_class = []
        for c in range(num_classes):
            mu = np.power(gamma, - c / (num_classes - 1))
            samples_per_class.append(int(max_num_labels * mu))
    elif imb_type == 'step':
        min_num_labels = max_num_labels / gamma
        classes_per_shot = int(num_classes / num_steps)
        samples_per_class = []
        for shot_idx in range(num_steps):
            current_step_num_labels = int(max_num_labels - (max_num_labels - min_num_labels) * shot_idx / num_steps)
            samples_per_class.extend([current_step_num_labels] * classes_per_shot)
    elif imb_type == 'pxe':
        samples_per_class = []
        for c in range(num_classes):
            mu = np.power(abs(gamma), - c / (num_classes - 1))
            samples_per_class.append(int(max_num_labels * mu))

        samples_per_class = samples_per_class[::-1]
    elif imb_type == 'pets':
        min_num_labels = max_num_labels / abs(gamma)
        classes_per_shot = int(num_classes / num_steps)
        samples_per_class = []
        for shot_idx in range(num_steps):
            current_step_num_labels = int(max_num_labels - (max_num_labels - min_num_labels) * shot_idx / num_steps)
            samples_per_class.extend([current_step_num_labels] * classes_per_shot)
        samples_per_class = samples_per_class[::-1]

    return samples_per_class


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset: random_offset + sample_length]
