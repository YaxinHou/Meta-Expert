# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment


mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]

def get_stl10(args, name, data_dir='./data', include_lb_to_ulb=False):
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    
    train_dset_lb = dset(data_dir, split='train', download=True)
    train_dset_ulb = dset(data_dir, split='unlabeled', download=True)
    test_dset = dset(data_dir, split='test', download=True)
    
    lb_data, lb_targets = train_dset_lb.data.transpose([0, 2, 3, 1]), train_dset_lb.labels.astype(np.int64)
    ulb_data = train_dset_ulb.data.transpose([0, 2, 3, 1])
    ulb_targets = (np.ones((ulb_data.shape[0], )) * -1).astype(np.int64)
    test_data, test_targets = test_dset.data.transpose([0, 2, 3, 1]), test_dset.labels.astype(np.int64)
    
    train_data, train_targets = np.concatenate([lb_data, ulb_data], axis=0), np.concatenate([lb_targets, ulb_targets], axis=0)

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(args, train_data, train_targets,
                                                                               num_classes=args.num_classes,
                                                                               lb_num_labels=args.num_labels,
                                                                               ulb_num_labels=args.ulb_num_labels,
                                                                               lb_imbalance_ratio=args.lb_imb_ratio,
                                                                               ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                               noise_ratio=args.noise_ratio,
                                                                               noise_per_class=args.noise_per_class,
                                                                               lb_imb_type=args.lb_imb_type,
                                                                               ulb_imb_type=args.ulb_imb_type,
                                                                               num_steps=args.num_steps,
                                                                               include_lb_to_ulb=include_lb_to_ulb)

    data, targets, noised_targets = np.array(train_data), np.array(train_targets), np.array(train_targets)
    lb_count = [0 for _ in range(args.num_classes)]
    lb_clean_count = [0 for _ in range(args.num_classes)]
    lb_noise_count = [0 for _ in range(args.num_classes)]
    new_lb_noise_count = [0 for _ in range(args.num_classes)]
    ulb_count = [0]

    for c in targets[lb_idx]:
        lb_count[c] += 1
    for c in targets[ulb_idx]:
        ulb_count[0] += 1
    for c in targets[lb_clean_idx]:
        lb_clean_count[c] += 1
    if len(lb_noise_idx) > 0:
        for c in targets[lb_noise_idx]:
            lb_noise_count[c] += 1

    p_noise = np.zeros((args.num_classes, args.num_classes))
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            if i != j:
                p_noise[i][j] = lb_count[j] / (sum(lb_count) - lb_count[i])
    p_noise = p_noise / p_noise.sum(axis=1, keepdims=True)

    for i in lb_noise_idx:
        if args.noise_type == 'sym':
            noised_targets[i] = (random.randint(1, args.num_classes - 1) + targets[i]) % args.num_classes
        elif args.noise_type == 'asym':
            noised_targets[i] = np.random.choice(args.num_classes, p=p_noise[targets[i]])
        elif args.noise_type == 'circle':
            noised_targets[i] = (targets[i] + 1) % args.num_classes
    if len(lb_noise_idx) > 0:
        for c in noised_targets[lb_noise_idx]:
            new_lb_noise_count[c] += 1

    print("lb count: {}".format(lb_count))
    print("lb clean count: {}".format(lb_clean_count))
    print("lb noise count: {}".format(lb_noise_count))
    print("new lb noise count: {}".format(new_lb_noise_count))
    print("ulb count: {}".format(ulb_count))

    lb_dset = BasicDataset(data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(data[ulb_idx], targets[ulb_idx], None, args.num_classes, True,
                            weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(test_data, test_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)

    lb_count_message = {'lb_count': lb_count, 'ulb_count': ulb_count, 'lb_clean_count': lb_clean_count,
                        'lb_noise_count': lb_noise_count, 'new_lb_noise_count': new_lb_noise_count}

    return lb_dset, ulb_dset, eval_dset, lb_count_message
