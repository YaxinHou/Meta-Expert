# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from semilearn.datasets.utils import get_onehot
from semilearn.datasets.augmentation import RandAugment


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self, data, targets=None, noised_targets=None, num_classes=None, is_ulb=False, weak_transform=None,
                 medium_transform=None, strong_transform=None, onehot=False, *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.noised_targets = noised_targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        # self.medium_transform = medium_transform

    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        if self.noised_targets is None:
            noised_target = target
        else:
            noised_target_ = self.noised_targets[idx]
            noised_target = noised_target_ if not self.onehot else get_onehot(self.num_classes, noised_target_)

        # set augmented images
        img = self.data[idx]
        return img, target, noised_target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target, noised_target = self.__sample__(idx)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not self.is_ulb:
            return {'idx_lb': idx, 'x_lb': self.weak_transform(img), 'y_lb': target, 'y_lb_noised': noised_target}
        else:
            # if self.alg == 'fullysupervised':
            #     return {'idx_ulb': idx}
            return {'idx_ulb': idx, 'x_ulb_w': self.weak_transform(img), 'x_ulb_s': self.strong_transform(img),
                    'y_ulb': target}

    def __len__(self):
        return len(self.data)
