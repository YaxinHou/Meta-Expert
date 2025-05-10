# A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning
Yaxin Hou, Yuheng Jia, A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning, International Conference on Machine Learning, 13th-19th July, Vancouver, 2025.
This is an official [PyTorch](http://pytorch.org) implementation for **A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning**.

## Introduction
This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning) and the previous method [CPE](https://github.com/machengcheng2016/CPE-LTSSL).

What I've done is just adding our Meta-Expert algorithm in `semilearn/imb_algorithms/metaexpert`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run
For example, on CIFAR-10-LT with $\gamma_l=\gamma_u=150$

```
CUDA_VISIBLE_DEVICES=0 python train.py --c config/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2.yaml
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `saved_models/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2/log.txt` will tell you the best accuracy. 

For example,
```
[2024-07-26 03:54:08,086 INFO] model saved: ./saved_models/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2/latest_model.pth
[2024-07-26 03:54:08,089 INFO] Model result - eval/best_acc : 0.8248
[2024-07-26 03:54:08,090 INFO] Model result - eval/best_it : 179199
```

## Results

The reported accuracies in Table 3 and 4 in our paper are the average over three different runs (random seeds are 0/2/4). 

## Citation

If you find our method useful, please consider citing our paper:

  ```
  @inproceedings{metaexperticml2025,
    title={A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning},
    author={Jia, Yuheng and Hou, Yaxin},
    booktitle={International Conference on Machine Learning},
    volume={},
    number={},
    pages={},
    year={2025}
  }
  ```
