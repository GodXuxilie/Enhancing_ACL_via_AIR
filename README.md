
# Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization
---
This repository provides codes for NeurIPS 2023 paper: **Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization** (https://arxiv.org/pdf/2305.00374.pdf)
<br> Xilie Xu* (NUS), Jingfeng Zhang* (RIKEN-AIP/University of Auckland), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS).

## Environment
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6


## Script
We provide the detailed script for pre-training and fine-tuning in [```run.sh```](./run.sh).

### Pre-Training
```
python pretraining.py exp_dir --ACL_DS --DynAug --dataset pretraining_dataset
```

### Pre-Training with Post-Processing
```
python LP-AFF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task
```

### Fine-Tuning
```
### SLF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task --cvt_state_dict --bnNameCnt 1 --evaluation_mode SLF
### ALF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task --cvt_state_dict --bnNameCnt 1 --evaluation_mode ALF
### AFF ###
python test_AFF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task
```

### Pre-Trained Checkpoints
We provide the pre-trained checkpoints for DynACL-AIR and DynACL-AIR++ on CIFAR-10/CIFAR-100/STL-10 in [Model Zoo](https://github.com/GodXuxilie/RobustSSL_Benchmark) of [RobustSSL Benchmark](https://robustssl.github.io).

## BibTeX
```
@inproceedings{xu2023AIR,
  title={Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Contact
Please drop an e-mail to xuxilie@comp.nus.edu.sg and jingfeng.zhang@auckland.ac.nz if you have any issue.
