
# Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization

This repository provides codes for NeurIPS 2023 paper: **Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization** (https://arxiv.org/pdf/2305.00374.pdf)
<br> Xilie Xu* (NUS), Jingfeng Zhang* (RIKEN-AIP/University of Auckland), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS).

## Environment
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6


## Script & Pre-Trained Checkpoints
We provide the detailed script for pre-training and fine-tuning in [```run.sh```](./run.sh).

The pre-trained checkpoints for DynACL-AIR and DynACL-AIR++ on CIFAR-10/CIFAR-100/STL-10 using ResNet-18 are in [Model Zoo](https://github.com/GodXuxilie/RobustSSL_Benchmark) of [RobustSSL Benchmark](https://robustssl.github.io).

#### Pre-Training (DynACL-AIR)
```
python pretraining.py DynACL_AIR_cifar10 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_cifar100 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_stl10 --dataset stl10 --ACL_DS --DynAug
```
#### Pre-Training with Post-Processing (DynACL-AIR++)
```
python LP-AFF.py --experiment DynACL_AIR++_cifar10 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_cifar10/model.pt --dataset cifar10
python LP-AFF.py --experiment DynACL_AIR++_cifar100 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_cifar100/model.pt --dataset cifar100
python LP-AFF.py --experiment DynACL_AIR++_stl10 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_stl10/model.pt --dataset stl10
```
#### Fine-Tuning
Detailed fine-tuning scripts are in [`run.sh`](./run.sh).
```
### fine-tuning pre-trained models ###
python finetuning.py --gpu 2 --experiment DynACL_AIR_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL-AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_cifar10/model.pt --mode ALL --eval-AA --eval-OOD

### fine-tuning pre-trained models with post-processing ###
python finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL-AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_cifar10/model.pt --mode ALL --eval-AA --eval-OOD
```

It is recommended to use the code of [AutoLoRa](https://github.com/GodXuxilie/RobustSSL_Benchmark/tree/main/Finetuning_Methods/AutoLoRa) to further justify the effectiveness of your pre-training method. AutoLoRa is an automated robust fine-tuning framework, which exempts the burden of searching for appropriate hyper-parameters and can further unleash the power of pre-trained models in downstream tasks.

## BibTeX
```
@inproceedings{xu2023AIR,
  title={Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  booktitle={NeurIPS},
  year={2023}
}
```

## Contact
Please drop an e-mail to xuxilie@comp.nus.edu.sg and jingfeng.zhang@auckland.ac.nz if you have any issue.
