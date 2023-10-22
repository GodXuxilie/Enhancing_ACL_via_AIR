### pre-training ###

python pretraining.py DynACL_AIR_cifar10 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_cifar100 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_stl10 --dataset stl10 --ACL_DS --DynAug


### finetuning pre-trained models###

python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_cifar10/model.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar100_r18_cifar10 --dataset cifar100 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_cifar100/model.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_stl10_r18_stl10 --resize 96 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_stl10/model.pt --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_cifar10/model.pt --mode ALL --eval-AA 
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_cifar100/model.pt --mode ALL --eval-AA

### pre-training with post-processing ###

python LP-AFF.py --experiment DynACL_AIR++_cifar10 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_cifar10/model.pt --dataset cifar10
python LP-AFF.py --experiment DynACL_AIR++_cifar100 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_cifar100/model.pt --dataset cifar100
python LP-AFF.py --experiment DynACL_AIR++_stl10 --gpu 0 --checkpoint ./checkpoints/DynACL_AIR_stl10/model.pt --dataset stl10

### finetuning pre-trained models with post-processing ###

python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_cifar10/model.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar100_r18_cifar10 --dataset cifar100 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_cifar100/model.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_stl10_r18_stl10 --resize 96 --dataset stl10 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_stl10/model.pt --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR++_cifar10/model.pt --mode ALL --eval-AA 
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR++_cifar100/model.pt --mode ALL --eval-AA

