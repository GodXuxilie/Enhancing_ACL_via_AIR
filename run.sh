### pre-training ###

### ACL ###
python pretraining.py ACL_cifar10 --lambda1 0.0 --lambda2 0.0 --dataset cifar10 --ACL_DS
python pretraining.py ACL_cifar100 --lambda1 0.0 --lambda2 0.0 --dataset cifar100 --ACL_DS
python pretraining.py ACL_stl10 --lambda1 0.0 --lambda2 0.0 --dataset stl10 --ACL_DS

### ACL with SIR ###
python pretraining.py ACL_SIR_cifar10 --lambda1 0.5 --lambda2 0.0 --dataset cifar10 --ACL_DS
python pretraining.py ACL_SIR_cifar100 --lambda1 0.5 --lambda2 0.0 --dataset cifar100 --ACL_DS
python pretraining.py ACL_SIR_stl10 --lambda1 0.5 --lambda2 0.0 --dataset stl10 --ACL_DS

### ACL with AIR ###
python pretraining.py ACL_AIR_cifar10 --lambda1 0.0 --lambda2 0.5 --dataset cifar10 --ACL_DS
python pretraining.py ACL_AIR_cifar100 --lambda1 0.0 --lambda2 0.5 --dataset cifar100 --ACL_DS
python pretraining.py ACL_AIR_stl10 --lambda1 0.0 --lambda2 0.5 --dataset stl10 --ACL_DS

### ACL with IR ###
python pretraining.py ACL_IR_cifar10 --lambda1 0.5 --lambda2 0.5 --dataset cifar10 --ACL_DS
python pretraining.py ACL_IR_cifar100 --lambda1 0.5 --lambda2 0.5 --dataset cifar100 --ACL_DS
python pretraining.py ACL_IR_stl10 --lambda1 0.5 --lambda2 0.5 --dataset stl10 --ACL_DS

### DynACL ###
python pretraining.py DynACL_cifar10 --lambda1 0.0 --lambda2 0.0 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_cifar100 --lambda1 0.0 --lambda2 0.0 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_stl10 --lambda1 0.0 --lambda2 0.0 --dataset stl10 --ACL_DS --DynAug

### ACL with SIR ###
python pretraining.py DynACL_SIR_cifar10 --lambda1 0.5 --lambda2 0.0 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_SIR_cifar100 --lambda1 0.5 --lambda2 0.0 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_SIR_stl10 --lambda1 0.5 --lambda2 0.0 --dataset stl10 --ACL_DS --DynAug

### ACL with AIR ###
python pretraining.py DynACL_AIR_cifar10 --lambda1 0.0 --lambda2 0.5 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_cifar100 --lambda1 0.0 --lambda2 0.5 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_AIR_stl10 --lambda1 0.0 --lambda2 0.5 --dataset stl10 --ACL_DS --DynAug

### ACL with IR ###
python pretraining.py DynACL_IR_cifar10 --lambda1 0.5 --lambda2 0.5 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_IR_cifar100 --lambda1 0.5 --lambda2 0.5 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_IR_stl10 --lambda1 0.5 --lambda2 0.5 --dataset stl10 --ACL_DS --DynAug


### finetuning procedures ###

### SLF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset cifar10 --cvt_state_dict --bnNameCnt 1 --evaluation_mode SLF
### ALF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset cifar10 --cvt_state_dict --bnNameCnt 1 --evaluation_mode ALF
### AFF ###
python test_AFF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset cifar10


