### pre-training ###

python pretraining.py DynACL_IR_cifar10 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_IR_cifar100 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_IR_stl10 --dataset stl10 --ACL_DS --DynAug


### finetuning procedures ###

### SLF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task --cvt_state_dict --bnNameCnt 1 --evaluation_mode SLF
### ALF ###
python test_LF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task --cvt_state_dict --bnNameCnt 1 --evaluation_mode ALF
### AFF ###
python test_AFF.py --experiment exp_name --gpu 0 --checkpoint path_of_pre-trained_model --dataset downstream_task


