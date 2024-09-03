# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20003 ./train.py --model=transbts --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=8

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=20002 ./train.py --model=unet3d --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=8

# torchrun --nproc_per_node=8 --master_port=20004 ./train.py --model=vit2d --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=1720

# torchrun --nproc_per_node=8 --master_port=20005 ./train.py --model=unet2d --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=1240

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20003 ./train.py --model=swinunetr --start_epoch=1 --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.001 --seed=1000 --batch_size=8