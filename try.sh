# MASTER_PORT=29498 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./train.py --model=transbts --start_epoch=1 --end_epoch=5 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.0004 --seed=1000 --batch_size=8

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./train.py --model=unet3d --start_epoch=1 --end_epoch=5 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.0004 --seed=1000 --batch_size=8


# torchrun --nproc_per_node=8 ./train.py --model=vit2d --start_epoch=1 --end_epoch=5 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.0004 --seed=1000 --batch_size=720

MASTER_PORT=29450 torchrun --nproc_per_node=8 ./train.py --model=unet2d --start_epoch=1 --end_epoch=5 --dataset=brats_ped_2023 --save_freq=1000 --lr=0.0004 --seed=1000 --batch_size=720