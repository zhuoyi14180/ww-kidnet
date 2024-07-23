torchrun --nproc_per_node=4 ./train.py --model=transbts --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --gpu=0,1,2,3 --batch_size=12 &

torchrun --nproc_per_node=4 ./train.py --model=unet3d --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --gpu=4,5,6,7 --batch_size=12 &