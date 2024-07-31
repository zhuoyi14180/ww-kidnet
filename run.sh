# torchrun --nproc_per_node=4 ./train.py --model=transbts --end_epoch=2000 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --batch_size=8

# torchrun --nproc_per_node=4 ./train.py --model=unet3d --end_epoch=2000 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --batch_size=8



torchrun --nproc_per_node=4 ./train.py --model=transbts --end_epoch=5 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --batch_size=8

torchrun --nproc_per_node=4 ./train.py --model=unet3d --end_epoch=5 --dataset=brats_ped_2023 --save_freq=400 --lr=0.0004 --seed=1000 --batch_size=8