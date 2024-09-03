import os
import time
import logging
import torch
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
from config import Config, PediatricConfig, AdultConfig
from evaluate import tailor_and_concat_3d, tailor_and_concat_2d, softmax_dice_score, softmax_mIOU_score, aggregate_2d_to_3d, compute_hd95_batch
from data.dataset import BraTS3D, BraTS2D
from models.transbts.transbts_downsample8x import get_default as TransBTS
from models.unet.unet3d import get_default as UNet3D
from models.unet.unet2d import get_default as UNet2D
from models.vit.vit2d import get_default as ViT2D
from models.swinunetr.swinunetr import get_default as SwinUNETR
from torch.utils.data import DataLoader
from criterion import softmax_dice
import json
import gc
from utils import Accumulator
from torch import nn


config = Config()


def validate(
        title, 
        data_loader,
        model,
        load_file,
        verbose=False,
        tta=False,
        format='nii',
        snapshot=False,
        valid=True,
        name_list=None, 
        save_path=None, 
        affine=None, 
        dim="3D", 
        final_act=False
        ):

    H, W, D = 240, 240, 155
    model.eval()

    tailor_and_concat = tailor_and_concat_3d if dim == "3D" else tailor_and_concat_2d

    runtimes = []

    stats = []

    criterion = softmax_dice

    checkpoint = torch.load(load_file)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    # model.cuda()
    model.to("cuda:1")
    
    metric = Accumulator(3)
    more = Accumulator(6)

    for i, (data, idx) in enumerate(data_loader):
        msg = 'Iter {}/{}, '.format(i + 1, len(data_loader))

        if name_list:
            if dim == "3D":
                name = name_list[idx]
            else:
                name = name_list[idx[0] // 155]
        else:
            if dim == "3D":
                name = str(idx)
            else:
                name = str(idx[0] // 155)
            

        if valid:
            # data = [sample.cuda(non_blocking=True) for sample in data]
            data = [sample.to("cuda:1") for sample in data]
            x, target = data
            if dim == "3D":
                target = target[..., :155]
        else:
            x = data.cuda(non_blocking=True)


        torch.cuda.synchronize()
        start_time = time.time()
        logit = tailor_and_concat(x, model, final_act=final_act)

        torch.cuda.synchronize()
        duration = time.time() - start_time
        runtimes.append(duration)

        if not tta:
            output = logit.clone()
        else:
            x = x[..., :155]
            logit = tailor_and_concat(x, model, final_act=final_act)  # no flip
            logit += tailor_and_concat(x.flip(dims=(2,)), model, final_act=final_act).flip(dims=(2,))  # flip H
            logit += tailor_and_concat(x.flip(dims=(3,)), model, final_act=final_act).flip(dims=(3,))  # flip W
            logit += tailor_and_concat(x.flip(dims=(2, 3)), model, final_act=final_act).flip(dims=(2, 3))  # flip H, W
            if dim == "3D":
                logit += tailor_and_concat(x.flip(dims=(4,)), model, final_act=final_act).flip(dims=(4,))  # flip D

                logit += tailor_and_concat(x.flip(dims=(2, 4)), model, final_act=final_act).flip(dims=(2, 4))  # flip H, D
                logit += tailor_and_concat(x.flip(dims=(3, 4)), model, final_act=final_act).flip(dims=(3, 4))  # flip W, D
                logit += tailor_and_concat(x.flip(dims=(2, 3, 4)), model, final_act=final_act).flip(dims=(2, 3, 4))  # flip H, W, D
                output = logit / 8.0  # mean
            else:
                output = logit / 4.0 # mean

        if dim == "2D":
            output, target = aggregate_2d_to_3d(output, target)
        
        if valid:
            _, score1, score2, score3 = criterion(output, target)
            dice_out = [score1.detach().cpu().item(), score2.detach().cpu().item(), score3.detach().cpu().item()]
            
            output = output[0, :, :H, :W, :D]
            output = output.argmax(0)
            target = target[0, :]
            
            miou_out = softmax_mIOU_score(output, target)
            hd95_out = compute_hd95_batch(output, target)
            eval = softmax_dice_score(output, target)

            stats.append({"dice": dice_out, "dice_more": eval, "miou": miou_out, "hd95": hd95_out, "name": name, "wt": eval[0], "tc": eval[1], "et": eval[2]})
            metric.add(*eval)
            more.add(*hd95_out[4:], *miou_out[3:])

            output = output.cpu().detach().numpy()
        else:
            output = output[0, :, :H, :W, :D].cpu().detach().numpy()
            output = output.argmax(0)

        logit.to("cpu")
        x.to("cpu")

        del logit, x
        gc.collect()
        
        torch.cuda.empty_cache()

        print(msg + name)

        if save_path:
            assert format in ['npy', 'nii'], "Invalid save format, check first (`nii` and `npy` only)"
            if not os.path.exists(os.path.join(save_path, title)):
                os.makedirs(os.path.join(save_path, title))
            if format == 'npy':
                np.save(os.path.join(save_path, title, name + '.npy'), output)
            if format == 'nii':
                path = os.path.join(save_path, title, name + '.nii.gz')
                if verbose:
                    print('1:', np.sum(output == 1), ' | 2:', np.sum(output == 2), ' | 3:', np.sum(output == 3))
                    print('WT:', np.sum((output == 1) | (output == 2) | (output == 3)), ' | TC:',
                          np.sum((output == 1) | (output == 3)), ' | ET:', np.sum(output == 3))
                nib.save(nib.Nifti1Image(output.astype(np.float64), affine), path)
                print('{} has been saved successfully.'.format(name))

                if snapshot:
                    ss_img = np.zeros(shape=(H, W, 3, D), dtype=np.uint8)
                    ss_img[:, :, 0, :][np.where(output == 1)] = 255
                    ss_img[:, :, 1, :][np.where(output == 2)] = 255
                    ss_img[:, :, 2, :][np.where(output == 3)] = 255

                    if not os.path.exists(os.path.join(visual_path, title, name)):
                            os.makedirs(os.path.join(visual_path, title, name))

                    for frame in range(D):
                        imageio.imwrite(os.path.join(visual_path, title, name, str(frame)+'.png'), ss_img[:, :, :, frame])
    
    if valid:
        valid_stats_dir = os.path.join(config.COLLECTION_DIR, "valid")
        if not os.path.exists(valid_stats_dir):
            os.makedirs(valid_stats_dir)
        with open(os.path.join(valid_stats_dir, f"valid_stats-{title}.json"), "w") as f:
            json.dump(stats, f)
        print(metric.avg())
        print(more.avg())

    print(f'Total runtime: {round(sum(runtimes)/len(runtimes), 4)} minutes.')


if __name__ == "__main__":

    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())

    save_path=config.RES_DIR
    visual_path = config.VISUAL_DIR
    
    config = PediatricConfig()
    brats_ped_valid = config.BRATS_TRAIN
    valid_dir = brats_ped_valid["dir"]
    valid_list = brats_ped_valid["list"]

    # valid_set = BraTS3D(os.path.join(valid_dir, valid_list), valid_dir, "valid")
    # print(valid_set.names)
    # valid_set = BraTS2D(os.path.join(valid_dir, valid_list), valid_dir, "valid")
    # print(valid_set.names)

    # valid_set = BraTS3D(os.path.join(valid_dir, valid_list), valid_dir, "valid")
    # valid_loader = DataLoader(dataset=valid_set, batch_size=1,
    #                           drop_last=False, num_workers=6, pin_memory=True, shuffle=False)

    # model = TransBTS()
    # load_file = os.path.join(config.CHECK_POINT_DIR, "transbts-brats_ped_2023-2024-08-07", "transbts-brats_ped_2023-last.pth")
    # validate("transbts-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.names, verbose=True, save_path=save_path, valid=True, affine=config.affine)

    # model = UNet3D()
    # load_file = os.path.join(config.CHECK_POINT_DIR, "unet3d-brats_ped_2023-2024-08-07", "unet3d-brats_ped_2023-last.pth")
    # validate("unet3d-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.names, verbose=True, save_path=save_path, valid=True, affine=config.affine)

    # model = SwinUNETR()
    # load_file = os.path.join(config.CHECK_POINT_DIR, "swinunetr-brats_ped_2023-2024-08-07", "swinunetr-brats_ped_2023-last.pth")
    # validate("swinunetr-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.names, verbose=True, save_path=save_path, valid=True, affine=config.affine, final_act=True)

    valid_set = BraTS2D(os.path.join(valid_dir, valid_list), valid_dir, "valid")
    print(valid_set.names)
    valid_loader = DataLoader(dataset=valid_set, batch_size=155,
                              drop_last=False, num_workers=6, pin_memory=True, shuffle=False)
    
    model = UNet2D()
    load_file = os.path.join(config.CHECK_POINT_DIR, "unet2d-brats_ped_2023-2024-08-08", "unet2d-brats_ped_2023-last.pth")
    validate("unet2d-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.names, verbose=True, save_path=save_path, valid=True, affine=config.affine, dim="2D")

    model = ViT2D()
    load_file = os.path.join(config.CHECK_POINT_DIR, "vit2d-brats_ped_2023-2024-08-07", "vit2d-brats_ped_2023-last.pth")
    validate("vit2d-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.names, verbose=True, save_path=save_path, valid=True, affine=config.affine, dim="2D")