import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
from config import Config, PediatricConfig, AdultConfig
from evaluate import dice_score
from prepare.data import BraTS
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models.unet.unet3d import UNet3D
from torch.utils.data import DataLoader
from criterion import softmax_dice
from evaluate import softmax_output_dice
import json
import gc

from utils import Accumulator

config = Config()

def tailor_and_concat(x, model, flag=True):
    if (flag == False):
        return model(x)

    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()

    for i in range(len(temp)):
        temp[i] = model(temp[i]).detach().cpu()

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


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
        save_path=None
        ):

    H, W, D = 224, 224, 155
    model.eval()

    runtimes = []

    stats = []

    dice_out = []

    criterion = softmax_dice
    
    checkpoint = torch.load(load_file)

    origin_state_dict = checkpoint['state_dict']

    current_state_dict = model.state_dict()

    # 修改不匹配的键名
    for key in list(origin_state_dict.keys()):
        if key not in current_state_dict:
            new_key = key.replace("module.", "")  # 去除多GPU前缀
            origin_state_dict[new_key] = origin_state_dict.pop(key)
    # 加载修改后的state_dict
    model.load_state_dict(origin_state_dict)

    model.cuda()
    
    metric = Accumulator(3)

    for i, (data, idx) in enumerate(data_loader):
        msg = 'Iter {}/{}, '.format(i + 1, len(data_loader))

        if name_list:
            name = name_list[idx]
        else:
            name = str(idx)

        if valid:
            data = [sample.cuda(non_blocking=True) for sample in data]
            x, target = data
            target = target[..., :155]
        else:
            x = data.cuda(non_blocking=True)


        torch.cuda.synchronize()
        start_time = time.time()
        logit = tailor_and_concat(x, model)

        torch.cuda.synchronize()
        duration = time.time() - start_time
        runtimes.append(duration)

        if not tta:
            # output = F.softmax(logit, dim=1)
            output = logit.clone()
        else:
            x = x[..., :155]
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0  # mean
        
        if valid:
            loss, score1, score2, score3 = criterion(output, target)
            stats.append({"data": [loss.detach().cpu().item(), score1.detach().cpu().item(), score2.detach().cpu().item(), score3.detach().cpu().item()], "name": name})
            res = softmax_output_dice(output, target)
            dice_out.append(res)
            metric.add(*res)

        logit.to("cpu")
        x.to("cpu")

        del logit, x
        gc.collect()

        output = output[0, :, :H, :W, :D].cpu().detach().numpy()
        torch.cuda.empty_cache()
        output = output.argmax(0)
        
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
                nib.save(nib.Nifti1Image(output.astype(np.float32), None), path)
                print('{} has been saved successfully.'.format(name))

                if snapshot:
                    ss_img = np.zeros(shape=(H, W, 3, D), dtype=np.uint8)
                    ss_img[:, :, 0, :][np.where(output == 1)] = 255
                    ss_img[:, :, 1, :][np.where(output == 2)] = 255
                    ss_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(D):
                        if not os.path.exists(os.path.join(visual_path, title, name)):
                            os.makedirs(os.path.join(visual_path, title, name))
                        imageio.imwrite(os.path.join(visual_path, title, name, str(frame)+'.png'), ss_img[:, :, :, frame])
    
    
    if valid:
        with open(os.path.join(config.PIC_DIR, f"valid_stats-{title}.json"), "w") as f:
            json.dump(stats, f)

        print(dice_out)

        print(metric.avg())


    print(f'Total runtime: {round(sum(runtimes)/len(runtimes), 4)} minutes.')


if __name__ == "__main__":

    save_path=config.RES_DIR
    visual_path = config.VISUAL_DIR


    brats_ped_valid = PediatricConfig().BRATS_VALID
    valid_dir = brats_ped_valid["dir"]
    valid_list = brats_ped_valid["list"]

    valid_set = BraTS(os.path.join(valid_dir, valid_list), valid_dir, "test")
    valid_loader = DataLoader(dataset=valid_set, batch_size=1,
                              drop_last=False, num_workers=6, pin_memory=True, shuffle=False)


    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    load_file = os.path.join(config.CHECK_POINT_DIR, "transbts-brats_ped_2023-2024-06-28", "transbts-brats_ped_2023-last.pth")
    validate("transbts-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.name_list, verbose=True, save_path=save_path, valid=False)


    model = UNet3D(4, 4)
    load_file = os.path.join(config.CHECK_POINT_DIR, "unet3d-brats_ped_2023-2024-06-28", "unet3d-brats_ped_2023-last.pth")
    validate("unet3d-brats_ped_2023-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.name_list, verbose=True, save_path=save_path, valid=False)




    # brats_valid = AdultConfig().BRATS_VALID
    # valid_dir = brats_valid["dir"]
    # valid_list = brats_valid["list"]

    # valid_set = BraTS(os.path.join(valid_dir, valid_list), valid_dir, "test")
    # valid_loader = DataLoader(dataset=valid_set, batch_size=1,
    #                           drop_last=False, num_workers=6, pin_memory=True, shuffle=False)


    # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    # load_file = os.path.join(config.CHECK_POINT_DIR, "transbts-brats_2019-2024-06-26", "transbts-brats_2019-last.pth")
    # validate("transbts-brats_2019-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.name_list, verbose=True, save_path=save_path, valid=False)


    # model = UNet3D(4, 4)
    # load_file = os.path.join(config.CHECK_POINT_DIR, "unet3d-brats_2019-2024-06-27", "unet3d-brats_2019-last.pth")
    # validate("unet3d-brats_2019-poly-valid", valid_loader, model, load_file, snapshot=False, name_list=valid_set.name_list, verbose=True, save_path=save_path, valid=False)