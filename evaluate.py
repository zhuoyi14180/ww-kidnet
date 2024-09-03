import torch.nn.functional as F
import torch
import numpy as np
from utils import one_hot
from scipy.spatial.distance import directed_hausdorff
import GeodisTK
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


def compute_hd95(prediction, target):
    # Ensure the prediction and target are binary
    prediction = (prediction > 0.5).astype(float)
    target = (target > 0.5).astype(float)

    if np.sum(prediction) == 0 and np.sum(target) == 0:
        return 0.0  # or a very large number like float('inf')

    # If one of them is empty, the distance is also considered large
    if np.sum(prediction) == 0 or np.sum(target) == 0:
        # return float('inf')
        # return abs(np.sum(prediction) - np.sum(target))
        return 100.0
    
    # Compute the distance transform for both the prediction and the target
    pred_dist = distance_transform_edt(1 - prediction)
    target_dist = distance_transform_edt(1 - target)
    
    # Compute distances from predicted to nearest target voxel
    pred_to_target = pred_dist[target > 0]
    target_to_pred = target_dist[prediction > 0]
    
    # Compute the 95th percentile of the distances
    hd95 = max(np.percentile(pred_to_target, 95), np.percentile(target_to_pred, 95))
    
    return hd95

# Example usage
def compute_hd95_batch(pred_batch, target_batch):
    hd95_list = []
    target_batch = one_hot(target_batch, 4)
    pred_batch = one_hot(pred_batch, 4)
    for c in range(pred_batch.shape[0]):  # Iterate over the channels
        pred = pred_batch[c].cpu().numpy()
        target = target_batch[c].cpu().numpy()
        hd95 = compute_hd95(pred, target)
        hd95_list.append(hd95)

    hd95_wt, hd95_tc, hd95_et = compute_hd95_sub(pred_batch, target_batch)
    hd95_list.append(hd95_wt)
    hd95_list.append(hd95_tc)
    hd95_list.append(hd95_et)
    return hd95_list

def compute_hd95_sub(pred_batch, target_batch):
    # 合并类别以得到 WT 区域
    pred_wt = (pred_batch[1] > 0.5) | (pred_batch[2] > 0.5) | (pred_batch[3] > 0.5)
    target_wt = (target_batch[1] > 0.5) | (target_batch[2] > 0.5) | (target_batch[3] > 0.5)
    
    # 计算 WT 的 HD95
    hd95_wt = compute_hd95(pred_wt.cpu().numpy(), target_wt.cpu().numpy())


    pred_tc = (pred_batch[1] > 0.5) | (pred_batch[3] > 0.5)
    target_tc = (target_batch[1] > 0.5) | (target_batch[3] > 0.5)
    
    hd95_tc = compute_hd95(pred_tc.cpu().numpy(), target_tc.cpu().numpy())

    pred_et = (pred_batch[3] > 0.5)
    target_et = (target_batch[3] > 0.5)
    
    hd95_et = compute_hd95(pred_et.cpu().numpy(), target_et.cpu().numpy())
    
    return hd95_wt, hd95_tc, hd95_et


def dice_score(output, target, eps=1e-8):
    target = target.float()
    num = 2. * (output * target).sum() + eps
    den = output.sum() + target.sum() + eps
    return num / den


def mIOU(o, t, eps=1e-8):
    num = (o * t).sum() + eps
    den = (o | t).sum() + eps
    return num / den


def softmax_mIOU_score(output, target, et=3):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1), t=(target==1)).detach().item())
    mIOU_score.append(mIOU(o=(output==2), t=(target==2)).detach().item())
    mIOU_score.append(mIOU(o=(output==3), t=(target==3)).detach().item())

    # whole
    o = output > 0; t = target > 0 # ce
    mIOU_score.append(mIOU(o, t).item())

    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    mIOU_score.append(mIOU(o, t).item())

    # active
    o = (output == et); 
    t = (target == et)
    mIOU_score.append(mIOU(o, t).item())


    return mIOU_score


def softmax_dice_score(output, target, et=3):
    res = []
    # whole
    o = output > 0; t = target > 0 # ce
    res.append(dice_score(o, t).item())

    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    res.append(dice_score(o, t).item())

    # active
    o = (output == et); 
    t = (target == et)
    res.append(dice_score(o, t).item())

    return res


def tailor_and_concat_3d(x, model, flag=True, final_act=False):
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
        if final_act:
            temp[i] = F.softmax(model(temp[i]), dim=1).detach().cpu()
        else:
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


def tailor_and_concat_2d(x, model, patch_size=224, flag=True, final_act=False):
    if not flag:
        return model(x)

    _, _, h, w = x.size()
    temp = []

    # 划分为4个patches
    temp.append(x[..., :patch_size, :patch_size])  # top-left
    temp.append(x[..., :patch_size, w - patch_size:])  # top-right
    temp.append(x[..., h - patch_size:, :patch_size])  # bottom-left
    temp.append(x[..., h - patch_size:, w - patch_size:])  # bottom-right

    y = x.clone()

    for i in range(len(temp)):
        if final_act:
            temp[i] = F.softmax(model(temp[i]), dim=1).detach().cpu()
        else:
            temp[i] = model(temp[i]).detach().cpu()

    y[..., :patch_size, :patch_size] = temp[0]
    y[..., :patch_size, patch_size:] = temp[1][..., :, patch_size - 16:]
    y[..., patch_size:, :patch_size] = temp[2][..., patch_size - 16:, :]
    y[..., patch_size:, patch_size:] = temp[3][..., patch_size - 16:, patch_size - 16:]

    return y


def aggregate_2d_to_3d(o2d, t2d):
    o3d = o2d.permute(1, 2, 3, 0)
    o3d = o3d.unsqueeze(0)

    t3d = t2d.permute(1, 2, 0)
    t3d = t3d.unsqueeze(0)
    return o3d, t3d
