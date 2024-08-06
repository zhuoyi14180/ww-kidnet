import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_score(output, target, eps=1e-8):
    target = target.float()
    num = 2. * (output * target).sum() + eps
    den = output.sum() + target.sum() + eps
    return num / den


def mIOU(output, target, eps=1e-8):
    num = (output * target).sum() + eps
    den = (output | target).sum() + eps
    return num / den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1), t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2), t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3), t=(target==3)))
    return mIOU_score


def softmax_dice_score(output, target, et=3):
    
    output = output.argmax(1)

    output = torch.squeeze(output)
    res = []
    # whole
    o = output > 0; t = target > 0 # ce
    res += dice_score(o, t).item(),

    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    res += dice_score(o, t).item(),

    # active
    o = (output == et); 
    t = (target == et)
    res += dice_score(o, t).item(),

    return res


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