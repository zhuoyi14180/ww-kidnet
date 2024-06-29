import torch

def dice_score(output, target, eps=1e-8):
    target = target.float()
    num = 2. * (output * target).sum() + eps
    den = output.sum() + target.sum() + eps
    return num / den


def mIOU(output, target, eps=1e-8):
    num = (output * target).sum() + eps
    den = (output | target).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==3)))
    return mIOU_score


def softmax_output_dice(output, target):
    
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
    o = (output == 3); 
    t = (target == 3)
    res += dice_score(o, t).item(),

    return res