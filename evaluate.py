def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o * t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==3)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),

    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += dice_score(o, t),

    # active
    o = (output == 3); 
    t = (target == 3)
    ret += dice_score(o, t),

    return ret