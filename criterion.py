import torch
import torch.nn.functional as F


def expand_target(x, n_class, mode='softmax'):
    """
    Converts the shape of annotations from (N, D, H, W) to (N, C, D, H, W), where each label is stored in a separate channel
    :param input: 4D input image (N, D, H, W)
    :param C: number of labels
    :return: 5D output image (N, C, D, H, W)
    """
    assert x.dim() == 4, "Invalid input dimension, check first."
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        for c in range(1, n_class):
            xx[:, c, :, :, :] = (x == c)
    if mode.lower() == 'sigmoid':
        for c in range(0, n_class-1):
            xx[:, c, :, :, :] = (x == c)
    return xx.to(x.device)


def flatten(x):
    """
    Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows: (N, C, D, H, W) -> (C, N * D * H * W)
    """
    num_c = x.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, x.dim())) # (1, 0, 2, 3, 4)
    # transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = x.permute(axis_order)
    # flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(num_c, -1)


def dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss - version 1
    '''
    loss1 = dice(output[:, 1, ...], (target == 1).float())
    loss2 = dice(output[:, 2, ...], (target == 2).float())
    loss3 = dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def softmax_dice_back(output, target):
    '''
    The dice loss for using softmax activation function (background considered)
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss - version 2
    '''
    loss0 = dice(output[:, 0, ...], (target == 0).float())
    loss1 = dice(output[:, 1, ...], (target == 1).float())
    loss2 = dice(output[:, 2, ...], (target == 2).float())
    loss3 = dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3 + loss0, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = dice(output[:, 0, ...], (target == 1).float())
    loss2 = dice(output[:, 1, ...], (target == 2).float())
    loss3 = dice(output[:, 2, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data