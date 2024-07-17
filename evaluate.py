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
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==3)))
    return mIOU_score


def softmax_output_dice(output, target, et=3):
    
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


def calculate_hd95(pred, label, voxel_spacing=(1.0, 1.0, 1.0), et=3):
    """
    Calculate the 95th percentile of the Hausdorff Distance (HD95) between the predicted and ground truth labels.
    
    Parameters:
    - pred: Predicted labels (torch tensor of shape (1, height, width, depth)).
    - label: Ground truth labels (torch tensor of shape (1, height, width, depth)).
    - voxel_spacing: The voxel spacing along each dimension (default is (1.0, 1.0, 1.0)).
    
    Returns:
    - hd95_wt: HD95 for WT (label 1 + label 2 + label 3).
    - hd95_tc: HD95 for TC (label 1 + label 3).
    - hd95_et: HD95 for ET (label 3).
    """
    pred = pred.argmax(1)

    pred = pred.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()
    
    # Define labels for WT, TC, and ET
    wt_label = [1, 2, 3]
    tc_label = [1, 3]
    et_label = [et]
    
    # Function to compute the 95th percentile Hausdorff distance
    def hd95(pred_mask, label_mask):
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            return np.inf
        
        pred_points = np.transpose(np.nonzero(pred_mask))
        label_points = np.transpose(np.nonzero(label_mask))
        
        # Apply voxel spacing
        pred_points = pred_points * voxel_spacing
        label_points = label_points * voxel_spacing
        
        # Compute directed Hausdorff distances
        forward_hd = directed_hausdorff(pred_points, label_points)[0]
        backward_hd = directed_hausdorff(label_points, pred_points)[0]
        
        # Return the 95th percentile distance
        return np.percentile([forward_hd, backward_hd], 95)
    
    # Create masks for WT, TC, and ET
    wt_pred_mask = np.isin(pred, wt_label)
    wt_label_mask = np.isin(label, wt_label)
    
    tc_pred_mask = np.isin(pred, tc_label)
    tc_label_mask = np.isin(label, tc_label)
    
    et_pred_mask = np.isin(pred, et_label)
    et_label_mask = np.isin(label, et_label)
    
    # Calculate HD95 for each region
    hd95_wt = hd95(wt_pred_mask, wt_label_mask)
    hd95_tc = hd95(tc_pred_mask, tc_label_mask)
    hd95_et = hd95(et_pred_mask, et_label_mask)
    
    return hd95_wt, hd95_tc, hd95_et