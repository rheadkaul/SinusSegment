


def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    valid_mask = target_binary.sum(dim=(1, 2, 3)) > 0

    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3)) - intersection

    iou = intersection[valid_mask] / union[valid_mask]

    if valid_mask.sum() == 0:
        return 0.0

    return iou.mean().item()


def compute_dice(pred, target, threshold=0.5):

    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    valid_mask = target_binary.sum(dim=(1, 2, 3)) > 0

    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))

    dice = (2 * intersection[valid_mask]) / (
                pred_binary.sum(dim=(1, 2, 3))[valid_mask] + target_binary.sum(dim=(1, 2, 3))[valid_mask])

    if valid_mask.sum() == 0:
        return 0.0

    return dice.mean().item()


def compute_sensitivity_specificity(preds, labels, threshold=0.5):

    preds = (preds >= threshold).float()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return sensitivity, specificity
    
import torch

def extract_foreground_points(segmentation):
    points = torch.nonzero(segmentation, as_tuple=False)
    return points


def hausdorff_distance(A, B):

    dists_A_to_B = torch.cdist(A.float(), B.float(), p=2)
    min_dists_A_to_B = torch.min(dists_A_to_B, dim=1)[0]
    max_min_dist_A_to_B = torch.max(min_dists_A_to_B)

    dists_B_to_A = torch.cdist(B.float(), A.float(), p=2)
    min_dists_B_to_A = torch.min(dists_B_to_A, dim=1)[0]
    max_min_dist_B_to_A = torch.max(min_dists_B_to_A)

    hausdorff_dist = torch.max(max_min_dist_A_to_B, max_min_dist_B_to_A)

    return hausdorff_dist.item()


def hausdorff_distance_for_segmentation(segmentation1, segmentation2,threshold = 0.5):

    segmentation1 = (segmentation1 > threshold).float()
    segmentation2 = (segmentation2 > threshold).float()

    points1 = extract_foreground_points(segmentation1)
    points2 = extract_foreground_points(segmentation2)

    if points1.size(0) == 0 or points2.size(0) == 0:
        return 0.0

    return hausdorff_distance(points1, points2)

