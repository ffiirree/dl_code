import torch
import torch.nn as nn

__all__ = ['mean_iou', 'mean_dice', 'mean_acc', 'intersect_and_union']


def f_score(precision, recall, beta=1):
    """
        TP: true positive
        FP: false positive
        TN: true negative
        FN: false negative
        
        precision   = TP / (TP + FP)
        recall      = TP / (TP + FN)
        
        
        f1-score    = 2 * (precision * recall) / (precision + recall)
                    = 2 * TP / (2 * TP + FN + FP)
                    = dice coefficient
    """
    return (1 + beta**2) * (precision * recall) / (beta ** 2 * precision + recall)

def mean_iou(pred: torch.Tensor, label: torch.Tensor, num_classes:int, ignore_index:int):
    """ Mean Intersection and Union(mIoU)
    """
    n_intersect, n_union, n_pred, n_label = intersect_and_union(pred, label, num_classes, ignore_index)
    return n_intersect / (n_union + 10e-8)

def mean_dice(pred: torch.Tensor, label: torch.Tensor, num_classes:int):
    """ Mean Dice(mDice)
    """
    n_intersect, n_union, n_pred, n_label = intersect_and_union(pred, label, num_classes)
    return 2 * n_intersect / (n_pred + n_label + + 10e-8)

def mean_acc(pred: torch.Tensor, label: torch.Tensor, num_classes:int, ignore_index:int):
    """ mean accuraccy
        $ \frac{1}{n_{c1}} $
    """
    n_intersect, n_union, n_pred, n_label = intersect_and_union(pred, label, num_classes, ignore_index)
    return n_intersect / (n_label + 10e-8)


def mean_fscore(pred: torch.Tensor, label: torch.Tensor, num_classes:int, ignore_index:int, beta:int = 1):
    n_intersect, n_union, n_pred, n_label = intersect_and_union(pred, label, num_classes, ignore_index)
    
    precision = n_intersect / (n_pred + 10e-8)
    recall = n_intersect / (n_label + 10e-8)
    
    return [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
    

def intersect_and_union(pred:torch.Tensor, label:torch.Tensor, num_classes:int, ignore_index:int):
    mask = (label != ignore_index)

    pred = pred[mask]
    label = label[mask]
    
    intersect = pred[pred == label]
    n_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes-1)
    n_pred      = torch.histc(pred.float(), bins=(num_classes), min=0, max=num_classes-1)
    n_label     = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes-1)
    
    n_union = n_pred + n_label - n_intersect
    
    return n_intersect, n_union, n_pred, n_label
    
    