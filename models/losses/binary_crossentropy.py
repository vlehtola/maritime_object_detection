import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

"""
    Computes the Binary crossentropy between mask and predicted mask
"""
@MODELS.register_module()
class BackgroundForewardClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target):
        logits = logits.view(-1)
        target = target.view(-1)
        
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()

        pos_weight = num_neg / (num_pos + 1e-6) 
        pos_weight = torch.tensor([pos_weight], device=logits.device)

        loss = F.binary_cross_entropy_with_logits(logits, target=target, pos_weight=pos_weight)
        return loss