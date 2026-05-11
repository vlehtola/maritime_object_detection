import torch

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS

@MODELS.register_module()
class CustomDetDataPreprocessor(DetDataPreprocessor):
    '''
       turn list ground truth background/foreground mask to stack of tensors.
    '''
    def __init__(self, 
                 mean = None, 
                 std = None, 
                 pad_size_divisor = 1, 
                 pad_value = 0, 
                 pad_mask = False, 
                 mask_pad_value = 0, 
                 pad_seg = False, 
                 seg_pad_value = 255, 
                 bgr_to_rgb = False, 
                 rgb_to_bgr = False, 
                 boxtype2tensor = True, 
                 non_blocking = False, 
                 batch_augments = None):
        super().__init__(mean, 
                         std, 
                         pad_size_divisor, 
                         pad_value, 
                         pad_mask, 
                         mask_pad_value, 
                         pad_seg, 
                         seg_pad_value, 
                         bgr_to_rgb, 
                         rgb_to_bgr, 
                         boxtype2tensor, 
                         non_blocking, 
                         batch_augments)
        
        
    
    def pack_mask(self, data):
        out = torch.stack(data["gt_mask"])
        return out
    
    def forward(self, data, training = False):
        gt_masks = self.cast_data(self.pack_mask(data))
        result = super().forward(data, training)
        result['gt_masks'] = gt_masks
        return result