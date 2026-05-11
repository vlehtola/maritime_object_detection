import torch
from torchvision.ops import box_iou

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS

"""
   Turns bounding boxes into mask where the area under bounding box considered foreground
   and area outside the bounding box is the background.
"""
@TRANSFORMS.register_module()
class LoadMask(BaseTransform):
    def __init__(self, height, width, patch_size):
        super().__init__()
        self.height = height
        self.width  = width
        self.patch_size = patch_size

    def create_mask(self, gt_bboxes):
        H_patch = self.height // self.patch_size
        W_patch = self.width // self.patch_size
        patch_boxes = []
        for i in range(H_patch):
            for j in range(W_patch):
                x1 = j * self.patch_size
                y1 = i * self.patch_size
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                patch_boxes.append([x1, y1, x2, y2])
        patch_boxes = torch.tensor(patch_boxes)
        iou_matrix = box_iou(patch_boxes, gt_bboxes)
        iou_matrix = iou_matrix.sum(dim=-1)
        gt_mask    = (iou_matrix != 0).long()
        gt_mask    = gt_mask.view(H_patch, W_patch)
        gt_mask    = gt_mask.unsqueeze(0).to(torch.float32)
        return gt_mask
    
    def transform(self, results):
        results["gt_mask"] = self.create_mask(results["gt_bboxes"].tensor)
        return results

"""
   Stack the generated mask form bounding box into one big tensor. works for single batch.
"""
@TRANSFORMS.register_module()
class CustomPackDetInputs(PackDetInputs):
    def __init__(self, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')):
        super().__init__(meta_keys)
    
    def transform(self, results):
        gt_mask = results["gt_mask"]
        results = super().transform(results)
        results["gt_mask"] = gt_mask
        return results