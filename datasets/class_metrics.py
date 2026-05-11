import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


"""
Calculates the patch classification accuracy of the pre backbone. 
It also calculate the precentage of foreground and background area in an image.
"""
@METRICS.register_module()
class PatchClassificationMetrics(BaseMetric):
    def __init__(self, collect_device='gpu', prefix=None, collect_dir=None):
        super().__init__(collect_device, prefix, collect_dir)
        self.accuracy = BinaryAccuracy().to("cuda")
        self.f1 = BinaryF1Score().to("cuda")
        self.precision = BinaryPrecision().to("cuda")
        self.recall = BinaryRecall().to("cuda")
        self.total_foreground_pixels = 0
        self.total_pixels = 0

    def prep_batch(self, datasamples):
        gt_mask = [sample["gt_mask"] for sample in datasamples]
        pred_mask = [sample["pred_mask"] for sample in datasamples]
        return torch.stack(gt_mask).long(), torch.stack(pred_mask)

    def process(self, data_batch, data_samples):
        gt, logits = self.prep_batch(data_samples)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        gt_flat = gt.view(-1)
        preds_flat = preds.view(-1)

        self.accuracy.update(preds_flat, gt_flat)
        self.f1.update(preds_flat, gt_flat)
        self.precision.update(preds_flat, gt_flat)
        self.recall.update(preds_flat, gt_flat)

        # Compute foreground and total pixel count for this batch
        batch_foreground = torch.sum(gt).item()
        batch_total = gt.numel()
        
        self.total_foreground_pixels += batch_foreground
        self.total_pixels += batch_total

        return super().process(data_batch, data_samples)

    def compute_metrics(self, results):
        acc = self.accuracy.compute()
        f1 = self.f1.compute()
        prec = self.precision.compute()
        rec = self.recall.compute()

        print(f"[Patch Classification] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

        self.accuracy.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

        # compute and print percentage
        foreground_ratio = (self.total_foreground_pixels / self.total_pixels) * 100
        background_ratio = 100 - foreground_ratio

        print(f"Foreground: {foreground_ratio:.2f}% | Background: {background_ratio:.2f}%")
        
        self.total_foreground_pixels = 0
        self.total_pixels = 0

        return { "accuracy": acc,  "f1_score": f1,  "precision": prec,  "recall": rec, "Fore": foreground_ratio, "Back": background_ratio}
    

"""
This custom metric computes patch-level binary classification accuracy for predicted masks within 
bounding boxes, categorized by object size (small, medium, large) using the COCO size thresholds.
"""
@METRICS.register_module()
class BoxPatchClassificationMetrics(BaseMetric):
    def __init__(self, patch_size, collect_device='gpu', prefix=None, collect_dir=None):
        super().__init__(collect_device, prefix, collect_dir)
        self.patch_size  = patch_size
        self.metrics = {
            'small':  BinaryAccuracy().to("cuda"),
            'medium': BinaryAccuracy().to("cuda"),
            'large':  BinaryAccuracy().to("cuda")}
        
        self.data = {
            'small':  {'gts': [], 'preds': []},
            'medium': {'gts': [], 'preds': []},
            'large':  {'gts': [], 'preds': []}}

    def get_size_category(self, bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area < 32 ** 2:
            return 'small'
        elif area < 96 ** 2:
            return 'medium'
        else:
            return 'large'

    def convert_bbox_to_patch_coords(self, bbox, original_size, input_size):
        x1, y1, x2, y2 = bbox
        H, W = original_size  # original height and width
        h, w = input_size     # resized height and width used in model

        # compute scaling factors
        scale_x = w / W
        scale_y = h / H

        # rescale bbox to match resized input size
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

        # convert to patch indices
        patch_x1 = int(x1 // self.patch_size)
        patch_y1 = int(y1 // self.patch_size)
        patch_x2 = int((x2 + self.patch_size - 1) // self.patch_size)
        patch_y2 = int((y2 + self.patch_size - 1) // self.patch_size)

        return patch_x1, patch_y1, patch_x2, patch_y2

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            gt_mask   = sample["gt_mask"]       
            pred_mask = sample["pred_mask"]   
            bboxes    = sample["gt_instances"]["bboxes"]   
               
            orig_shape  = sample["ori_shape"]
            input_shape = sample["img_shape"]
            for bbox in bboxes:
                size_cat       = self.get_size_category(bbox)
                x1, y1, x2, y2 = self.convert_bbox_to_patch_coords(bbox, orig_shape, input_shape)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                gt_patch   = gt_mask[0, y1:y2, x1:x2].flatten().long()
                pred_probs = torch.sigmoid(pred_mask[0, y1:y2, x1:x2])
                pred_patch = (pred_probs > 0.3).flatten().long()
             
                if gt_patch.numel() == 0 or pred_patch.numel() == 0:
                    continue
                self.data[size_cat]['gts'].append(gt_patch)
                self.data[size_cat]['preds'].append(pred_patch)

        return super().process(data_batch, data_samples)

    def compute_metrics(self, results):
        final_results = {}
        for size_cat, metric in self.metrics.items():
            if not self.data[size_cat]['gts']:
                continue
            gts   = torch.cat(self.data[size_cat]['gts'])
            preds = torch.cat(self.data[size_cat]['preds'])

            metric.update(preds, gts)
            acc = metric.compute().item()
            print(f"[{size_cat.capitalize()} BBox] Accuracy: {acc:.4f}")
            final_results[f"{size_cat}_accuracy"] = acc
            metric.reset()

        # Clear saved data
        for cat in self.data:
            self.data[cat]['gts'].clear()
            self.data[cat]['preds'].clear()

        return final_results