"""
  Originally written by lyuwneyu @ https://github.com/lyuwenyu/RT-DETR
  Copied and modified here to make it part of MMDetection framework!!!
"""
import os
import cv2
import copy
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
import torchvision
import torchvision.transforms.functional as TF

from mmengine.optim import OptimWrapper
from mmengine.model.base_model import BaseModel
from mmengine.structures import InstanceData
from mmdet.registry import MODELS

from ..layers.common import MLP
from ..layers.encoder_decoder_layers import EncoderLayer, Encoder, Decoder, DecoderLayer
from ..backbones import PrunedBackbone
from ..utils import bias_init_with_prob, get_contrastive_denoising_training_group
 
@MODELS.register_module()              
class HybridEncoder(nn.Module):
    def __init__(self,
                 attn_module,
                 fpn_module,
                 in_channels = [512, 1024, 2048],
                 feat_strides = [8, 16, 32],
                 hidden_dim = 256, # or d_model
                 dim_feedforward = 1024,
                 dropout = 0.0,
                 enc_act = "gelu",
                 use_encoder_idx = [2],
                 num_encoder_layers = 1,
                 pe_temperature = 10000,
                 eval_spatial_size = None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)))
        
        # encoder 
        attn_module = MODELS.build(attn_module)
        encoder_layer = EncoderLayer(att_module = attn_module, 
                                     d_model = hidden_dim, 
                                     dim_feedforward = dim_feedforward, 
                                     dropout = dropout, 
                                     act = enc_act)
        
        self.encoder  = nn.ModuleList([Encoder(copy.deepcopy(encoder_layer), num_layers=num_encoder_layers ) for _ in range(len(use_encoder_idx))])
        
        # feature pyramid network
        self.fpn  = MODELS.build(fpn_module) if fpn_module is not None else nn.Identity()

        # initialized weights
        self._reset_parameters()
        
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
    
    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # feature pyramid fusion
        outs = self.fpn(proj_feats) if not isinstance(self.fpn, nn.Identity) else proj_feats
        return outs

@MODELS.register_module()
class RTDETRTransformer(nn.Module):
    def __init__(self,
                 self_attn,
                 cross_attn,
                 num_classes=12,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):

        super().__init__()
        assert position_embed_type in ['sine', 'learned'], f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size  = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # creating attention layers
        self_attn  = MODELS.build(self_attn)
        cross_attn = MODELS.build(cross_attn)

        # decoder
        decoder_layer = DecoderLayer(self_attn_obj=self_attn, 
                                     cross_atten_obj=cross_attn, 
                                     d_model=hidden_dim,
                                     dim_feedforward=dim_feedforward, 
                                     activation=activation, 
                                     dropout=0.0)
        
        self.decoder  = Decoder(hidden_dim=hidden_dim, 
                                decoder_layer=decoder_layer, 
                                num_layers=num_decoder_layers, 
                                eval_idx=eval_idx)

        self.num_denoising     = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale   = box_noise_scale

        # denoising part
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed  = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,))
        
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)])
        
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])))

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])))
            in_channels = self.hidden_dim
    
    # create a big sequence of token from all feature maps of feature pyramid
    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    # do the feature map selection based on confidence label
    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        
        # enc_outputs [bs, num patches, class score] 
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
    
        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
    
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

@MODELS.register_module()
class RTDETRPostProcessor(nn.Module):
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        # print(f"bbox ori { (bbox_pred*orig_target_sizes.repeat(1, 2).unsqueeze(1))[0] }")
        bbox_pred[..., [0, 2]] = bbox_pred[..., [0, 2]] * orig_target_sizes[1]
        bbox_pred[..., [1, 3]] = bbox_pred[..., [1, 3]] * orig_target_sizes[0] 

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])) 
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores
        
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        
    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )

@MODELS.register_module()
class RTDETR(BaseModel):
    def __init__(self,  backbone, encoder, decoder, post_processor, loss_fn, data_preprocessor = None, mask_fn = None, init_cfg = None):
        super().__init__(init_cfg)
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.post_processor    = MODELS.build(post_processor)
        
        self.backbone = MODELS.build(backbone) 
        self.encoder  = MODELS.build(encoder)
        self.decoder  = MODELS.build(decoder)
        self.loss_fn  = MODELS.build(loss_fn)
        self.mask_fn  = MODELS.build(mask_fn) if mask_fn else None
        self.iter     = 1

    def train_step(self, data: Union[dict, Tuple, List], optim_wrapper: OptimWrapper)->Dict[str, torch.Tensor]:
        return super().train_step(data=data, optim_wrapper=optim_wrapper)

    def val_step(self, data: Union[tuple, dict, list])->list:
        return super().val_step(data=data)

    def test_step(self, data: Union[dict, tuple, list])->list:
        return super().test_step(data=data)
    
    def bbox_to_cxcywh_format(self, bbox, img_shape):  # change bbox format xyxy to cycywh normalized format
        H, W = img_shape
        x, y, x2, y2 = bbox
        cx = (((x2 - x) / 2.0) + x) / W
        cy = (((y2 - y) / 2.0) + y) / H
        w = (x2 - x) / W
        h = (y2 - y) / H
        return [cx, cy, w, h]
    
    def change_gt_format(self, bboxes, img_shape):
        num_box = bboxes.shape[0]
        converted_bboxes = torch.zeros_like(bboxes)
        for i in range(num_box):
            converted_bboxes[i] = torch.tensor(self.bbox_to_cxcywh_format(bboxes[i], img_shape) , device=bboxes.device)
        return converted_bboxes
    
    def sort_ground_truth(self, data_samples):
        targets = [{"labels" : det_data_sample.gt_instances.labels, 
                    "boxes" : self.change_gt_format(det_data_sample.gt_instances.bboxes, det_data_sample.img_shape)} for det_data_sample in data_samples]
        return targets
    
    def format_postprocessed(self, outs, datasamples, gt_mask=None, pred_mask=None):
        formated_pred = []
        for i, out in enumerate(outs):
            datasample  = datasamples[i]
            H, W        = datasample.ori_shape
            h, w        = datasample.img_shape
            datasample.gt_instances.bboxes[..., [0, 2]] *= (W/w)
            datasample.gt_instances.bboxes[..., [1, 3]] *= (H/h)
            pred        = InstanceData()
            pred.bboxes = out['boxes']
            pred.scores = out['scores']
            pred.labels = out['labels']
            if gt_mask is not None and pred_mask is not None:
                datasample.gt_mask   = gt_mask[i]
                datasample.pred_mask = pred_mask[i]
            datasample.pred_instances = pred
            formated_pred.append(datasample)
        return formated_pred
           
    def forward(self, inputs, data_samples = None, gt_masks = None, mode: str = 'tensor'):
        orig_img_size = torch.tensor(data_samples[0].ori_shape, device=inputs.device) if data_samples else inputs[0].shape[2:]
        targets = self.sort_ground_truth(data_samples=data_samples) if data_samples else None

        if isinstance(self.backbone, PrunedBackbone):
            out, prune_mask_logit = self.backbone(inputs)
            if not self.training and mode=="predict": # save sample mask during val loop
                if self.iter % 50 == 0:
                        pred_mask = torch.sigmoid(prune_mask_logit)
                        pred_mask = (pred_mask >= 0.5).long()
                        self.overlay_mask_on_image(mask_tensor=pred_mask[0], image_tensor=inputs[0], save_path=str(self.iter) + ".png")
                self.iter +=1
        else:
            out = self.backbone(inputs)
            prune_mask_logit = None

        out = self.encoder(out)
        out = self.decoder(out, targets) # output normalized cxcywh
      
        if mode == "loss":    
            out = self.loss_fn(out, targets)
            if isinstance(self.backbone, PrunedBackbone):
                class_loss = self.mask_fn(prune_mask_logit, gt_masks)
                out["back_fore_class_loss"] = class_loss
            return out
        if mode == "predict":
            out = self.post_processor(out, orig_img_size)
            out = self.format_postprocessed(outs=out, datasamples=data_samples, gt_mask=gt_masks, pred_mask=prune_mask_logit) if data_samples else out
            return out
        if mode == "tensor":
            return out
        
    def draw_bboxes_on_image(self, image, pred_bboxes=None, gt_bboxes=None, 
                         pred_labels=None, gt_labels=None, 
                         pred_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=2):
       
        # Convert tensor to NumPy and transpose to (H, W, 3)
        image = image.permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 3]
        H, W, _ = image.shape  # Get image dimensions

        # Convert from float [0,1] to uint8 [0,255] if needed
        if image.dtype == np.float32 or image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        def draw_boxes(bboxes, labels, color):
            """Helper function to draw bounding boxes with labels."""
            if bboxes is None:
                return
            for i, box in enumerate(bboxes):
                cx, cy, w, h = box.tolist()

                # Convert normalized to absolute pixel values
                x_min = int((cx - w / 2) * W)
                y_min = int((cy - h / 2) * H)
                x_max = int((cx + w / 2) * W)
                y_max = int((cy + h / 2) * H)

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

                # Draw label text if provided
                if labels:
                    label_text = str(labels[i])
                    cv2.putText(image, label_text, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Draw predictions (green)
        draw_boxes(pred_bboxes, pred_labels, pred_color)

        # Draw ground truth (blue)
        draw_boxes(gt_bboxes, gt_labels, gt_color)

        # Convert BGR back to RGB for Matplotlib display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # save to disc
        cv2.imwrite("visualization/test_ground.png", image)

    def draw_bboxes_on_image_(self, image, bboxes, labels=None, color=(0, 255, 0), thickness=2):
        
        # Convert tensor to NumPy and transpose to (H, W, 3)
        image = image.permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 3]

        # Convert from float [0,1] to uint8 [0,255] if needed
        if image.dtype == np.float32 or image.max() <= 1.0:
            image = (image).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes
        for i, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = map(int, box.tolist())  # Convert tensor to int
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Draw label text if provided
            if labels:
                label_text = str(labels[i])
                cv2.putText(image, label_text, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert BGR back to RGB for Matplotlib display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image using Matplotlib
        cv2.imwrite("visualization/test_pred.png", image)
    
    def overlay_mask_on_image(self, mask_tensor, image_tensor, save_path="overlay_mask.png"):
        mask_tensor = mask_tensor.detach().cpu()
        image_tensor = image_tensor.detach().cpu()

        image_np = image_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        _, h, w = image_tensor.shape
        mask_resized = TF.resize(mask_tensor.float(), [h, w], interpolation=TF.InterpolationMode.NEAREST)
        mask_np = mask_resized.squeeze().byte().numpy()  # [H, W]

        bw_mask = np.stack([mask_np * 255] * 3, axis=-1).astype(np.uint8)  # [H, W, 3]

        # Convert to same format
        if image_np.shape[2] == 1:  # grayscale image
            image_np = np.repeat(image_np, 3, axis=2)

        # Blend: image * 0.7 + mask * 0.3
        overlay = cv2.addWeighted(image_np, 0.7, bw_mask, 0.3, 0)
        
        # creat dir if it does not exist
        os.makedirs("pruned_mask", exist_ok=True)

        # Save (no conversion to BGR unless needed)
        cv2.imwrite(os.path.join("pruned_mask", save_path), overlay)

    def parse_losses(self, losses):
        loss = sum(losses.values())
        losses["total loss"] = loss
        return loss, losses
    
    def _run_forward(self, data, mode):
        return super()._run_forward(data=data, mode=mode)
    
    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
 