from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from timm.layers import trunc_normal_

from ..layers.common import PatchEmbed, MLP, SimpleFeaturePyramid, EfficientFeaturePyramid
from ..layers.attention_layers import ViMBlock, VSSDBlock
from ..utils.rtdetr_utils import segm_init_weights

@MODELS.register_module()
class PrunedBackbone(nn.Module):
    def __init__(self, 
                 main_backbone,
                 d_model,
                 d_state,
                 # =====================
                 img_size=[640, 640],
                 in_channels=3,
                 patch_size=[16, 16],  
                 stride=16,
                 drop_pose_rate = 0.5,
                 # =====================
                 pre_backbone = None,
                 ba_class_ratio = 2,
                 ba_class_layers = 2,
                 pruning_ratio = 0.2,
                 # =====================
                 if_simple_fpn = False,
                 if_upsample_fpn  = False,
                 if_efficient_fpn = True,
                 fpn_mixer = "ViMBlock",
                 eff_scale_factor = 2, 
                 scale_factors = [],
                 out_channels = [512, 1024, 2048]):
        super().__init__()
        self.if_simple_fpn    = if_simple_fpn
        self.if_efficient_fpn = if_efficient_fpn
        self.pruning_ratio    = pruning_ratio
        self.d_model          = d_model

        # patch embedding
        self.patch_embedding = PatchEmbed(img_size=img_size,  
                                          patch_size=patch_size, 
                                          stride=stride, 
                                          in_chans=in_channels, 
                                          embed_dim=d_model)
        self.num_patches     = self.patch_embedding.num_patches
        self.grid_size       = self.patch_embedding.grid_size

        # position encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model)) # learnable abs pos 
        self.pos_drop  = nn.Dropout(p=drop_pose_rate)

        # background/foreground classification
        self.pre_backbone  = MODELS.build(pre_backbone)
        self.bf_class_head = MLP(input_dim=d_model,  output_dim=1,  hidden_dim=int(d_model * ba_class_ratio),  num_layers=ba_class_layers)

        # backbone
        self.main_backbone = MODELS.build(main_backbone)

        # feature pyramid
        if self.if_simple_fpn:
               self.simp_fpn = SimpleFeaturePyramid(input_dim=d_model, 
                                                    out_channels=out_channels, 
                                                    scale_factors=scale_factors)
        
        if self.if_efficient_fpn:
               mixer_cls     = ViMBlock if fpn_mixer == "ViMBlock" else VSSDBlock
               self.eff_fpn  = EfficientFeaturePyramid(in_channels=d_model, 
                                                       upsample= if_upsample_fpn, 
                                                       scale_factor= eff_scale_factor, 
                                                       out_channels=out_channels, 
                                                       mixer_cls=partial(mixer_cls, d_state = d_state, drop_prob = 0.5))
             
        # weight initialization
        self.patch_embedding.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        self.bf_class_head.apply(segm_init_weights)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embedding(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # prune patches
        if self.pruning_ratio > 0.0:
            pre_prund = x
            x = self.pre_backbone(x)
            back_fore_logit = self.bf_class_head(x)
            _, topk_ind = torch.topk(back_fore_logit.squeeze(-1), int((1-self.pruning_ratio) * self.num_patches), dim=-1)
            x = torch.gather(x, dim=1, index=topk_ind.unsqueeze(-1).expand(-1, -1, self.d_model)) # foreground_patches 
            prune_mask_logit = back_fore_logit.permute(0, 2, 1).reshape(B, -1, *self.grid_size)

        x = self.main_backbone(x)
        
        # reactivate pruned patches
        if self.pruning_ratio > 0.0:
            x_reintegrated = pre_prund.clone()  
            assert x.shape == topk_ind.unsqueeze(-1).expand(-1, -1, self.d_model).shape, "scatter_ source and index shape mismatch!"
            x_reintegrated.scatter_(1, topk_ind.unsqueeze(-1).expand(-1, -1, self.d_model), x)  # reintegration
            x = x_reintegrated  

        if self.if_simple_fpn:
             output = x.permute(0, 2, 1).reshape(B, -1, *self.grid_size)
             output = self.simp_fpn(output)

        if self.if_efficient_fpn:
             output = x.permute(0, 2, 1).reshape(B, -1, *self.grid_size)
             output = self.eff_fpn(output)
        
        return output, prune_mask_logit