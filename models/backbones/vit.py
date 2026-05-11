from functools import partial

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from timm.layers import  trunc_normal_

from ..layers.common import PatchEmbed, SimpleFeaturePyramid, EfficientFeaturePyramid
from ..layers.attention_layers import ViTBlock, ViMBlock
from ..utils.rtdetr_utils import _init_weights, segm_init_weights

@MODELS.register_module()
class ViTDet(nn.Module):
    def __init__(
            self,
            img_size=[640, 640],
            in_channels=3,
            patch_size=[16, 16],  
            stride=16,
            if_patch_embed = True,
            # =================
            depth=12, 
            d_model=256, 
            d_state = 16,
            num_heads = 8,
            drop_pose_rate=0.5,
            # =================
            if_simple_fpn = False,
            if_efficient_fpn = True,
            if_upsample_fpn  = False,
            eff_scale_factor = 2, 
            scale_factors = None,
            out_channels = [512, 1024, 2048],
            # =================
            initializer_cfg=None):
        super().__init__()
        self.if_simple_fpn  = if_simple_fpn
        self.if_efficient_fpn = if_efficient_fpn
        self.if_patch_embed = if_patch_embed

        if self.if_patch_embed:
            # patch embedding
            self.patch_embedding = PatchEmbed(img_size=img_size, 
                                              patch_size=patch_size, 
                                              stride=stride, 
                                              in_chans=in_channels, 
                                              embed_dim=d_model)
            num_patches = self.patch_embedding.num_patches
            self.grid_size = self.patch_embedding.grid_size

            # position encoding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model)) # learnable abs pos 
            self.pos_drop  = nn.Dropout(p=drop_pose_rate)

            # weight initialization
            self.patch_embedding.apply(segm_init_weights)
            trunc_normal_(self.pos_embed, std=.02)
        
        # mamba blocks
        self.layers = nn.ModuleList([ViTBlock(d_model=d_model, num_heads=num_heads) for i in range(depth)])
        
        # feature pyramid
        if self.if_simple_fpn:
            self.simp_fpn = SimpleFeaturePyramid(input_dim=d_model, 
                                                 out_channels=out_channels, 
                                                 scale_factors=scale_factors)
        
        if self.if_efficient_fpn:
            self.eff_fpn  = EfficientFeaturePyramid(in_channels=d_model, 
                                                    upsample= if_upsample_fpn, 
                                                    scale_factor= eff_scale_factor, 
                                                    out_channels=out_channels,  
                                                    mixer_cls=partial(ViMBlock, d_state = d_state, drop_prob = 0.5))
                        
        # weight initialization
        self.apply(partial(_init_weights, n_layer=depth, **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, x):
        if self.if_patch_embed:
            B, _, _, _ = x.shape
            x = self.patch_embedding(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for layer in self.layers:
              hidden_states = layer(x)
        
        if self.if_simple_fpn:
             output = hidden_states.permute(0, 2, 1).reshape(B, -1, *self.grid_size)
             output = self.simp_fpn(output)
             return output

        if self.if_efficient_fpn:
             output = hidden_states.permute(0, 2, 1).reshape(B, -1, *self.grid_size)
             output = self.eff_fpn(output)
             return output
        
        return hidden_states