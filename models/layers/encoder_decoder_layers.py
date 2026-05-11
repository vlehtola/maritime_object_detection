"""
  Originally written by lyuwneyu @ https://github.com/lyuwenyu/RT-DETR
  Copied and modified here to make it part of MMDetection framework. 
  It also edited for modularity and works for custom attention later.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..utils import inverse_sigmoid, get_activation
from .attention_layers import MultiheadAttention

class EncoderLayer(nn.Module):
    def __init__(self, att_module, d_model, dim_feedforward = 2028, dropout = 0.1, act = None, normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.att_module       = att_module
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation(act) if act is not None else nn.Identity()

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def forward(self, src, src_mask = None, pos_embed= None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        if isinstance(self.att_module, MultiheadAttention):
            q  = k = self.with_pos_embed(src, pos_embed=pos_embed)
            src, _ = self.att_module(q, k, value=src, src_mask = src_mask)
        else:
            src = self.with_pos_embed(src, pos_embed=pos_embed)
            src = self.att_module(src) # for other type of attention modules.
        
        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        if not self.normalize_before:
            src = self.norm2(src)
        
        return src
    
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm if norm is not None else nn.Identity()

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        output = self.norm(output)

        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn_obj,  cross_atten_obj, dim_feedforward = 1024, activation = 'relu', dropout = 0.0):
        super().__init__()
        
        # self attention
        self.self_attn  = self_attn_obj
        self.dropout1   = nn.Dropout(dropout)
        self.norm1      = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = cross_atten_obj
        self.dropout2   = nn.Dropout(dropout)
        self.norm2      = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    
    def forward(self,
            tgt,       #  object queries
            reference_points,
            memory,    #  output from encoder
            memory_spatial_shapes,
            memory_level_start_index,
            attn_mask=None,
            memory_mask=None,
            query_pos_embed=None):
        
        # self attention
        if isinstance(self.self_attn, MultiheadAttention):
            q = k = self.with_pos_embed(tgt, query_pos_embed)
            tgt2, _ = self.self_attn(q, k, value=tgt, src_mask=attn_mask)
        else:
            tgt2 = self.self_attn(tgt)   # for ssm layers, 

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt
 
class Decoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
