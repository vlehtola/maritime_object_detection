import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from timm.layers import DropPath
from mmdet.registry import MODELS

from ..utils.rtdetr_utils import deformable_attention_core_func
from ..layers.common import MLP

@MODELS.register_module()
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, batch_first=True):
        super().__init__()
        self.attn =  nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=batch_first)

    def forward(self, q, k, value, src_mask = None):
        return self.attn(q, k, value, attn_mask = src_mask)


# from https://github.com/lyuwenyu/RT-DETR  
@MODELS.register_module()
class DeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func
        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        # proj
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
    
@MODELS.register_module()
class ViMBlock(nn.Module):
    def __init__(self, d_model = 256, d_state = 12, layer_idx = None, bimamba_type = "v2", norm_layer_cls = nn.LayerNorm, drop_prob = 0.):
        super().__init__()
        self.mamba_block = Mamba(d_model=d_model, d_state=d_state, bimamba_type=bimamba_type, layer_idx=layer_idx)
        self.norm        = norm_layer_cls(normalized_shape=d_model)
        self.drop_path   = DropPath(drop_prob=drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.layer_idx   = layer_idx
     
    def forward(self, hidden_states, inference_param=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba_block(hidden_states, inference_param)
        hidden_states = self.drop_path(residual) + hidden_states
        return hidden_states

# from https://github.com/YuHengsss/VSSD/blob/main/classification/models/mamba2.py
@MODELS.register_module()
class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,          # ========>
        d_state = 64,     # ========>
        headdim=64,       # ========>
        ngroups=1,        # ========>
        expand=2,  
        d_conv=3, 
        conv_init=None,
        conv_bias=True,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        device=None,
        dtype=None,
        **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv  = d_conv
        self.conv_init = conv_init
        self.expand  = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0, f"d_inner {self.d_inner} and self.headdim {self.headdim}"
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
      
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
        
        # order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv1d = nn.Conv1d(in_channels=conv_dim, 
                                out_channels=conv_dim, 
                                groups=conv_dim, 
                                bias=conv_bias, 
                                kernel_size=d_conv, 
                                padding=(d_conv - 1) // 2, **factory_kwargs)
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # output proj
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''
        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))
        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()
        return x


    def forward(self, u):
        zxbcdt = self.in_proj(u)    # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
       
        # 1D convolution
        xBC = xBC.permute(0, 2, 1).contiguous()
        xBC = self.act(self.conv1d(xBC))
        xBC = xBC.permute(0, 2, 1).contiguous()

        # split into 3 main branches: X, B, C
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = (x, dt, A, B, C)
        
        # linear attention
        y = self.non_casual_linear_attn(rearrange(x, "b l (h p) -> b l h p", p=self.headdim), dt, A, B, C, self.D)
        y = rearrange(y, "b l h p -> b l (h p)")

        # multiply "gate" branch and apply extra normalization layer
        y = self.norm(y)
        y = y*z
        out = self.out_proj(y)
        return out

class VSSDBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_state,
                 num_heads = 3, 
                 mlp_ratio=4, 
                 drop_path=0.,
                 act = "gelu", 
                 norm_layer = nn.LayerNorm,
                 ssd_expansion = 2,
                 ssd_ngroups = 1, 
                 **kwargs):
        super().__init__()
        self.mamba2 = Mamba2(d_model = d_model, 
                             d_state = d_state,
                             expand=ssd_expansion,
                             headdim= (d_model * ssd_expansion) // num_heads,
                             ngroups=ssd_ngroups,
                             **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(d_model)
        self.mlp   = MLP(input_dim=d_model, hidden_dim=int(d_model * mlp_ratio), output_dim=d_model, num_layers=2, act=act)
        self.norm2 = norm_layer(d_model)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mamba2(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.5, batch_first=True, mlp_ratio=4, act='gelu'):
        super().__init__()
        self.mult_head = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.norm1     = nn.LayerNorm(d_model)
        self.mlp       = MLP(input_dim=d_model, hidden_dim=int(d_model * mlp_ratio), output_dim=d_model, num_layers=2, act=act)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        skip = x
        x    = self.norm1(x)
        x, _ = self.mult_head(x, x, x)
        x    = skip + x

        skip = x
        x    = self.norm2(x)
        x    = self.mlp(x)
        return x


        