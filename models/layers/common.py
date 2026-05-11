import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from mmdet.registry import MODELS

from ..utils.rtdetr_utils import get_activation, segm_init_weights

# from https://github.com/hustvl/Vim?tab=readme-ov-file
@ MODELS.register_module()
class PatchEmbed(nn.Module):
    def __init__(self, img_size=[640, 640], patch_size=[16, 16], stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = ((img_size[0]- patch_size[0]) // stride+1, (img_size[1] - patch_size[1]) // stride +1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H==self.img_size[0] and W== self.img_size[1], \
             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x 

# from https://github.com/lyuwenyu/RT-DETR
# conv -> batch norm -> act
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act  = nn.Identity() if act is None else get_activation(act=act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# from https://github.com/lyuwenyu/RT-DETR
# output = ConvNormLayer(input) + ConvNormLayer(input)
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

# from https://github.com/lyuwenyu/RT-DETR
# output = ConvNormLayer(ConvNormLayer(input))
class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[ RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

# from https://github.com/lyuwenyu/RT-DETR
# output= Linear(input with input_dim -> output with output_dim)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# depth wise convolution 
class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, act=None, norm=True):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=False)
        self.point_wise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm       = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act        = get_activation(act) if act else nn.Identity()

        # initialization
        segm_init_weights(self.depth_conv)
        segm_init_weights(self.point_wise)
        segm_init_weights(self.norm)
        
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_wise(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
# from https://github.com/hustvl/Vim?tab=readme-ov-file
@MODELS.register_module()
class SimpleFeaturePyramid(nn.Module):
    def __init__(
        self,
        input_dim,      # the number of input channels
        out_channels,   # list of channels the output feature map should have 
        scale_factors): # list of scales to downsample input feature map
     
        super().__init__()
        assert len(out_channels) == len(scale_factors), "The number of out_channels should be the same as the number of scale factors"
        self.scale_factors = scale_factors
        dim = input_dim
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.BatchNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
          
            layers.extend([ConvNormLayer(out_dim, out_channels[idx], kernel_size=1, stride=1,  bias=None),
                           ConvNormLayer(out_channels[idx], out_channels[idx], kernel_size=3, stride=1, padding=1,bias= None)])
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
        
        # initialization
        self.stages.apply(segm_init_weights)

    def forward(self, x):
        features = x
        results = []
        for stage in self.stages:
            results.append(stage(features))
        return results

@MODELS.register_module()
class EfficientFeaturePyramid(nn.Module):
    def __init__(self, in_channels, out_channels, mixer_cls, strides=[1, 2, 2], upsample=False, scale_factor = 2):
        super().__init__()
        assert len(out_channels) >= 2, "The number of feature pyramid levels must be at least 2."
        
        # upsampling 
        self.upsample = upsample
        if upsample:
            in_channels = in_channels // 4
            self.feat_upsample = nn.ConvTranspose2d(int(4*in_channels), in_channels, kernel_size=2, stride=scale_factor)

        # mixer layers (ssm)
        first_mixer       = mixer_cls(d_model = in_channels)
        self.mixer_layers = nn.ModuleList([mixer_cls(d_model=out_channel) for out_channel in out_channels[:-1]])
        self.mixer_layers.insert(0, first_mixer)

        # Depthwise convolution downsampling layers
        self.dw_layers = nn.ModuleList()
        in_channel = in_channels 
        for i, channel in enumerate(out_channels):  
            self.dw_layers.append(DWConvBlock(in_channels=in_channel, out_channels=channel, kernel_size=3, stride=strides[i]))
            in_channel = channel

        # initialization
        if upsample:
            self.feat_upsample.apply(segm_init_weights)    
      
    def forward(self, x):
        if self.upsample: 
            x = self.feat_upsample(x)
        out = []
        for i, layer in enumerate(self.mixer_layers):
            N, _, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)
            x = layer(x)
            x = x.permute(0, 2, 1).reshape(N, -1, H, W)
            x = self.dw_layers[i](x)
            out.append(x)
        return out

# from https://github.com/lyuwenyu/RT-DETR
@MODELS.register_module() 
class PyramidFeatFusion(nn.Module):
    def __init__(self, num_feat_map, hidden_dim = 256, act="relu", depth_multi = 1.0, expansion=1.0):
        super().__init__()
        self.num_feat_map = num_feat_map
        self.hidden_dem   = hidden_dim

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks    = nn.ModuleList()
        for _ in range(num_feat_map-1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_multi), act=act, expansion=expansion))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(num_feat_map - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_multi), act=act, expansion = expansion))
        
    def forward(self, feats):
        assert self.num_feat_map == len(feats), "number input feat maps mismatch"

        proj_feats = feats
        inner_outs = [proj_feats[-1]]
        for idx in range(self.num_feat_map - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low  = proj_feats[idx - 1]
            feat_high = self.lateral_convs[self.num_feat_map - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[self.num_feat_map-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(self.num_feat_map - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        return outs