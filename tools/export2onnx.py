import argparse

import torch
import torch.nn as nn

from mmengine.config import Config
from mmdet.registry import MODELS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("ckpt", type=str, help="path to checkpoint path")
    parser.add_argument("--file_name", type=str, default="model.onnx", help="path to save directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg   = Config.fromfile(args.config)
    ckpt  = torch.load(args.ckpt, map_location="cuda")
    state = ckpt['state_dict']

    class Model(nn.Module):
        def __init__(self,):
            super().__init__()
            self.model = MODELS.build(cfg.model)
            self.model.post_processor.deploy()
            self.model.load_state_dict(state, strict=False)
        
        def forward(self, images):
            return self.model(images)
    
    model = Model()
    dynamic_axes = {'images': {0: 'N', }}

    data = torch.rand(1, 3, 640, 640)
    torch.onnx.export(
        model, 
        (data), 
        args.file_name,
        input_names=['images'],
        output_names=['labels', 'boxes'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False)


if __name__ == "__main__":
    main()