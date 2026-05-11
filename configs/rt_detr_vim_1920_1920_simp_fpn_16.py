_base_ = "rt_detr_presnet_640_640.py"

# input shape
shape = (3, 1920, 1920)
eval_shape = [1920, 1920]

# model
model = dict(
    backbone=dict(
        _delete_ = True, 
        type="ViMDet", 
        img_size = eval_shape,
        patch_size = [16, 16],
        stride = 16,
        depth = 12, 
        d_model = 256,
        d_state = 16,
        if_simple_fpn = True,
        if_efficient_fpn = False, 
        scale_factors = [2, 1, 0.5],
        out_channels = [512, 1024, 2048]),
    encoder = dict(
        feat_strides = [8, 16, 32],
        in_channels  = [512, 1024, 2048],
        hidden_dim   = 256,
        eval_spatial_size = eval_shape),
    decoder = dict(
        feat_channels = [256, 256, 256],
        feat_strides  = [8, 16, 32],
        hidden_dim = 256,
        num_levels = 3,
        num_denoising = 100, 
        eval_spatial_size = eval_shape))

# train, val and test pipeline
resize  = dict(type="FixShapeResize", width=eval_shape[1], height=eval_shape[0], keep_ratio=False)
pack    = dict(type="PackDetInputs")

train_pipeline = _base_.train_pipeline_ + [resize, pack]
val_pipeline   = train_pipeline
test_pipeline  = train_pipeline

# dataset loaders
train_dataloader = dict(batch_size = 8, dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(batch_size = 8, dataset=dict(pipeline=val_pipeline))
test_dataloader  = dict(batch_size = 8, dataset=dict(pipeline=test_pipeline))

# train, val and test loops
train_cfg = dict(max_epochs=100)

# optimizer  
optim_wrapper = dict(
    type = "OptimWrapper",
    optimizer = dict(
        type="AdamW",
        lr  = 0.0001, 
        weight_decay = 0.0001),
        clip_grad = None)

# checkpoint load
resume    = True
load_from = None
              
# related to logging and visualization
work_dir   = _base_.work_dir_ + "/rt_detr_vim_1920_1920_sim_fpn_16_v1" # logges and checkpoints saved here
save_dir   = _base_.save_dir_ + "/rt_detr_vim_1920_1920_sim_fpn_16_v1" # visualization will be saved here
visualizer = dict(save_dir = save_dir)
