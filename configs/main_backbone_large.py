_base_ = "rt_detr_presnet_640_640.py"

# input shape
shape      = (3, 1920, 1920)
eval_shape = [1920, 1920]

# model
model = dict(
    data_preprocessor = dict(type = "CustomDetDataPreprocessor"), 
    backbone=dict(
        _delete_ = True, 
        type="PrunedBackbone", 
        d_model = 256,
        d_state = 16,
        main_backbone = dict(
            type = "VSSDDet",
            depth = 22,
            d_model = 256,
            d_state = 16,
            num_heads = 2, 
            if_patch_embed = False,
            if_simple_fpn  = False,
            if_efficient_fpn = False),
        # =====================
        img_size = eval_shape,
        patch_size = [32, 32], 
        stride = 32,
        # =====================
        pre_backbone  = dict(
            type = "VSSDDet",
            depth = 2,   
            d_model = 256,
            d_state = 16,
            num_heads = 2,
            if_patch_embed = False,
            if_simple_fpn  = False,
            if_efficient_fpn = False),
        ba_class_ratio  = 2,
        ba_class_layers = 2,
        pruning_ratio   = 0.5,
        # =====================
        fpn_mixer = "ViMBlock",
        if_simple_fpn = False,
        if_efficient_fpn = True, 
        if_upsample_fpn  = False,
        eff_scale_factor = 2, 
        scale_factors  = None,
        out_channels = [512, 1024, 2048]),
    encoder = dict(
        feat_strides = [32, 64, 128],
        in_channels  = [512, 1024, 2048],
        hidden_dim   = 256,
        eval_spatial_size = eval_shape),
    decoder = dict(
        feat_channels = [256, 256, 256],
        feat_strides  = [32, 64, 128],
        hidden_dim = 256,
        num_levels = 3,
        num_denoising = 100, 
        eval_spatial_size = eval_shape),
    mask_fn = dict(type="BackgroundForewardClassLoss"))

# train, val and test pipeline
resize    = dict(type="FixShapeResize",  height=eval_shape[0], width=eval_shape[1], keep_ratio=False)
load_mask = dict(type="LoadMask",  height=eval_shape[0],  width=eval_shape[1], patch_size = 32)
pack      = dict(type="CustomPackDetInputs")

train_pipeline = _base_.train_pipeline_ + [resize, load_mask, pack]
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

# meterics
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file = _base_.data_root_singapore + "valid/_annotations.coco.json",
        metric='bbox',
        classwise = True,
        format_only=False,
        backend_args= None),
    dict(type="PatchClassificationMetrics"),
    dict(type ="BoxPatchClassificationMetrics", patch_size=32)]

test_evaluator = [
    dict(
        type = "CocoMetric",
        ann_file = _base_.data_root_singapore + "test/_annotations.coco.json",
        classwise = True,
        format_only=False,
        backend_args= None),
    dict(type ="PatchClassificationMetrics"),
    dict(type ="BoxPatchClassificationMetrics", patch_size=32)]

# checkpoint load
resume    = True
load_from = None

# related to logging and visualization
work_dir   = _base_.work_dir_ + "/main_backbone_large_v1" # logges and checkpoints saved here
save_dir   = _base_.save_dir_ + "/main_backbone_large_v1" # visualization will be saved here
visualizer = dict(save_dir = save_dir)