_base_ = "rt_detr_presnet_640_640.py"

# input shape
shape = (3, 1920, 1920)
eval_shape = [1920, 1920]

# model
model = dict(
    backbone = dict(depth = 50),
    encoder=dict(eval_spatial_size = eval_shape),
    decoder=dict(eval_spatial_size = eval_shape))

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
work_dir = _base_.work_dir_ + "/rtdetr_presnet50_1920_1920_v1" # logges and checkpoints saved here
save_dir = _base_.save_dir_ + "/rtdetr_present50_1920_1920_v1" # visualization will be saved here
visualizer = dict(save_dir= save_dir)