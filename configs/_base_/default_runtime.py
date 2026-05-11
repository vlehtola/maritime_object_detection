# default runtime configs
default_scope = "mmdet"
env_cfg       = dict(
    cudnn_benchmark=False, 
    mp_cfg=dict( 
        mp_start_method='fork',  
        opencv_num_threads=0), 
    dist_cfg=dict(backend='nccl'))
launcher      = None
randomness    = dict(seed = 42, deterministic = True)

# used to register custom class definitions
custom_imports = dict(
    imports=[
        "datasets",
        "models"], 
    allow_failed_imports = False)

# root path to dataset relative to python system pwd
data_root_singapore  = "../datasets/Singapore_Maritime/"

# dataset, transforms and dataloaders
train_pipeline_ = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False, poly2mask=False)]

val_pipeline   = train_pipeline_
test_pipeline  = train_pipeline_

# datasets for train, val and test
singapore_dataset_train = dict(
    type        = "SingaporeOrig",
    ann_file    = data_root_singapore + "train/_annotations.coco.json",
    data_prefix = dict(img=data_root_singapore + "train"),
    pipeline    = train_pipeline_)

singapore_dataset_val = dict(
    type        = "SingaporeOrig",
    ann_file    = data_root_singapore + "valid/_annotations.coco.json",
    data_prefix = dict(img=data_root_singapore + "valid"),
    pipeline    = val_pipeline)

singapore_dataset_test = dict(
    type        = "SingaporeOrig",
    ann_file    = data_root_singapore + "test/_annotations.coco.json",
    data_prefix = dict(img = data_root_singapore + "test"),
    pipeline    = test_pipeline)

# train, validation and test dataloaders
train_dataloader = dict(
    batch_size = 2, 
    num_workers= 1,
    dataset = singapore_dataset_train)

val_dataloader   = dict(
    batch_size = 2, 
    num_workers= 1,
    dataset = singapore_dataset_val)

test_dataloader = dict(
    batch_size = 2,
    num_workers =1,
    dataset = singapore_dataset_test)

# train, validation and test loops
train_cfg  = dict(type = "EpochBasedTrainLoop", max_epochs = 50, val_interval = 1)
val_cfg    = dict(type = "ValLoop")
test_cfg   = dict(type = "TestLoop")

# val and test meterics and evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file = data_root_singapore + "valid/_annotations.coco.json",
    metric='bbox',
    classwise = True,
    format_only=False,
    backend_args= None)

test_evaluator = dict(
    type = "CocoMetric",
    ann_file = data_root_singapore + "test/_annotations.coco.json",
    classwise = True,
    format_only=False,
    backend_args= None)

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  
    logger=dict(type='LoggerHook', interval=1),  
    param_scheduler=dict(type='ParamSchedulerHook'), 
    checkpoint=dict(type='CheckpointHook', interval=5), 
    sampler_seed=dict(type='DistSamplerSeedHook'),  
    visualization=dict(type='DetVisualizationHook', draw=True, interval=50)) 

# related to logger 
log_out_dir     = "logs"              # created in the under the work_dir
log_level       = 'INFO'              # log level 
backend_args    =  None               # args for initializing the backend

# outputs
work_dir      = "experiment_output"
work_dir_     = work_dir
save_dir_     = "visualization"                                                        # relative to python system working directory
vis_backends  = [dict(type='LocalVisBackend'),  dict(type='TensorboardVisBackend'),] 
visualizer    = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer',  save_dir=save_dir_)
log_processor = dict(type='LogProcessor',  window_size=100, by_epoch=True) 