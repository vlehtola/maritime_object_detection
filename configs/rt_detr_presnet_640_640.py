_base_ = "./_base_/default_runtime.py"

# input shape 
shape      = (3, 640, 640) # for flop counter
eval_shape = [640, 640]

# model config
model = dict(
    type = "RTDETR",
    data_preprocessor = dict(type = "DetDataPreprocessor"), 
    backbone = dict(
        type = "PResNet",
        depth = 50,
        variant= "d",
        freeze_at = 0,
        return_idx = [1, 2, 3],
        num_stages = 4,
        freeze_norm = True,
        pretrained =  True),
    encoder  = dict(
        type = "HybridEncoder",
        attn_module = dict(
            type = "MultiheadAttention",
            d_model = 256,
            num_heads  = 8,
            dropout = 0.5),
        fpn_module = dict(
            type = "PyramidFeatFusion",
            num_feat_map = 3,
            hidden_dim = 256, 
            act = "relu",
            depth_multi = 1.0,
            expansion = 1.0),
        in_channels  = [512, 1024, 2048],
        feat_strides = [8, 16, 32],      
        hidden_dim =256,                 
        use_encoder_idx = [2],
        num_encoder_layers = 1,
        dim_feedforward = 1024,
        dropout = 0,
        enc_act = 'gelu',
        pe_temperature = 10000,
        eval_spatial_size = eval_shape),   
    decoder  = dict(
        type = "RTDETRTransformer",
        self_attn = dict(
            type = "MultiheadAttention",
            d_model = 256,
            num_heads = 8,
            dropout  = 0.5),
        cross_attn = dict(
            type = "DeformableAttention",
            embed_dim  = 256,
            num_heads = 8,
            num_levels = 3,
            num_points = 4),
        num_classes = 10,
        feat_channels = [256, 256, 256],    
        feat_strides  = [8, 16, 32],        
        hidden_dim = 256,                   
        num_levels = 3,                    
        num_queries = 300,
        num_decoder_layers = 6, 
        num_denoising = 100,                 
        eval_idx = -1,
        eval_spatial_size = eval_shape),     
    post_processor = dict(
        type = "RTDETRPostProcessor",
        num_classes = 10,
        num_top_queries = 300,
        use_focal_loss = True),
    loss_fn = dict(
        type   = "SetCriterion",
        num_classes = 10,
        losses = ['vfl', 'boxes'],
        weight_dict = dict(loss_vfl= 1, loss_bbox= 5, loss_giou= 2),
        alpha  = 0.75,
        gamma  = 2.0,
        matcher = dict(
            type = "HungarianMatcher",
            weight_dict = dict(cost_class = 2, cost_bbox = 5, cost_giou = 2),
            alpha = 0.25,
            gamma = 2.0)))

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
work_dir = _base_.work_dir_ + "/rtdetr_presnet50_640_640_v0"      # logges and checkpoints saved here
save_dir = _base_.save_dir_ + "/rtdetr_present50_640_640_v0"      # visualization will be saved here
visualizer = dict(save_dir= save_dir)