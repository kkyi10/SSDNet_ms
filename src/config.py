import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.data_path= "../datasets/"
    config.output_path= "./cache/train"
    config.load_path= "./cache/checkpoint_path/"
    config.checkpoint_path= "./checkpoint/"
    config.checkpoint_file_path = "./cache/train/ckpt_0/ssd-60_9.ckpt"


    # ==============================================================================
    # Training options
    config.img_shape= [300, 300]
    config.num_ssd_boxes= 8732

    config.match_threshold= 0.5
    config.nms_threshold= 0.6
    config.min_score= 0.1
    config.max_boxes= 100

    # learing rate settings
    config.lr_init= 0.001
    config.lr_end_rate= 0.001
    config.warmup_epochs= 2
    config.momentum= 0.9
    config.weight_decay= 0.00015
    config.device_target = 'GPU'
    config.run_eval = True
    # network
    config.num_default= [4, 6, 6, 6, 4, 4]
    config.extras_ratio= [0.2, 0.2, 0.2, 0.25, 0.5, 0.25]
    config.feature_size= [38, 19, 10, 5, 3, 1]
    config.min_scale= 0.1
    config.max_scale= 0.95
    config.aspect_ratios= [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    config.steps= [8, 16, 32, 64, 100, 300]
    config.prior_scaling= [0.1, 0.2]
    config.gamma= 2.0
    config.alpha= 0.75

    config.dataset= "coco"
    config.lr= 0.05
    config.device_id= 0
    # config.device_num= 1
    config.epoch_size= 60
    config.batch_size= 5
    config.loss_scale= 1024
    config.pre_trained_epoch_size= 0
    config.save_checkpoint_epochs= 10
    config.eval_start_epoch= 3
    config.eval_interval= 2

    config.mindrecord_dir= "MindRecord_COCO"
    config.coco_root= "../dataset/"
    config.train_data_type= "train2017"
    config.val_data_type= "val2017"
    config.instances_set= "annotations/instances_{}.json"

    config.classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
    config.num_classes= 81

    return config




