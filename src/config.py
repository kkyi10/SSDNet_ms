
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
    config.enable_modelarts= False
    # Url for modelarts
    # config.data_url= ""
    # config.train_url= ""
    # config.checkpoint_url= ""
    # Path for local
    config.run_distribute= False
    config.enable_profiling= False
    config.data_path= "/media/su/sdisk/mindspore/ssd/mini_dataset/"
    config.output_path= "./cache/train"
    config.load_path= "./cache/checkpoint_path/"
    config.device_target= "GPU"
    config.checkpoint_path= "./checkpoint/"
    config.checkpoint_file_path= "ssd-500_458.ckpt"

    # ==============================================================================
    # Training options
    config.model_name= "ssd_vgg16"
    config.img_shape= [300, 300]
    config.num_ssd_boxes= 7308
    config.match_threshold= 0.5
    config.nms_threshold= 0.6
    config.min_score= 0.1
    config.max_boxes= 100
    config.all_reduce_fusion_config= []
    config.use_float16= False

    # learing rate settings
    config.lr_init= 0.001
    config.lr_end_rate= 0.001
    config.warmup_epochs= 2
    config.momentum= 0.9
    config.weight_decay= 0.00015
    config.ssd_vgg_bn= False
    config.pretrain_vgg_bn= False

    # network
    config.num_default= [3, 6, 6, 6, 6, 6]
    config.extras_in_channels= [256, 512, 1024, 512, 256, 256]
    config.extras_out_channels= [512, 1024, 512, 256, 256, 256]
    config.extras_strides= [1, 1, 2, 2, 2, 2]
    config.extras_ratio= [0.2, 0.2, 0.2, 0.25, 0.5, 0.25]
    config.feature_size= [38, 19, 10, 5, 3, 1]
    config.min_scale= 0.1
    config.max_scale= 0.95
    config.aspect_ratios= [[], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    config.steps= [8, 16, 32, 64, 100, 300]
    config.prior_scaling= [0.1, 0.2]
    config.gamma= 2.0
    config.alpha= 0.75

    config.dataset= "coco"
    config.lr= 0.05
    config.mode_sink= "sink"
    config.device_id= 0
    config.device_num= 1
    config.epoch_size= 60
    config.batch_size= 5
    config.loss_scale= 1024
    config.pre_trained= ""
    config.pre_trained_epoch_size= 0
    config.save_checkpoint_epochs= 10
    config.only_create_dataset= False
    config.eval_start_epoch= 3
    config.eval_interval= 2
    config.run_eval= True
    config.filter_weight= False
    config.freeze_layer= None
    config.save_best_ckpt= True

    # config.result_path= ""
    # config.img_path= ""
    config.drop= False

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    # config.feature_extractor_base_param= ""
    config.checkpoint_filter_list= ['multi_loc_layers', 'multi_cls_layers']
    config.mindrecord_dir= "MindRecord_COCO"
    config.coco_root= "/media/su/sdisk/mindspore/ssd/mini_dataset/"
    config.train_data_type= "train2017"
    config.val_data_type= "val2017"
    config.instances_set= "annotations/instances_{}.json"

    # The annotation.json position of voc validation dataset.
    config.voc_json= "annotations/voc_instances_val.json"
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
    # voc original dataset.
    config.voc_root= "/data/voc_dataset"
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    # config.image_dir= ""
    # config.anno_path= ""
    config.file_name= "ssd"
    config.file_format= 'MINDIR'

    return config




