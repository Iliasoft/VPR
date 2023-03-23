
import os
import albumentations as A
abs_path = os.path.dirname(__file__)


args = {
    'model_path':'../models/',
    'data_path':'e:/',
    'data_path_2019':'e:/',
    'valid_csv_fn':'Train/recognition_solution_v2.1.csv',
    'train_csv_fn':'Train/train.csv',

    'gpus':'0',
    'filter_warnings':False,
    'logger': 'neptune',
    'num_sanity_val_steps': 0,

    'distributed_backend': None,
    'channels_last':False,

    'gradient_accumulation_steps':4,
    'precision':16,
    'sync_batchnorm':False,
    
    'seed':1138,
    'num_workers':2,
    'save_weights_only':False,

    'resume_from_checkpoint': None,#"../models/config2/ckpt/epoch=10.ckpt",
    'pretrained_weights': None,
    'model_weights_file_name': "../models/config2/config1_ckpt_10.pth",# IE added

    'normalization':'imagenet',
    'crop_size':448,

    'backbone':'gluon_seresnext101_32x4d',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.3,

    'p_trainable': True,

    'neck': 'option-D',
    'head':'arc_margin',

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
    'batch_size': 9,
    'max_epochs': 10,
    'scheduler': {"method":"cosine","warmup_epochs": 1},

    'n_classes':81313,
    'data_frac':1.,

    'neptune_project': 'ieldarov/VPR',
    'neptune_api_token': None

}

args['tr_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.RandomCrop(height=args['crop_size'], width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(512),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
