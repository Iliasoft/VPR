
import os
import albumentations as A
import cv2
import numpy as np

abs_path = os.path.dirname(__file__)

args = {
    'model_path': '../models/',
    'data_path': 'e:/',
    'data_path_2019': 'e:/',
    'valid_csv_fn': 'Train/recognition_solution_v2.1.csv',
    'train_csv_fn': 'Train/train.csv',
    
    'filter_warnings':True,
    'logger': None,
    'num_sanity_val_steps': 0,

    'gpus':'0',
    'distributed_backend': None,
    'sync_batchnorm': True,

    'gradient_accumulation_steps': 4,
    'precision':16,
    'hardmining': False,
    'seed':1337,
    
    'drop_last_n': 0,
    'save_weights_only': False,
    'resume_from_checkpoint': "../models/config5/ckpt/last.ckpt",
    'model_weights_file_name': "../models/config5/config5_ckpt_12.pth",  # IE added

    'p_trainable': True,

    'normalization':'imagenet',

    'backbone':'tf_efficientnet_b3_ns',
    'embedding_size': 512,
    'pool': "gem",
    'arcface_s': 45,
    'arcface_m': 0.35,

    'head': 'arc_margin',
    'neck': 'option-D',

    'loss': 'arcface',
    'crit': "bce",
   # 'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm': "batch",
    
    'optimizer': "sgd",
    'lr': 0.01,# it was 0.05
    'weight_decay': 1e-4,
    'batch_size': 10,# it was 13 and originally 8
    'max_epochs': 12,# it was 10

    'scheduler': {"method": "cosine", "warmup_epochs": 1},
    
    'pretrained_weights': None,

    'n_classes':81313,
    'data_frac':1.,
    
    'num_workers': 4,
    'crop_size': 600,

    'neptune_project':'ieldarov/VPR',
    'neptune_api_token':'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwN2E3YmRmNS05ZmUxLTQ2M2YtYTcxOS1lOGIzMzZlYmMxMDUifQ=='
}

args['tr_aug'] = A.Compose([A.LongestMaxSize(664,p=1),
                            A.PadIfNeeded(664, 664, border_mode=cv2.BORDER_CONSTANT, p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']), 
                            A.HorizontalFlip(always_apply=False, p=0.5), 
                           ],
                            p=1.0
                            )

args['val_aug'] = A.Compose([A.LongestMaxSize(664, p=1),
                             A.PadIfNeeded(664, 664, border_mode=cv2.BORDER_CONSTANT, p=1),
                             A.CenterCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']),
                            ], 
                            p=1.0
                            )

