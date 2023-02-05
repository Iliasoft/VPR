
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
    'filter_warnings':True,
    'logger': None,
    'num_sanity_val_steps': 50,

    'distributed_backend': None,

    'gradient_accumulation_steps':3,
    'precision':16,
    'sync_batchnorm':False,

    'seed':5553,
    'num_workers':25,
    'save_weights_only': False,
    'resume_from_checkpoint': "../models/config7/ckpt/last.ckpt",
    'pretrained_weights':None,
    'normalization':'imagenet',
    'crop_size':512,

    'backbone':'res2net101_26w_4s',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,
    'arcface_m': 0.4,

    'neck': 'option-D',
    'head':'arc_margin',
    'p_trainable':False,

    'crit': "bce",
    'loss':'arcface',
    #'focal_loss_gamma': 2,
    'class_weights': "log",
    'class_weights_norm' :'batch',
    
    'optimizer': "sgd",
    'weight_decay':1e-4,
    'lr': 0.05,
    'batch_size': 16,
    'max_epochs': 12,# it was 10
    'scheduler': {"method":"cosine","warmup_epochs": 1},
    
    'n_classes':81313,
    'data_frac':1.,

    'neptune_project':'ieldarov/VPR',
    'neptune_api_token':'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwN2E3YmRmNS05ZmUxLTQ2M2YtYTcxOS1lOGIzMzZlYmMxMDUifQ=='
}

args['tr_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.RandomCrop(height=args['crop_size'],width=args['crop_size'],p=1.),
    A.HorizontalFlip(p=0.5),
    ])

args['val_aug'] = A.Compose([A.Resize(height=544,width=672,p=1.),
    A.CenterCrop(height=args['crop_size'],width=args['crop_size'],p=1.)
])
