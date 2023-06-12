# this model is a derivation of model 9 (and therefore4) extended with OCR embeddings, trained from scratch to extended number of classes
import os
import albumentations as A
import cv2
import numpy as np

abs_path = os.path.dirname(__file__)
# this is a copy of models 9 and 4

args = {
    'model_path': '../models/',
    'data_path': 'e:/',
    'data_path_2019': 'e:/',
    'valid_csv_fn': 'Train/recognition_solution_v2.1_extended.csv',
    'train_csv_fn': 'Train/train_extended.csv',

    'filter_warnings': True,
    'logger': None,
    'num_sanity_val_steps': 0,

    'gpus': '0',
    'distributed_backend': None,
    'sync_batchnorm': True,

    'gradient_accumulation_steps': 4,
    'precision': 16,

    'seed': 1337,

    'drop_last_n': 0,
    'save_weights_only': False,
    'resume_from_checkpoint': "../models/config11/ckpt/last.ckpt",

    'text_embeddings_fn': "../models/config11/images_text_embeddings_ocr_only.pkl",  # IE added
    'text_embedding_size': 384,

    'p_trainable': True,

    'normalization': 'imagenet',
    'backbone': 'tf_efficientnet_b3_ns',
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
    'lr': 0.05,
    'weight_decay': 1e-4,
    'batch_size': 17,
    'max_epochs': 10,

    'scheduler': {"method": "cosine", "warmup_epochs": 1},

    'pretrained_weights': None,

    'n_classes': 84011,
    'data_frac': 1.,

    'num_workers': 4,
    'crop_size': 448,

    'neptune_project': 'ieldarov/VPR',
    'neptune_api_token': None
}

args['tr_aug'] = A.Compose([A.LongestMaxSize(512, p=1),
                            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']),
                            A.HorizontalFlip(always_apply=False, p=0.5),
                            ],
                           p=1.0
                           )

args['val_aug'] = A.Compose(
    [A.LongestMaxSize(512, interpolation=cv2.INTER_AREA, p=1),  # added by IE: interpolation=cv2.INTER_AREA
     A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, p=1),
     A.CenterCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']),
     ],
    p=1.0
    )

