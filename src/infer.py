import sys
import os
import time

import numpy
import torch
import pickle
import tqdm
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from models import *
from configs import config7
from files import get_dir_name, join, crc32, DIRS_KEY, DIRS_MAP_FILE_NAME, IMAGES_LIST_FILE_NAME, IMAGES_METADATA_FILE_NAME
from embeddings_multi_dir import Dict2Class, get_embedding_file_name, MultiDirDataset, normalize_imagenet_img
from torch.nn import CosineSimilarity
import time


def prepare_query_tensor(img_path, aug, normalizer):
    try:

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if aug:
            image = aug(image=image)['image']

        image = image.astype(np.float32)
        if normalizer:
            image = normalizer(image)

        image = torch.from_numpy(image.transpose((2, 0, 1))).cuda()
        image = torch.unsqueeze(image, 0)
        return {'input': image}

    except Exception as e:
        print(f"Error")
        print(e)
        return None


if __name__ == '__main__':
    args = Dict2Class(config7.args)
    query_image = sys.argv[1]
    ds_dir = sys.argv[2]
    threshold = float(sys.argv[3])
    print(f"Query image: {query_image}")

    query_tensor = prepare_query_tensor(query_image, args.val_aug, normalize_imagenet_img)

    start_embedding_file = 29
    finish_embedding_file = 29
    embeddings = np.empty((0, 512))
    for file_id in range(start_embedding_file, finish_embedding_file + 1):
        with open(join(ds_dir, get_embedding_file_name(start_embedding_file)), 'rb') as f:
            e = pickle.load(f)
            embeddings = np.vstack((embeddings, e))

    embeddings = torch.from_numpy(embeddings).cuda()

    images_to_skip = start_embedding_file * embeddings.shape[0]
    data_set = MultiDirDataset(ds_dir, args.val_aug, normalize_imagenet_img, images_to_skip)

    model = Net(args)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(args.model_weights_file_name))

    model_output = model.forward(query_tensor, get_embeddings=True)
    logits = model_output['logits'][0].detach().cpu().numpy()
    preds = np.argmax(logits)
    print(f"Predicted Landmark: {preds}, confidence: {logits[preds]:.2f}")

    query_embedding = logits = model_output['embeddings'][0]

    t_start = time.time()
    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6).cuda()
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    ).detach().cpu().numpy()

    t_finish = time.time()

    valid_dirs = None
    all_valid_imgs = None

    with open(join(ds_dir, DIRS_MAP_FILE_NAME), 'rb') as f:
        valid_dirs = pickle.load(f)[DIRS_KEY]

    with open(join(ds_dir, IMAGES_LIST_FILE_NAME), 'rb') as f:
        all_valid_imgs = pickle.load(f)

    sorted_ids = np.argsort(similarities)[::-1]
    similar_imgs_paths = []

    for id in sorted_ids:
        if similarities[id] < threshold:
            break

        similar_imgs_paths.append(
            join(get_dir_name(images_to_skip + id, valid_dirs), all_valid_imgs[images_to_skip + id] + '.jpg')
        )

    print(f"Similarity search time {t_finish - t_start:.2f} seconds, DB size = {embeddings.shape[0]}, threshold:{threshold}")
    print(f"Similar imgs:", similar_imgs_paths)
