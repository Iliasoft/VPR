import numpy as np
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from configs import config1
from torch.nn import CosineSimilarity
from embeddings_multi_dir import Dict2Class
from files import join
import sys
from embeddings_multi_dir import DIRS_PROCESSED_CACHE, IMAGES_LIST


def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


def get_dir_name(file_id, dirs):

    for dir in dirs:
        if dir[2] <= file_id <= dir[3]:
            return dir[1]

    return None


def join(r, d):
    return r + "/" + d


if __name__ == '__main__':

    args = Dict2Class(config1.args)
    print("Building duplicates list for " + sys.argv[1])
    args_horizontal_scope = 0
    args_vertical_scope_start = 0
    args_vertical_scope_finish = 1
    assert (args_horizontal_scope <= args_vertical_scope_start)
    assert (args_vertical_scope_start <= args_vertical_scope_finish)

    threshold = 0.75
    img_groups = []

    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6).cuda()

    with open(join(sys.argv[1], get_embedding_file_name(args_horizontal_scope)), 'rb') as f:
        embeddings_horizontal = pickle.load(f)
        embedding_file_length = embeddings_horizontal.shape[0]

    img_duplicate_ids = {}
    progress_bar = tqdm(total=(1 + args_vertical_scope_finish - args_vertical_scope_start) * embeddings_horizontal.shape[0])
    with torch.no_grad():
        for v in range(args_vertical_scope_start, args_vertical_scope_finish + 1):

            if v != args_horizontal_scope:
                with open(join(sys.argv[1], get_embedding_file_name(v)), 'rb') as f:
                    embedding_vertical = pickle.load(f)
            else:
                embedding_vertical = embeddings_horizontal

            embedding_vertical = torch.from_numpy(embedding_vertical).cuda()

            for idx, e in enumerate(embeddings_horizontal):
                if v == args_vertical_scope_start == args_horizontal_scope:
                    idx_correction = idx + 1
                else:
                    idx_correction = 0

                similarities = cosine_similarity(
                    torch.from_numpy(e).cuda(),
                    embedding_vertical[idx_correction:]
                ).cpu().numpy()

                indices = similarities >= threshold
                group = np.where(indices)[0]
                group += embedding_file_length * v + idx_correction

                '''
                similarities_idx_sorted = np.argsort(similarities)[::-1]
                group = []
                for id in similarities_idx_sorted:

                    if similarities[id] >= threshold:
                        group.append(embedding_file_length * v + id)
                    else:
                        break

                if v == args_horizontal_scope:
                    group.remove(embedding_file_length * args_horizontal_scope + idx)
                '''
                progress_bar.update()

                if (embedding_file_length * args_horizontal_scope + idx) in img_duplicate_ids:
                    np.append(img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx], group)
                elif len(group):
                    img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx] = np.append(
                        group, embedding_file_length * args_horizontal_scope + idx
                    )

    #img_duplicate_ids = {key: set(img_duplicate_ids[key]) for key in img_duplicate_ids.keys()}
    with open(join(sys.argv[1], f"similarity_{args_horizontal_scope}_{args_vertical_scope_start}_{args_vertical_scope_finish}.pkl"), 'wb') as f:
        pickle.dump(img_duplicate_ids, f)
    ##################################################################
    with open(join(sys.argv[1], IMAGES_LIST), 'rb') as f:
        images_file_names = pickle.load(f)
    with open(join(sys.argv[1], DIRS_PROCESSED_CACHE), 'rb') as f:
        s = pickle.load(f)
        image_dirs = s["dirs"]

    for key in img_duplicate_ids.keys():

        print(
            *["<a href=file://" + join(get_dir_name(file, image_dirs), images_file_names[file] + ".jpg>link</a>") for file in img_duplicate_ids[key]], "<br>"
        )
