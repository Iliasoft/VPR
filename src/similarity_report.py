import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from configs import config1, config2, config3, config4, config5, config6, config7
from torch.nn import CosineSimilarity
from embeddings_multi_dir import Dict2Class
import sys
from embeddings_multi_dir import DIRS_PROCESSED_CACHE, IMAGES_LIST
import shutil
from files import flatten, join, get_dir_name


def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


if __name__ == '__main__':

    args = Dict2Class(config6.args)
    print("Building duplicates list for " + sys.argv[1])
    args_horizontal_scope = 1
    args_vertical_scope_start = 1
    args_vertical_scope_finish = 5
    assert (args_horizontal_scope <= args_vertical_scope_start)
    assert (args_vertical_scope_start <= args_vertical_scope_finish)

    threshold = 0.68
    img_groups = []

    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6).cuda()

    with open(join(sys.argv[1], get_embedding_file_name(args_horizontal_scope)), 'rb') as f:
        embeddings_horizontal = pickle.load(f)
        embedding_file_length = embeddings_horizontal.shape[0]

    img_duplicate_ids = {}
    if args_vertical_scope_start > 0:
        #with open(join(sys.argv[1], f"similarity_{args_horizontal_scope}_0_{args_vertical_scope_start - 1}.pkl"), 'rb') as f:
        with open(join(sys.argv[1], f"similarity_1_1_5.pkl"), 'rb') as f:
            img_duplicate_ids = pickle.load(f)
    '''
    progress_bar = tqdm(
        total=(1 + args_vertical_scope_finish - args_vertical_scope_start) * embeddings_horizontal.shape[0],
        desc="Comparing images"
    )
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

                #indices = (similarities >= threshold and 0.9 >= similarities).any()
                indices = np.logical_and(similarities >= threshold, similarities <= 0.9)
                group = np.where(indices)[0]
                group += embedding_file_length * v + idx_correction

                progress_bar.update()

                if (embedding_file_length * args_horizontal_scope + idx) in img_duplicate_ids and len(group):
                    np.append(img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx], group)
                elif len(group):
                    img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx] = group

    # img_duplicate_ids = {key: set(img_duplicate_ids[key]) for key in img_duplicate_ids.keys()}
    with open(join(sys.argv[1], f"similarity_{args_horizontal_scope}_{args_vertical_scope_start}_{args_vertical_scope_finish}.pkl"), 'wb') as f:
        pickle.dump(img_duplicate_ids, f)
    '''
    '''
    with open(join(sys.argv[1], f"similarity_{args_horizontal_scope}_{args_vertical_scope_start}_{args_vertical_scope_finish}.pkl"), 'rb') as f:
        img_duplicate_ids = pickle.load(f)
    '''
    ##################################################################
    duplicates_for_flatten = []

    for key in img_duplicate_ids.keys():
        s = set()
        s.add(key)
        s.update(img_duplicate_ids[key])
        duplicates_for_flatten.append(s)
    print(f"Flattening started")
    duplicates = flatten(duplicates_for_flatten)
    print(f"Flattening completed:{len(duplicates)}")
    ##################################################################
    with open(join(sys.argv[1], IMAGES_LIST), 'rb') as f:
        images_file_names = pickle.load(f)
    with open(join(sys.argv[1], DIRS_PROCESSED_CACHE), 'rb') as f:
        s = pickle.load(f)
        image_dirs = s["dirs"]

    copies_dump_dir = join(sys.argv[1], "twins_report")
    group_counts = {}
    for group in tqdm(duplicates, desc="Generanting files"):

        if len(group) in group_counts:
            group_counts[len(group)] += 1
        else:
            group_counts[len(group)] = 1

        # similar_imgs = [join(get_dir_name(dup_id, image_dirs), images_file_names[dup_id] + ".jpg") for dup_id in img_duplicate_ids[key]]
        if not (20 > len(group) >= 10):
            continue

        '''
        shutil.copyfile(
            join(get_dir_name(key, image_dirs), images_file_names[key] + ".jpg"),
            join(copies_dump_dir, images_file_names[key] + ".jpg"),
        )
        '''

        files = list(group)
        for seq_number, file_id in enumerate(files):
            shutil.copyfile(
                join(get_dir_name(file_id, image_dirs), images_file_names[file_id] + ".jpg"),
                join(copies_dump_dir, images_file_names[files[0]] + "_" + str(seq_number) + ".jpg")
            )

    print(group_counts)