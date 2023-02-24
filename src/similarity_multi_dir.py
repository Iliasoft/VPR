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
from files import flatten, join, get_dir_name, flatten


def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


def get_similarity_file_name(h, v):
    return f"similarity_{h}_{v}.pkl"

similarity_ranges = ((0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0))


if __name__ == '__main__':

    args = Dict2Class(config7.args)
    print("Building duplicates list for " + sys.argv[1])
    args_horizontal_scope = int(sys.argv[2])
    args_vertical_scope_completed = int(sys.argv[3])
    args_vertical_scope_start = int(sys.argv[4])
    args_vertical_scope_finish = int(sys.argv[5])

    assert (args_horizontal_scope <= args_vertical_scope_start)
    assert (args_vertical_scope_start <= args_vertical_scope_finish)

    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6).cuda()

    with open(join(sys.argv[1], get_embedding_file_name(args_horizontal_scope)), 'rb') as f:
        embeddings_horizontal = pickle.load(f)
        embedding_file_length = embeddings_horizontal.shape[0]

    img_duplicate_ids = {}
    if args_vertical_scope_completed != 0:
        with open(join(sys.argv[1], get_similarity_file_name(args_horizontal_scope, args_vertical_scope_completed)), 'rb') as f:
            img_duplicate_ids = pickle.load(f)
    #with open("e:/150_0.pkl", 'wb') as f:
    #    pickle.dump(embeddings_horizontal[:150 * 1024], f)
    # e = np.vstack((embeddings_horizontal, embeddings_horizontal[:56 * 1024]))
    #with open("e:/embeddings_1.pkl", 'wb') as f:
    #    pickle.dump(embeddings_horizontal[:128 * 1024], f)

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
            idx_v_correction = embedding_file_length * v
            idx_diagonal_correction = 0
            for idx, e in enumerate(embeddings_horizontal):

                if v == args_vertical_scope_start == args_horizontal_scope:
                    idx_diagonal_correction = idx + 1

                similarities = cosine_similarity(
                    torch.from_numpy(e).cuda(),
                    embedding_vertical[idx_diagonal_correction:]
                ).cpu().numpy()

                indices = [np.logical_and(similarities > similarity_ranges[0], similarities <= similarity_ranges[1]) for similarity_ranges in similarity_ranges]
                group = [(np.nonzero(ind)[0] + idx_v_correction + idx_diagonal_correction).tolist() or None for ind in indices]

                if (embedding_file_length * args_horizontal_scope + idx) in img_duplicate_ids and list(filter(None, group)):

                    existing_groups = img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx]
                    for sim_range_id in range(len(existing_groups)):

                        if group[sim_range_id]:
                            gf = list(filter(None, group[sim_range_id]))
                            existing_groups[sim_range_id] = np.append(existing_groups[sim_range_id], gf).tolist() if existing_groups[sim_range_id] else gf

                elif list(filter(None, group)):
                    img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx] = group#[item or [] for item in group]#list(filter(None, group))

                progress_bar.update()

    # duplicated_items = flatten([[img_duplicate_ids[key][i] for i in range(len(img_duplicate_ids[key]))] for key in img_duplicate_ids.keys()])
    # print(f"Duplicated groups:{len(img_duplicate_ids)}, items:{len(duplicated_items)}")

    with open(join(sys.argv[1], get_similarity_file_name(args_horizontal_scope, args_vertical_scope_finish)), 'wb') as f:
        pickle.dump(img_duplicate_ids, f)

    if args_vertical_scope_completed != 0:
        shutil.move(
            join(sys.argv[1], get_similarity_file_name(args_horizontal_scope, args_vertical_scope_completed)),
            join(sys.argv[1], "_" + get_similarity_file_name(args_horizontal_scope, args_vertical_scope_completed)),
        )
