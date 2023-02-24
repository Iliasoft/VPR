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

def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


if __name__ == '__main__':

    args = Dict2Class(config1.args)
    print("Building duplicates list for " + sys.argv[1])
    args_horizontal_scope = 0
    args_vertical_scope_start = 0
    args_vertical_scope_finish = 0
    assert (args_vertical_scope_start <= args_vertical_scope_finish)

    threshold = 0.67
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
                similarities = cosine_similarity(
                    torch.from_numpy(e).cuda(),
                    embedding_vertical
                ).cpu().numpy()

                similarities_idx_sorted = np.argsort(similarities)[::-1]
                group = []
                for id in similarities_idx_sorted:

                    if similarities[id] >= threshold:
                        group.append(embedding_file_length * v + id)
                    else:
                        break

                if v == args_horizontal_scope:
                    group.remove(embedding_file_length * args_horizontal_scope + idx)

                progress_bar.update()

                if not group:
                    continue

                if (embedding_file_length * args_horizontal_scope + idx) in img_duplicate_ids:
                    img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx].extend(group)
                else:
                    img_duplicate_ids[embedding_file_length * args_horizontal_scope + idx] = group

    with open(join(sys.argv[1], f"similarity_{args_horizontal_scope}_{args_vertical_scope_start}_{args_vertical_scope_finish}.pkl"), 'wb') as f:
        pickle.dump(img_duplicate_ids, f)
