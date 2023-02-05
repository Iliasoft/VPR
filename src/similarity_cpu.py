import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from configs import config1

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


if __name__ == '__main__':
    args = Dict2Class(config1.args)
    print("Building duplicates list for " + args.small_ds_dir)

    d = pickle.load(
        open(os.path.join(args.small_ds_dir, "embeddings.pkl"), 'rb')
    )

    embeddings = d['embeddings']
    img_names = d['img_names']
    img_accounted = [False for img in range(len(img_names))]
    img_groups = []
    img_singles = []
    threshold = 0.8

    for idx, e in tqdm(enumerate(embeddings), total=embeddings.shape[0]):
        if img_accounted[idx]:
            continue

        similarities = cosine_similarity(np.array([e]), embeddings).ravel()

        similarities_idx_sorted = np.argsort(similarities)[::-1]
        group_img_ids = []

        for id in similarities_idx_sorted:

            if similarities[id] >= threshold and not img_accounted[id]:
                group_img_ids.append(id)
                img_accounted[id] = True
            elif similarities[id] < threshold:
                break

        if len(group_img_ids) > 1:
            img_groups.append("".join([str(img_names[id]) + ", " for id in group_img_ids])[:-2])
        elif len(group_img_ids) == 1:
            img_singles.append(str(img_names[group_img_ids[0]]))

    groups_df = pd.DataFrame(data=img_groups)
    groups_df.to_csv(
        os.path.join(args.small_ds_dir, "duplicates.csv"),
        header=False, index=False
    )
    singles_df = pd.DataFrame(data=img_singles)
    singles_df.to_csv(
        os.path.join(args.small_ds_dir, "singles.csv"),
        header=False, index=False
    )
