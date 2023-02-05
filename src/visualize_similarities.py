import numpy as np
import os
import shutil
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from configs import config1

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


if __name__ == '__main__':
    args = Dict2Class(config1.args)

    d = pickle.load(
        open(os.path.join(args.small_ds_dir, "embeddings.pkl"), 'rb')
    )

    embeddings = d['embeddings']
    img_names = d['img_names']

    copy_folder = "D:/Dell.me/twins/"
    #src_folder = 'F:/ebay/ebay.XIII/parsingTest0/'
    for idx, e in tqdm(enumerate(embeddings), total=embeddings.shape[0]):
        similarities = cosine_similarity(np.array([e]), embeddings).ravel()
        similarities_idx_sorted = np.argsort(similarities)[::-1]

        threshold = 0.7
        sorted_ids = []

        for id in similarities_idx_sorted:
            if id != idx:
                if 0.6 < similarities[id] < threshold:
                    sorted_ids.append(id)
                else:
                    break

        if len(sorted_ids) < 2:
            continue

        if len(sorted_ids) > 100:
            print(img_names[idx] + " Too many duplicates:" + str(len(sorted_ids)))
            continue

        shutil.copyfile(args.small_ds_dir + img_names[idx], copy_folder + img_names[idx])

        counter = 1
        for id in sorted_ids:
            if id != idx:
                shutil.copyfile(
                    args.small_ds_dir + img_names[id],
                    copy_folder + img_names[idx][:-4] + "_twin_" + str(counter) + f"_{similarities[id]:.4f}.jpg"
                )
                counter += 1
