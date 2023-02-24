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
from similarity_multi_dir import get_similarity_file_name, similarity_ranges


def cluster_duplicates(data):

    completed = False
    while not completed:
        completed = True
        #counter = 0
        for item_array_id in tqdm(range(len(data))):
            for inner_item_array_id in range(item_array_id, len(data)):
                if item_array_id == inner_item_array_id:
                    continue

                if len(data[item_array_id] & data[inner_item_array_id]):
                    # print(item_array_id, inner_item_array_id)
                    data[item_array_id].update(data[inner_item_array_id])
                    data[inner_item_array_id] = set()
                    completed = False
                    #counter += 1
        # print(counter)
        new_list = []
        for item in data:
            if len(item):
                new_list.append(item)
        data = new_list
    return data


if __name__ == '__main__':

    root_dir = sys.argv[1]
    print("Building duplicates list for " + root_dir)
    args_horizontal_scope_completion = int(sys.argv[2])
    args_low_threshold = float(sys.argv[3])
    args_high_threshold = float(sys.argv[4])
    img_duplicate_ids = {}
    try:
        with open(join(root_dir, DIRS_PROCESSED_CACHE), 'rb') as f:
            image_dirs = pickle.load(f)["dirs"]

        with open(join(root_dir, IMAGES_LIST), 'rb') as f:
            images_file_names = pickle.load(f)

    except FileNotFoundError:
        print("Error: list of directories/images is/are missing")

    for h in range(args_horizontal_scope_completion + 1):

        with open(join(sys.argv[1], get_similarity_file_name(h, args_horizontal_scope_completion)), 'rb') as f:
            img_duplicate_ids.update(pickle.load(f))

    print(similarity_ranges[9:10])

    unflattened_image_clusters = []
    for key in img_duplicate_ids.keys():

        f = flatten(img_duplicate_ids[key][8:9])
        if f:
            s = set(f)
            s.add(key)
            unflattened_image_clusters.append(s)

    duplicated_sets = cluster_duplicates(unflattened_image_clusters)
    # duplicated_items = flatten([[img_duplicate_ids[key][i] for i in range(len(img_duplicate_ids[key]))] for key in img_duplicate_ids.keys()])
    print(f"Duplicated groups:{len(duplicated_sets)}, items:{len(img_duplicate_ids)}")
    #print(duplicated_sets)
    duplicated_file_names = []
    for entry in duplicated_sets:
        # print(entry)
        print(
            list(
                map(
                    lambda idx: join(get_dir_name(idx, image_dirs), images_file_names[idx] + ".jpg"),
                    entry
                )
            )
        )




    print(duplicated_file_names)
