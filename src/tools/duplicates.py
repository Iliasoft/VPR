import os
import pickle
import torch
from tqdm import tqdm
from torch.nn import CosineSimilarity
import sys
#from embeddings_multi_dir import
from files import join, get_dir_name, flatten, DIRS_MAP_FILE_NAME, IMAGES_LIST_FILE_NAME, IMAGES_COMMENTS_FILE_NAME
from similarity_multi_dir import get_similarity_file_name, similarity_ranges
import statistics


def check_intersection_similarity(intersection, cosine_similarity, threshold):

    intersection = torch.tensor(list(intersection)).cuda()
    similarities = []
    for id in range(len(intersection)):

        similarities.append(
            cosine_similarity(
                intersection[id],
                intersection[id + 1:]
            ).cpu().numpy()
        )

    return min(similarities) >= threshold

def cluster_duplicates(data, cosine_similarity):

    completed = False
    while not completed:
        completed = True

        for item_array_id in tqdm(range(len(data))):
            for inner_item_array_id in range(item_array_id + 1, len(data)):
                #if item_array_id == inner_item_array_id:
                #    continue
                if not data[item_array_id] or not data[inner_item_array_id]:#mid april added
                    continue
                intersection = data[item_array_id] & data[inner_item_array_id]

                #if len(intersection) and check_intersection_similarity(intersection, cosine_similarity, 0.6):
                if len(intersection):

                    data[item_array_id].update(data[inner_item_array_id])
                    data[inner_item_array_id] = None
                    completed = False

        new_list = []
        for item in data:
            if item:
                new_list.append(item)
        data = new_list
    return data


def generate_gml_file_name(group_counter, id):
    return 'ggg_' + str(group_counter) + '_' + str(id) + ".jpg"

def generate_test_file_name(group_counter, id):
    return 'ttt_' + str(group_counter) + '_' + str(id) + ".jpg"


if __name__ == '__main__':

    root_dir = sys.argv[1]
    print("Building duplicates list for " + root_dir)
    args_vertical_scope_completion = int(sys.argv[2])
    args_horizontal_scope_completion = int(sys.argv[3])
    args_low_threshold = float(sys.argv[4])
    args_high_threshold = float(sys.argv[5])
    assert args_low_threshold < args_high_threshold <= 1

    low_range_idx = -1
    high_range_idx = -1
    for idx, s_range in enumerate(similarity_ranges):
        if s_range[0] < args_low_threshold <= s_range[1]:
            low_range_idx = idx
            break

    for idx, s_range in enumerate(similarity_ranges):
        if s_range[1] >= args_high_threshold > s_range[0]:
            high_range_idx = idx
            break

    assert low_range_idx != -1 and high_range_idx != -1
    # print(args_low_threshold, similarity_ranges[low_range_idx])
    # print(args_high_threshold, similarity_ranges[high_range_idx])
    print(f"Effective range:{similarity_ranges[low_range_idx][0]} - {similarity_ranges[high_range_idx][1]}: {similarity_ranges[low_range_idx:high_range_idx + 1]}")

    img_duplicate_ids = {}
    images_comments = None
    try:
        with open(join(root_dir, DIRS_MAP_FILE_NAME), 'rb') as f:
            image_dirs = pickle.load(f)["dirs"]

        with open(join(root_dir, IMAGES_LIST_FILE_NAME), 'rb') as f:
            images_file_names = pickle.load(f)

        with open(join(root_dir, IMAGES_COMMENTS_FILE_NAME), 'rb') as f:
            images_comments = pickle.load(f)

    except FileNotFoundError:
        print("Error: list of directories/images is/are missing")

    for v in range(args_vertical_scope_completion + 1):

        with open(join(sys.argv[1], get_similarity_file_name(v, args_horizontal_scope_completion)), 'rb') as f:
            img_duplicate_ids.update(pickle.load(f))

    unflattened_image_clusters = []
    for key in img_duplicate_ids.keys():

        f = flatten(img_duplicate_ids[key][low_range_idx:high_range_idx + 1])
        if f:
            s = set(f)
            s.add(key)
            unflattened_image_clusters.append(s)

    cosine_similarity = CosineSimilarity(dim=1, eps=1e-6).cuda()

    duplicated_sets = cluster_duplicates(unflattened_image_clusters, cosine_similarity)
    #duplicated_sets = flatten([[img_duplicate_ids[key][i] for i in range(len(img_duplicate_ids[key]))] for key in img_duplicate_ids.keys()])
    print(f"Duplicated groups:{len(duplicated_sets)}, items:{len(img_duplicate_ids)}")

    duplicated_file_names = []
    entry_sizes = []
    save_dir = "e://ebay/sim_imgs/"
    test_dir = "e://ebay/test_imgs/"
    images_2_texts = {}
    for entry in duplicated_sets:
        #print(len(entry))
        if len(entry) > 6 and len(entry) <= 50:
            entry_sizes.append(len(entry))

            for seq, img in enumerate(entry):

                src_fn = join(get_dir_name(img, image_dirs), images_file_names[img] + ".jpg")
                if seq > 1:
                    dst_fn = join(save_dir, generate_gml_file_name(len(entry_sizes), seq))
                else:
                    dst_fn = join(test_dir, generate_test_file_name(len(entry_sizes), seq))
                images_2_texts[dst_fn] = images_comments[img]
                # print(src_fn, dst_fn)
                # shutil.copyfile(src_fn, dst_fn)
            '''
            print(
                list(
                    map(
                        lambda idx: join(get_dir_name(idx, image_dirs), images_file_names[idx] + ".jpg"),
                        entry
                    )
                )
            )
            '''

    print(len(entry_sizes), statistics.mean(entry_sizes), statistics.mode(entry_sizes), statistics.median(entry_sizes), max(entry_sizes))
    with open(os.path.join(sys.argv[1], "texts.pkl"), 'wb') as f:
        pickle.dump(
            images_2_texts,
            f
        )
