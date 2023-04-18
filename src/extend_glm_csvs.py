import pandas as pd
import numpy as np
from tqdm import tqdm
import statistics
from os import listdir, remove
from os.path import isfile, isdir
from files import join


def group_id_from_fn(s):

    group_id_train_prefix = "ggg_"
    group_id_test_prefix = "ttt_"
    #if not s.startswith(group_id_train_prefix) or not s.startswith(group_id_test_prefix):
    #    return -1

    group_id_start = len(group_id_train_prefix)
    group_id_finish = s[group_id_start:].index('_')
    return int(s[group_id_start:group_id_start + group_id_finish])




original_train_csv_file_name = "E:/Dell.me/GLM/train.csv"
original_test_csv_file_name = "E:/Dell.me/GLM/recognition_solution_v2.1.csv"
extended_train_csv_file_name = "E:/Dell.me/GLM/train_extended.csv"
extended_test_csv_file_name = "E:/Dell.me/GLM/recognition_solution_v2.1_extended.csv"

sample1 = pd.read_csv(original_train_csv_file_name)

#print(sample1.values)
lmk = sample1['landmark_id'].unique()
ids = sample1['id'].unique()
max_lmk = max(lmk)

group = sample1.groupby(['landmark_id'])
array = []
for r in group:
    array.append(len(r[1]))

#sample1 = sample1.v

print(f"Original train DS stats:")
print(f"Unique LMKs:{len(lmk)}, Unique files:{len(ids)}")
print(f"Files per LMK: mode:{statistics.mode(array)}, median:{statistics.median(array)}, min:{min(array)}, max:{max(array)}")

# хочу получить версию train.csv с дополненными файлами
# 0. Прочитать первоначльную версию train.csv в памать, вычислить max(landmark_id)
# 1. получить список файлов в train директори
# 2. добавить

train_imgs_dir = "e://ebay/sim_imgs/"
test_imgs_dir = "e://ebay/test_imgs/"
skip_groups = [2,3,5,7,8,11,14,18,21,23,25,29,34,35,41,44,49,58,66,73,74,101,104,142,149,168,213,261,283,330,352,429,437,490,515,524,547,555,572,574,579,596,597,623,624,625,627,643,646,657,665,671,674,731,736,771,782,799,832,856,854,916,922,944,947,988,1002,1027,1034,1036,1038,1042,1045,1081,1085,1091,1132,1151,1196,1197,1222,1357,1425,1483,1571,1666,1737,1747,1762,1767,1771,1784,1800,1804,1855,1906,1975,1987,1992,1996,1998,2005,2006,2013,2020,2027,2028,2034,2036,2038,2039,2042,2049,2055,2061,2071,2068,2148,2202,2206,2306,2368,2372,2385,2407,2411,2419,2458,2482,2510,2521,2587,2700,2746,2751,2754,2805]

train_imgs_names = listdir(train_imgs_dir)
extra_train_lmks = set()
category_2_group = {}
extended_sample = []
for name in tqdm(train_imgs_names):
    group = group_id_from_fn(name)
    if isdir(join(train_imgs_dir, name)) or group in skip_groups:
        if isfile(join(train_imgs_dir, name)):
            remove(join(train_imgs_dir, name))
        continue
    extra_train_lmks.add(group)
    extended_sample.append([name[:-4], len(extra_train_lmks) + max_lmk])
    category_2_group[group] = len(extra_train_lmks) + max_lmk
    #if len(extra_train_lmks) > 300:
    #    break

extended_sample = pd.DataFrame(extended_sample, columns=['id', 'landmark_id'])
extended_sample.to_csv(extended_train_csv_file_name, index=False)
#############
sample2 = pd.read_csv(original_test_csv_file_name)

lmk = sample2['landmarks'].unique()
ids = sample2['id'].unique()
unique_lmks = set()

for i in lmk:
    if type(i) == str:
        ss = i.split()
        for s in ss:
            unique_lmks.add(s)


print(f"Original test DS stats:")
print(f"Unique LMKs:{len(unique_lmks)}, Unique files:{len(ids)}")

test_imgs_names = listdir(test_imgs_dir)
extra_train_lmks = set()
extended_sample = []

for name in tqdm(test_imgs_names):
    group = group_id_from_fn(name)
    if isdir(join(test_imgs_dir, name)) or group in skip_groups:
        if isfile(join(test_imgs_dir, name)):
            remove(join(test_imgs_dir, name))

        continue
    extended_sample.append([name[:-4], category_2_group[group], 'Public'])

extended_sample = pd.DataFrame(extended_sample, columns=['id', 'landmarks', 'Usage'])
extended_sample.to_csv(extended_test_csv_file_name, index=False)
