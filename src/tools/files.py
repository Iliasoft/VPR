import pickle
import sys
import zlib
import numpy as np
from os import listdir
from os.path import isfile, isdir
from tqdm import tqdm
import pandas as pd


def flatten(l):
    return [item for sublist in l for item in sublist or []]


def get_dir_name(file_id, dirs):

    for dir in dirs:
        if dir[2] <= file_id <= dir[3]:
            return dir[1]

    return None


def join(r, d):
    return r + "/" + d


def crc32(fileName):
    with open(fileName, 'rb') as fh:
        data = np.fromstring(fh.read(), dtype='uint8')
        return data, "%08X" % (zlib.crc32(data, 0) & 0xFFFFFFFF)


DIRS_MAP_FILE_NAME = "directories.pkl"
DIRS_MAP_FILE_NAME_CSV = "directories.csv"
IMAGES_LIST_FILE_NAME = "image_file_names.pkl"
IMAGES_COMMENTS_FILE_NAME = "image_comments.pkl"
IMAGES_METADATA_FILE_NAME = "image_metadata.pkl"

DIRS_KEY = "dirs"
LAST_KEY = "last"
JPG_EXT = ".jpg"
JPG_EXT_LEGACY = "__0.jpg"
TXT_EXT = ".txt"


if __name__ == '__main__':

    SOURCE_DIRS_CACHE = "dirs_cache.pkl"
    IMAGES_AUCTIONS_FILE_NAME = "image_auctions.pkl"
    IMAGES_URLS_FILE_NAME = "image_urls.pkl"

    args_disk_tag = sys.argv[1]
    args_src_dir = sys.argv[2]
    args_save_dir = sys.argv[3]
    start_files_dir = None
    args_start_file_counter = 0  # 4940970
    all_valid_imgs = []
    all_valid_imgs_comments = []
    unique_auctions = set()
    unique_urls = set()
    duplicated_auctions = 0
    invalid_txt = 0
    jpg_without_txt_files = 0

    if len(sys.argv) > 4:
        start_files_dir = sys.argv[4]
        assert args_save_dir != start_files_dir

        with open(join(start_files_dir, DIRS_MAP_FILE_NAME), 'rb') as f:
            valid_dirs = pickle.load(f)[DIRS_KEY]

        with open(join(start_files_dir, IMAGES_LIST_FILE_NAME), 'rb') as f:
            all_valid_imgs = pickle.load(f)

        with open(join(start_files_dir, IMAGES_COMMENTS_FILE_NAME), 'rb') as f:
            all_valid_imgs_comments = pickle.load(f)

        with open(join(start_files_dir, IMAGES_AUCTIONS_FILE_NAME), 'rb') as f:
            unique_auctions = pickle.load(f)

        with open(join(start_files_dir, IMAGES_URLS_FILE_NAME), 'rb') as f:
            unique_urls = pickle.load(f)

        args_start_file_counter = len(all_valid_imgs)

    try:
        with open(join(args_save_dir, SOURCE_DIRS_CACHE), 'rb') as f:
            all_dirs = pickle.load(f)
        print("loaded dirs from cache")

    except FileNotFoundError:
        l1_dirs = [join(args_src_dir, dir) for dir in listdir(args_src_dir) if isdir(join(args_src_dir, dir))]
        l2_dirs = flatten([[join(l1_dir, l2_dir) for l2_dir in listdir(l1_dir) if isdir(join(l1_dir, l2_dir))] for l1_dir in l1_dirs])
        l3_dirs = flatten([[join(l2_dir, l3_dir) for l3_dir in listdir(l2_dir) if isdir(join(l2_dir, l3_dir))] for l2_dir in l2_dirs])
        l4_dirs = flatten([[join(l3_dir, l4_dir) for l4_dir in listdir(l3_dir) if isdir(join(l3_dir, l4_dir))] for l3_dir in l3_dirs])
        l5_dirs = flatten([[join(l4_dir, l5_dir) for l5_dir in listdir(l4_dir) if isdir(join(l4_dir, l5_dir))] for l4_dir in l4_dirs])

        all_dirs = list(set(l2_dirs + l3_dirs + l4_dirs + l5_dirs))
        print(f"Total dirs: {len(all_dirs)}, L2:{len(l2_dirs)}, L3:{len(l3_dirs)}, L4:{len(l4_dirs)}, L5:{len(l5_dirs)}")

        with open(join(args_save_dir, SOURCE_DIRS_CACHE), 'wb') as f:
            pickle.dump(all_dirs, f)

    files_counter = args_start_file_counter
    try:
        with open(join(args_save_dir, DIRS_MAP_FILE_NAME), 'rb') as f:
            s = pickle.load(f)
            valid_dirs = s[DIRS_KEY]
            last_processed_seq = s[LAST_KEY]
    except FileNotFoundError:
        last_processed_seq = 0
        if len(sys.argv) <= 4:
            valid_dirs = []

    for seq, dir in tqdm(enumerate(all_dirs), total=len(all_dirs)):
        if seq < last_processed_seq:
            continue

        valid_dir_files = []
        valid_dir_files_comments = []
        all_dir_files = [f for f in listdir(dir) if isfile(join(dir, f))]
        for f in all_dir_files:

            if f.endswith(JPG_EXT) and f[:-7 if f.endswith(JPG_EXT_LEGACY) else -4] + TXT_EXT in all_dir_files:

                text_file_data = []
                with open(join(dir, f[:-7 if f.endswith(JPG_EXT_LEGACY) else -4] + TXT_EXT), encoding='utf8') as text_file:
                    try:
                        for line in text_file:
                            text_file_data.append(line)

                    except UnicodeDecodeError:
                        invalid_txt += 1
                        continue

                if len(text_file_data) < 3:

                    invalid_txt += 1
                    continue

                try:
                    url_id = text_file_data[2][32:32 + text_file_data[2][32:].index("/")]

                except ValueError:
                    url_id = None

                if text_file_data[0][25:-1] not in unique_auctions and (not url_id or url_id not in unique_urls):

                    valid_dir_files.append(f[:-4])
                    valid_dir_files_comments.append(text_file_data[1][:-1])

                    unique_auctions.add(text_file_data[0][25:-1])
                    unique_urls.add(url_id)
                else:
                    duplicated_auctions += 1

            elif f.endswith(JPG_EXT):
                jpg_without_txt_files += 1

        if valid_dir_files:
            '''
            for d in valid_dirs:
                assert not d[1] == dir
            '''
            valid_dirs.append(
                (args_disk_tag, dir, files_counter, len(all_valid_imgs) + len(valid_dir_files) - 1)
            )
            files_counter += len(valid_dir_files)
            all_valid_imgs.extend(valid_dir_files)
            all_valid_imgs_comments.extend(valid_dir_files_comments)

            with open(join(args_save_dir, DIRS_MAP_FILE_NAME), 'wb') as f:
                pickle.dump({DIRS_KEY: valid_dirs, LAST_KEY: seq}, f)

            with open(join(args_save_dir, IMAGES_LIST_FILE_NAME), 'wb') as f:
                pickle.dump(all_valid_imgs, f)

            with open(join(args_save_dir, IMAGES_COMMENTS_FILE_NAME), 'wb') as f:
                pickle.dump(all_valid_imgs_comments, f)

            with open(join(args_save_dir, IMAGES_AUCTIONS_FILE_NAME), 'wb') as f:
                pickle.dump(unique_auctions, f)

            with open(join(args_save_dir, IMAGES_URLS_FILE_NAME), 'wb') as f:
                pickle.dump(unique_urls, f)

    ###########################################################################################
    print(f"Extracted images:{len(all_valid_imgs)}, Duplicated auctions/urls:{duplicated_auctions}, Invalid txt:{invalid_txt}, Missing txt:{jpg_without_txt_files}")
    ###########################################################################################

    groups_df = pd.DataFrame(data=valid_dirs)
    groups_df.to_csv(
        join(args_save_dir, DIRS_MAP_FILE_NAME_CSV),
        header=False, index=True
    )
