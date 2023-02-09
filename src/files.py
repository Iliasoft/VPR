from tqdm import tqdm
import zlib
import pickle
import sys
from os import listdir
from os.path import isfile, isdir


def crc32(fileName):
    with open(fileName, 'rb') as fh:
        return "%08X" % (zlib.crc32(fh.read(4096), 0) & 0xFFFFFFFF)


def crc32_complete(fileName):
    with open(fileName, 'rb') as fh:
        hash = 0
        while True:
            s = fh.read(4096)
            if not s:
                hash = zlib.crc32(s, hash)
        return "%08X" % (hash & 0xFFFFFFFF)


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_dir_name(file_id, dirs):

    for dir in dirs:
        if dir[2] <= file_id <= dir[3]:
            return dir[1]

    return None


def join(r, d):
    return r + "/" + d


if __name__ == '__main__':

    # print(sys.argv)
    args_disk = 'EbayI'
    args_root_dir = sys.argv[1]  # 'f:/test'# 'f:/ebay/ebay.IX' #'f:/test'
    args_save_dir = sys.argv[2]  # "d:/"
    args_start_file_counter = 0
    SOURCE_DIRS_CACHE = "source_dirs_cache.pkl"
    DIRS_PROCESSED_CACHE = "scanned_dirs.pkl"
    IMAGES_LIST = "images_file_names.pkl"
    IMAGES_COMMENTS = "images_comments.pkl"
    try:
        with open(join(args_save_dir, SOURCE_DIRS_CACHE), 'rb') as f:
            all_dirs = pickle.load(f)
        # print("loaded dirs from cache")

    except FileNotFoundError:
        l1_dirs = [join(args_root_dir, dir) for dir in listdir(args_root_dir) if isdir(join(args_root_dir, dir))]
        l2_dirs = flatten([[join(l1_dir, l2_dir) for l2_dir in listdir(l1_dir) if isdir(join(l1_dir, l2_dir))] for l1_dir in l1_dirs])
        l3_dirs = flatten([[join(l2_dir, l3_dir) for l3_dir in listdir(l2_dir) if isdir(join(l2_dir, l3_dir))] for l2_dir in l2_dirs])
        all_dirs = list(set(l2_dirs + l3_dirs))
        print(len(all_dirs), len(l2_dirs), len(l3_dirs))

        with open(join(args_save_dir, SOURCE_DIRS_CACHE), 'wb') as f:
            pickle.dump(all_dirs, f)

    files_counter = args_start_file_counter
    all_valid_imgs = []
    all_valid_imgs_comments = []
    unique_auctions = set()
    unique_urls = set()
    duplicated_auctions = 0
    invalid_txt = 0

    try:
        with open(join(args_save_dir, DIRS_PROCESSED_CACHE), 'rb') as f:
            s = pickle.load(f)
            valid_dirs = s["dirs"]
            last_processed_seq = s["last"]
    except FileNotFoundError:
        last_processed_seq = 0
        valid_dirs = []

    for seq, dir in tqdm(enumerate(all_dirs), total=len(all_dirs)):
        if seq < last_processed_seq:
            continue

        valid_dir_files = []
        valid_dir_files_comments = []
        all_dir_files = [f for f in listdir(dir)]
        for f in all_dir_files:

            if f.endswith(".jpg") and f[:-4] + ".txt" in all_dir_files:

                text_file_data = []
                with open(join(dir, f[:-4] + ".txt")) as text_file:
                    try:
                        for line in text_file:
                            text_file_data.append(line)

                    except UnicodeDecodeError:
                        # print(join(dir, f[:-4] + ".txt"))
                        continue

                if len(text_file_data) < 3:
                    invalid_txt += 1
                    continue

                try:
                    url_id = text_file_data[2][32:32 + text_file_data[2][32:].index("/")]

                except ValueError:
                    url_id = None

                # print(url_id, text_file_data[2][32:], text_file_data[2][32:].index("/"))

                if text_file_data[0][25:-1] not in unique_auctions and (not url_id or url_id not in unique_urls):

                    valid_dir_files.append(f[:-4])
                    valid_dir_files_comments.append(text_file_data[1][:-1])

                    unique_auctions.add(text_file_data[0][25:-1])
                    unique_auctions.add(url_id)
                else:
                    duplicated_auctions += 1
                    continue

        if valid_dir_files:
            valid_dirs.append(
                (args_disk, dir, files_counter, len(all_valid_imgs) + len(valid_dir_files) - 1)
            )
            files_counter += len(valid_dir_files)
            all_valid_imgs.extend(valid_dir_files)
            all_valid_imgs_comments.extend(valid_dir_files_comments)

        with open(join(args_save_dir, DIRS_PROCESSED_CACHE), 'wb') as f:
            pickle.dump({"dirs": valid_dirs, "last": seq}, f)

        with open(join(args_save_dir, IMAGES_LIST), 'wb') as f:
            pickle.dump(all_valid_imgs, f)

        with open(join(args_save_dir, IMAGES_COMMENTS), 'wb') as f:
            pickle.dump(all_valid_imgs_comments, f)

    ###########################################################################################
    '''
    crc_duplicates = set()
    for fs in all_valid_imgs_metadata:
        if len(all_valid_imgs_metadata[fs]) > 1:
            crc_reads += len(all_valid_imgs_metadata[fs])
            unique_images = set()
            fss = []
            for f in all_valid_imgs_metadata[fs]:
                crc = crc32(
                    join(get_dir_name(f, valid_dirs), all_valid_imgs[f][0] + ".jpg")
                )
                crc_reads += 1
                if crc in unique_images:

                    print(
                        join(get_dir_name(f, valid_dirs), all_valid_imgs[f][0] + ".jpg")
                    )
                    crc_duplicates.add(f)
                else:
                    unique_images.add(crc)
            '''
    ###########################################################################################
    print(len(all_valid_imgs), duplicated_auctions, invalid_txt)
    ###########################################################################################
    '''
    groups_df = pd.DataFrame(data=valid_dirs)
    groups_df.to_csv(
        join(args_save_dir, "folders.csv"),
        header=False, index=True
    )
    '''
