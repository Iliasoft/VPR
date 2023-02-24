import sys
import pandas as pd
from torch.utils.data import Dataset
import itertools
import math
import random
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

NO_CATEGORY_REPLACEMENT = -1
NEGATIVE_PAIRS_FILE_NAME = "sampled_negative_pairs.pkl"
IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME = "imgs2categories.csv"
EMBEDDINGS_FILE_NAME = "embeddings.pkl"
IMG = 'img'
CAT = 'cat'
EXT_CATS = 'ext_cats'
EMBEDDINGS = "embeddings"
IMAGE_NAMES = 'img_names'
CONFUSION_LIST = "confusion_list.html"
MODE_BLEND = 0
MODE_LANDMARKS_ONLY = 1


class ImgPairsDataset(Dataset):

    def __init__(self, root_dir, mode):
        super().__init__()
        self.df = pd.read_csv(
            ImgPairsDataset.join(root_dir, IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME),
            header=0, names=[IMG, CAT, EXT_CATS]
        )
        self.mode = mode
        assert self.mode in [MODE_BLEND, MODE_LANDMARKS_ONLY]
        self.df[CAT] = self.df[CAT].fillna(NO_CATEGORY_REPLACEMENT)
        self.df[CAT] = self.df.apply(lambda x: int(x[CAT]), axis=1)
        self.categories = self.df[CAT].unique()

        self.categories_to_images = {cat: self.df[IMG][self.df[CAT] == cat].values for cat in self.categories}

        self.positive_pairs = []
        expected_sum = 0
        for key in self.categories_to_images.keys():
            if key != NO_CATEGORY_REPLACEMENT:
                self.positive_pairs.extend(list(itertools.combinations(self.categories_to_images[key], 2)))
                expected_sum += math.factorial(len(self.categories_to_images[key]))/(math.factorial(len(self.categories_to_images[key]) - 2) * 2)

        assert expected_sum == len(self.positive_pairs)

        try:

            with open(ImgPairsDataset.join(root_dir, NEGATIVE_PAIRS_FILE_NAME), 'rb') as f:
                self.negative_pairs = pickle.load(f)
                assert len(self.negative_pairs) == 2 * len(self.positive_pairs)

        except FileNotFoundError:
            print("Warning: no pre-calculated negative pairs file available")

            self.negative_pairs = []
            positive_categories = np.delete(self.categories, np.where(self.categories == NO_CATEGORY_REPLACEMENT))

            while len(self.negative_pairs) != len(self.positive_pairs):
                pair = (
                    random.choice(self.categories_to_images[random.choice(positive_categories)]),
                    random.choice(self.categories_to_images[random.choice(positive_categories)])
                )

                if pair not in self.positive_pairs and (pair[1], pair[0]) not in self.positive_pairs and pair[0] != pair[1] and pair not in self.negative_pairs and (pair[1], pair[0]) not in self.negative_pairs:
                    self.negative_pairs.append(pair)

            while len(self.negative_pairs) != 2 * len(self.positive_pairs):
                pair = (
                    random.choice(self.categories_to_images[NO_CATEGORY_REPLACEMENT]),
                    random.choice(self.categories_to_images[NO_CATEGORY_REPLACEMENT])
                )

                if pair not in self.positive_pairs and (pair[1], pair[0]) not in self.positive_pairs and pair[0] != pair[1] and pair not in self.negative_pairs and (pair[1], pair[0]) not in self.negative_pairs:
                    self.negative_pairs.append(pair)

            assert len(self.negative_pairs) == 2 * len(self.positive_pairs)

            with open(ImgPairsDataset.join(root_dir, NEGATIVE_PAIRS_FILE_NAME), 'wb') as f:
                pickle.dump(self.negative_pairs, f)

        with open(ImgPairsDataset.join(ds_root_folder, EMBEDDINGS_FILE_NAME), 'rb') as f:
            self.embeddings = pickle.load(f)

        assert len(self.df[IMG].unique()) == len(self.embeddings[IMAGE_NAMES])

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs) // 2

    def __getitem__(self, idx):
        '''
        :returns pairs of images in the DS, with class label 1 - positive, 0 - negative
        '''

        if idx < len(self.positive_pairs):
            pair = self.positive_pairs[idx]
        elif self.mode == MODE_LANDMARKS_ONLY:
            pair = self.negative_pairs[idx - len(self.positive_pairs)]
        else:
            pair = self.negative_pairs[idx - len(self.positive_pairs) + len(self.negative_pairs) // 2]

        return [
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(pair[0])],
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(pair[1])],
            idx < len(self.positive_pairs), # true class label
            pair[0],
            pair[1]
        ]

    def __iter__(self):
        return iter(map(self.__getitem__, np.arange(self.__len__())))

    @staticmethod
    def join(r, d):
        return r + "/" + d


def html_img(file_name):
    return f"<td><img height=350 src=\"file://{ImgPairsDataset.join(ds_root_folder, file_name)}\" alt=\"{file_name}\"</img></td>"


def html_pair(file_name_1, file_name_2, confidence, type):

    return f"<tr><td>{type}<br>Similarity:{confidence:.2f}</td>{html_img(file_name_1)}{html_img(file_name_2)}</tr>"


def benchmark(test_name, ds, confusion_file_name, discretisation=100):

    best_f1 = 0
    best_threshold = 0

    for threshold in range(discretisation, 0, -1):
        predicted_labels = []
        true_labels = []
        for pair in ds:
            similarity = cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)).ravel()[0]
            predicted_labels.append(similarity >= threshold/discretisation)
            true_labels.append(pair[2])

        f1 = f1_score(true_labels, predicted_labels)
        # print(threshold, f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold/discretisation

    predicted_labels = []
    true_labels = []
    confusion_matrix_fp = []
    confusion_matrix_fn = []

    for pair in ds:

        similarity = cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)).ravel()[0]
        predicted_labels.append(similarity >= best_threshold)
        true_labels.append(pair[2])

        if true_labels[-1] != predicted_labels[-1] and predicted_labels[-1]:
            confusion_matrix_fp.append((*pair, similarity))
        elif true_labels[-1] != predicted_labels[-1] and not predicted_labels[-1]:
            confusion_matrix_fn.append((*pair, similarity))

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    print(f"{test_name}: Threshold:{best_threshold:.5f}, Acc:{acc:.3f}, F1:{f1:.3f}, Precision:{precision:.3f}, Recall:{recall:.3f}")

    with open(ImgPairsDataset.join(ds_root_folder, confusion_file_name), "w") as text_file:
        print("<html><body><table width=66% border=1>", file=text_file)
        for pair in confusion_matrix_fp:
            print(
                html_pair(pair[3], pair[4], pair[5], "False Positive"),
                file=text_file
            )

        for pair in confusion_matrix_fn:
            print(
                html_pair(pair[3], pair[4], pair[5], "False Negative"),
                file=text_file
            )
        print("</body></html>", file=text_file)

    # print(f"See table with FNs and FPs: {ImgPairsDataset.join(ds_root_folder, confusion_file_name)}")


if __name__ == '__main__':

    ds_root_folder = sys.argv[1]

    print(f"Testing performance on KISA (1K Image Similarity Assembly) Data Set: {ds_root_folder}")

    test_name = "Landmark postcards"
    ds = ImgPairsDataset(ds_root_folder, MODE_LANDMARKS_ONLY)
    benchmark(test_name, ds, test_name + "_" + CONFUSION_LIST)

    test_name = "Blend of landmark and non-landmark postcards"
    ds = ImgPairsDataset(ds_root_folder, MODE_BLEND)
    benchmark(test_name, ds, test_name + "_" + CONFUSION_LIST)
