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


class ImgPairsDataset(Dataset):

    def __init__(self, root_dir):

        self.df = pd.read_csv(
            ImgPairsDataset.join(root_dir, IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME),
            header=0, names=[IMG, CAT, EXT_CATS]
        )

        self.df[CAT] = self.df[CAT].fillna(NO_CATEGORY_REPLACEMENT)
        self.df[CAT] = self.df.apply(lambda x: int(x[CAT]), axis=1)
        self.categories = self.df[CAT].unique()

        self.categories_to_images = {cat: self.df[IMG][self.df[CAT] == cat].values for cat in self.categories}

        self.positive_pairs = []
        expected_sum = 0
        for key in self.categories_to_images.keys():
            if key != -1:
                self.positive_pairs.extend(list(itertools.combinations(self.categories_to_images[key], 2)))
                expected_sum += math.factorial(len(self.categories_to_images[key]))/(math.factorial(len(self.categories_to_images[key]) - 2) * 2)

        assert(expected_sum == len(self.positive_pairs))
        try:

            with open(ImgPairsDataset.join(root_dir, NEGATIVE_PAIRS_FILE_NAME), 'rb') as f:
                self.negative_pairs = pickle.load(f)
                assert(len(self.negative_pairs) == len(self.positive_pairs))

            print("Loaded pre-calculated negative pairs")

        except FileNotFoundError:

            self.negative_pairs = set()
            positive_categories = np.delete(self.categories, np.where(self.categories == NO_CATEGORY_REPLACEMENT))
            for i in range(int(expected_sum/1.9)):
                pair = (
                    random.choice(self.categories_to_images[random.choice(positive_categories)]),
                    random.choice(self.categories_to_images[random.choice(positive_categories)])
                )

                if pair not in self.positive_pairs and pair[0] != pair[1] and pair not in self.negative_pairs and (pair[1], pair[0]) not in self.negative_pairs:
                    self.negative_pairs.add(pair)

            # print(len(self.negative_pairs))

            for i in range(len(self.positive_pairs) - len(self.negative_pairs) + 2):
                pair = (
                    random.choice(self.categories_to_images[NO_CATEGORY_REPLACEMENT]),
                    random.choice(self.categories_to_images[NO_CATEGORY_REPLACEMENT]))

                if pair not in self.positive_pairs and pair[0] != pair[1] and pair not in self.negative_pairs and (pair[1], pair[0]) not in self.negative_pairs:
                    self.negative_pairs.add(pair)

            print(len(self.negative_pairs), len(self.positive_pairs))
            self.negative_pairs = list(self.negative_pairs)
            assert(len(self.negative_pairs) == len(self.positive_pairs))

            with open(ImgPairsDataset.join(root_dir, NEGATIVE_PAIRS_FILE_NAME), 'wb') as f:
                pickle.dump(self.negative_pairs, f)

        with open(ImgPairsDataset.join(ds_root_folder, EMBEDDINGS_FILE_NAME), 'rb') as f:
            self.embeddings = pickle.load(f)

        assert(len(self.df[IMG].unique()) == len(self.embeddings[IMAGE_NAMES]))


    def __len__(self):
        return len(self.negative_pairs) + len(self.positive_pairs)

    def __getitem__(self, idx):
        '''
        :return pairs of images in the DS, with class label 1 - positive, 0 - negative
        '''

        if idx < len(self.positive_pairs):
            pair = self.positive_pairs[idx]
        else:
            pair = self.negative_pairs[idx - len(self.positive_pairs)]

        return [
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(pair[0])],
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(pair[1])],
            idx < len(self.positive_pairs)
        ]

    @staticmethod
    def join(r, d):
        return r + "/" + d


if __name__ == '__main__':

    ds_root_folder = sys.argv[1]
    print(f"Testing model performance on KCP (1K manually Categorized Postcards) Data Set: {ds_root_folder}")
    ds = ImgPairsDataset(ds_root_folder)

    best_f1 = 0
    best_threshold = None
    discretisation = 1000
    for threshold in range(discretisation, 0, -1):
        predicted_labels = []
        true_labels = []
        for pair in ds:
            similarity = cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)).ravel()[0]
            predicted_labels.append(similarity >= threshold/discretisation)
            true_labels.append(pair[2])

        f1 = f1_score(true_labels, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold/discretisation

    predicted_labels = []
    true_labels = []
    for pair in ds:
        similarity = cosine_similarity(pair[0].reshape(1, -1), pair[1].reshape(1, -1)).ravel()[0]
        predicted_labels.append(similarity >= best_threshold)
        true_labels.append(pair[2])
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    print(f"Threshold: {best_threshold:.5f}, Acc: {acc:.3f}, F1:{f1:.3f}, Precision:{precision:.3f}, Recall: {recall:.3f}")
