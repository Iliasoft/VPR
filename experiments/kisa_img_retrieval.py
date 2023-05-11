import sys
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.metrics import accuracy_score

import itertools
import math
import random
import numpy as np
import pickle
import torch
from torch.nn import CosineSimilarity

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

NO_CATEGORY_REPLACEMENT = -1
IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME_KISA = "imgs2categories.csv"
IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME_GLM = "recognition_solution_v2.1_extended.csv"
EMBEDDINGS_FILE_NAME = "embeddings.pkl"
IMG = 'img'
CAT = 'cat'
EXT_CATS = 'ext_cats'
EMBEDDINGS = "embeddings"
IMAGE_NAMES = 'img_names'


class IRTestDataset(Dataset):

    def __init__(self, root_dir):
        super().__init__()
        df = pd.read_csv(
            IRTestDataset.join(root_dir, IMAGE_TO_CATEGORIES_MAPPING_FILE_NAME_KISA),
            header=0, names=[IMG, CAT, EXT_CATS]
        )

        self.images = []
        self.img_labels = []

        for img in df.values:
            if img[1] == img[1]:
                self.images.append(img[0])
                self.img_labels.append(int(img[1]))

        with open(IRTestDataset.join(root_dir, EMBEDDINGS_FILE_NAME), 'rb') as f:
            self.embeddings = pickle.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        :returns class label, embedding
        '''
        return [
            self.img_labels[idx],
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(self.images[idx])],
        ]

    def get_same_labeled(self, idx):

        result = []
        for id, l in enumerate(self.img_labels):
            if l == self.img_labels[idx] and id != idx:
                result.append(self.images[id])

        return result

    def __iter__(self):
        return iter(map(self.__getitem__, np.arange(self.__len__())))

    @staticmethod
    def join(r, d):
        return r + "/" + d


def top_k(images_id, array, k, threshold):
    '''returns labels of top k items in array, if values are => threshold'''
    sorted_labels = np.argsort(array)[::-1][:k]
    labels_above_th = []
    for l in sorted_labels:
        if array[l] >= threshold:
            labels_above_th.append(images_id[l])
        else:
            break
    return labels_above_th


def num_related_items(preds, actuals):
    '''returns number of preds-labels available in actuals'''
    return len(set(preds) & set(actuals))


def get_ir_metrics(images_id, actuals, predictions, k, threshold=-0.99):
    """Return precision and recall at k metrics averaged for images"""
    # actual - for each given images contains labels of actual similar images
    # predictions - for each given images has similarity scores with all other images

    precisions_at_k = []
    precisions_at_r = []
    average_precisions_at_k = []
    recalls_at_k = []
    for img_id in range(len(predictions)):
        predicted_top_k = top_k(images_id, predictions[img_id], k, threshold)
        predicted_top_r = top_k(images_id, predictions[img_id], len(actuals[img_id]), threshold)
        n_related_in_top_k = num_related_items(predicted_top_k, actuals[img_id])
        precisions_at_k.append(n_related_in_top_k / k)
        precisions_at_r.append(num_related_items(predicted_top_r, actuals[img_id]) / len(actuals[img_id]))
        recalls_at_k.append(n_related_in_top_k / min(len(actuals[img_id]), k))
        if n_related_in_top_k != 0:
            pk = []
            for id, i in enumerate(predicted_top_k):
                if i in actuals[img_id]:
                    pk.append(1/(id + 1))

            average_precisions_at_k.append(sum(pk)/n_related_in_top_k)
        else:
            average_precisions_at_k.append(0)

    return np.mean(precisions_at_k), np.mean(precisions_at_r), np.mean(average_precisions_at_k), np.mean(recalls_at_k)


def benchmark(ds):

    similarity_actual = []

    for img_id in range(len(ds)):
        similarity_actual.append(ds.get_same_labeled(img_id))

    img_embeddings = torch.empty((len(ds), 512))
    for id, emb in enumerate(ds):
        img_embeddings[id] = torch.tensor(emb[1])

    cs_matrix = pairwise_cosine_similarity(img_embeddings, img_embeddings)
    cs_matrix.fill_diagonal_(-1)
    cs_matrix = cs_matrix.numpy()

    for k in range(1, 6):
        precision_k, precision_r, avarage_precision, recall = get_ir_metrics(ds.images, similarity_actual, cs_matrix, k=k)
        print(f"k:{k} =>Precision@k:{precision_k:.2f} Precision@R:{precision_r:.2f} Average Precision@k:{avarage_precision:.2f} Recall@k:{recall:.2f}")


if __name__ == '__main__':

    ds_root_folder = sys.argv[1]

    print(f"Testing performance on KISA (1K Image Similarity Assembly) Data Set: {ds_root_folder}")

    test_name = "Landmark postcards"
    ds = IRTestDataset(ds_root_folder)
    benchmark(ds)
