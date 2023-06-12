import sys
import pandas as pd
from torch.utils.data import Dataset
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import pickle
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm
import matplotlib.pyplot as plt

# прочитать эмбеддинги тестовых изображений
# прочитать истинные метки тестовых изображений
# прочитать эмбеддинги индексовых изображений
# для каждого тестового изображений запустить расчет cs между ним и всеми индексными изображений
# выбрать top k
#
NO_CATEGORY_REPLACEMENT = -1
TEST_FILES_RETRIEVAL_MAPPING_FILE_NAME = "retrieval_solution_v2.1.csv"
TEST_FILES_EMBEDDINGS_FILE_NAME = "Test/embeddings_4.pkl"
INDEX_FILES_EMBEDDINGS_FILE_NAME = "Index/embeddings_4.pkl"

IMG = 'img'
CAT = 'cat'
EXT_CATS = 'ext_cats'
EMBEDDINGS = "embeddings"
IMAGE_NAMES = 'img_names'


class IRTestDataset(Dataset):

    def __init__(self, root_dir):
        super().__init__()
        df = pd.read_csv(
            IRTestDataset.join(root_dir, TEST_FILES_RETRIEVAL_MAPPING_FILE_NAME),
            header=0, names=[IMG, CAT, EXT_CATS]
        )

        self.images = []
        self.images_2_similar_images = {}

        for img in df.values:
            if img[1] != 'None':
                self.images.append(img[0])
                self.images_2_similar_images[img[0]] = img[1].split()

        with open(IRTestDataset.join(root_dir, TEST_FILES_EMBEDDINGS_FILE_NAME), 'rb') as f:
            self.embeddings = pickle.load(f)

        with open(IRTestDataset.join(root_dir, INDEX_FILES_EMBEDDINGS_FILE_NAME), 'rb') as f:
            self.index_embeddings = pickle.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        :returns class label, embedding
        '''
        id = self.images[idx]
        fn = id[2] + '/' + id + ".jpg"

        return [
            self.images_2_similar_images[self.images[idx]],
            self.embeddings[EMBEDDINGS][self.embeddings[IMAGE_NAMES].index(fn)],
        ]

    def __iter__(self):
        return iter(map(self.__getitem__, np.arange(self.__len__())))

    @staticmethod
    def join(r, d):
        return r + "/" + d


def top_k(images_id, array, k, threshold):
    '''returns labels of top k items in array, if values are => threshold'''
    sorted_labels = np.argsort(array)[::-1][:k]
    labels_above_th = []
    sim_scores_above_th = []
    for l in sorted_labels:
        if array[l] >= threshold:
            labels_above_th.append(images_id[l])
            sim_scores_above_th.append(array[l])
        else:
            break
    return labels_above_th, sim_scores_above_th


def num_related_items(preds, actuals):
    '''returns number of preds-labels available in actuals'''
    return len(set(preds) & set(actuals))


def get_ir_metrics(actuals, predictions, predictions_r, k):
    """Return precision and recall at k metrics averaged for images"""
    # actual - for each given images contains labels of actual similar images
    # predictions - for each given images has similarity scores with all other images

    precisions_at_k = []
    precisions_at_r = [-1]
    average_precisions_at_k = []
    recalls_at_k = []
    for img_id in range(len(predictions)):
        predicted_top_k = predictions[img_id]
        predicted_top_r = predictions_r[img_id]
        n_related_in_top_k = num_related_items(predicted_top_k, actuals[img_id])
        precisions_at_k.append(n_related_in_top_k / k)
        precisions_at_r.append(num_related_items(predicted_top_r, actuals[img_id]) / len(actuals[img_id]))
        recalls_at_k.append(n_related_in_top_k / min(len(actuals[img_id]), k))
        if n_related_in_top_k != 0:
            pk = []
            for id, i in enumerate(predicted_top_k):
                if i in actuals[img_id]:
                    pk.append(1 / (id + 1))

            average_precisions_at_k.append(sum(pk)/n_related_in_top_k)
        else:
            average_precisions_at_k.append(0)

    return np.mean(precisions_at_k), np.mean(precisions_at_r), np.mean(average_precisions_at_k), np.mean(recalls_at_k)

def prepare_top_predictions(ds, k=3, threshold=0):
    # получить следующий двумерный массив
    # index изображения x k названия файлов отсортированных по убыванию симиларити скор
    cosine_similarity = CosineSimilarity().cuda()

    similarity_actual = []
    ir_predicted = []
    ir_predicted_r = []
    all_index_images_embeddings = torch.from_numpy(ds.index_embeddings[EMBEDDINGS]).cuda()
    all_index_images_file_names = ds.index_embeddings[IMAGE_NAMES]
    half_len = int(len(all_index_images_embeddings) / 2)
    all_top_sim_scores = []
    for img in tqdm(ds):

        fn = map(lambda x: x[2] + '/' + x + '.jpg', img[0])
        similarity_actual.append(list(fn))
        # I have to divide embeddings it into 2 chunks to fit data in GPU memory
        similarity_scores_1 = cosine_similarity(
            torch.from_numpy(img[1]).cuda(),
            all_index_images_embeddings[:half_len]
        ).cpu().numpy()
        similarity_scores_2 = cosine_similarity(
            torch.from_numpy(img[1]).cuda(),
            all_index_images_embeddings[half_len:]
        ).cpu().numpy()

        # Precision@k / Recall@k section
        predicted_labels_at_k, sim_scores_at_k = top_k(all_index_images_file_names[:half_len], similarity_scores_1, k, threshold)
        predicted_labels_at_k_2, sim_scores_at_k_2 = top_k(all_index_images_file_names[half_len:], similarity_scores_2, k, threshold)
        predicted_labels_at_k.extend(predicted_labels_at_k_2)
        sim_scores_at_k.extend(sim_scores_at_k_2)
        predicted_labels, sim_scores_at_k = top_k(predicted_labels_at_k, sim_scores_at_k, k, threshold)
        ir_predicted.append(predicted_labels)
        all_top_sim_scores.append(sim_scores_at_k[0])

        # Precision@R section
        predicted_labels_at_r, sim_scores_at_r = top_k(all_index_images_file_names[:half_len], similarity_scores_1, len(similarity_actual[-1]), threshold)
        predicted_labels_at_r_2, sim_scores_at_r_2 = top_k(all_index_images_file_names[:half_len], similarity_scores_2, len(similarity_actual[-1]), threshold)
        predicted_labels_at_r.extend(predicted_labels_at_r_2)
        sim_scores_at_r.extend(sim_scores_at_r_2)
        predicted_labels_r, _ = top_k(predicted_labels_at_r, sim_scores_at_r_2, len(similarity_actual[-1]), threshold)
        ir_predicted_r.append(predicted_labels_r)

    return similarity_actual, ir_predicted, ir_predicted_r, all_top_sim_scores


def benchmark(ds):
    th = 0
    for k in [3, 5, 10, 100]:
        similarity_actual, ir_predicted, ir_predicted_r, top_sim_scores = prepare_top_predictions(ds, k, th/100)
        # print(np.percentile(top_sim_scores, [i for i in range(1, 101)]))
        precision_k, precision_r, avarage_precision, recall = get_ir_metrics(similarity_actual, ir_predicted, ir_predicted_r, k)
        print(f"k:{k} Precision@k:{precision_k:.3f} Precision@R:{precision_r:.3f} Average Precision@k:{avarage_precision:.3f} Recall@k:{recall:.3f}")
        '''
        plt.hist(x=top_sim_scores, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('My Very Own Histogram')
        plt.show()
        '''

if __name__ == '__main__':

    ds_root_folder = sys.argv[1]

    print(f"Testing IR performance on Data Set: {ds_root_folder}")

    ds = IRTestDataset(ds_root_folder)
    benchmark(ds)

