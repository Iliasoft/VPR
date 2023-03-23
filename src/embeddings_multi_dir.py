import sys
import os
import torch
import pickle
import tqdm
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from models import *
from configs import config7
from files import get_dir_name, join, crc32, DIRS_KEY, DIRS_MAP_FILE_NAME, IMAGES_LIST_FILE_NAME, IMAGES_METADATA_FILE_NAME


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def normalize_imagenet_img(img):

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean
    img *= np.reciprocal(std, dtype=np.float32)

    return img


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class MultiDirDataset(Dataset):

    def __init__(self, root_dir, aug=None, normalize=None, ignore_imgs=None):

        self.normalize = normalize
        self.aug = aug
        self.images = None
        self.image_dirs = None
        self.ignore_imgs = ignore_imgs

        try:
            with open(join(root_dir, DIRS_MAP_FILE_NAME), 'rb') as f:
                s = pickle.load(f)
                self.image_dirs = s[DIRS_KEY]

            with open(join(root_dir, IMAGES_LIST_FILE_NAME), 'rb') as f:
                self.images = pickle.load(f)[self.ignore_imgs:]

        except FileNotFoundError:
            print("Error: list of directories/images is/are missing")

    def __len__(self):
        return len(self.images)

    def augment(self, img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def __getitem__(self, idx):

        try:
            img_path = join(get_dir_name(idx + self.ignore_imgs if self.ignore_imgs else idx, self.image_dirs), self.images[idx] + ".jpg")

            data, crc_value = crc32(img_path)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

            if image is None:
                data, crc_value = crc32('H:/ebay/ebay.v/parsingTest0/0.jpg')
                image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                print(f"Warning: ignoring item getItem({idx} @ {img_path})")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[0], image.shape[1]

            if self.aug:
                image = self.augment(image)

            image = image.astype(np.float32)
            if self.normalize:
                image = self.normalize(image)

            image = torch.from_numpy(image.transpose((2, 0, 1)))
            return {'input': image, "len": len(data), "crc": crc_value, "h": img_h, "w": img_w}

        except Exception as e:
            print(f"Error in getItem({idx})")
            print(e)
            return None


def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


def get_embeddings(dl, model, args, desired_store_size, images_to_ignore=None):

    batches_per_store_file = (desired_store_size // batch_size)
    images_len, images_crc, images_h, images_w = [], [], [], []
    batches_to_ignore = 0
    if images_to_ignore:

        with open(os.path.join(sys.argv[1], IMAGES_METADATA_FILE_NAME), 'rb') as f:
            t = pickle.load(f)
            images_len = t['len'][:images_to_ignore]
            images_crc = t['crc'][:images_to_ignore]
            images_h = t['h'][:images_to_ignore]
            images_w = t['w'][:images_to_ignore]

        batches_to_ignore = int(images_to_ignore/(batches_per_store_file*batch_size))

    with torch.no_grad():
        embeddings = np.empty((batches_per_store_file * batch_size, args.embedding_size))

        iterator = iter(dl)
        idx = batches_to_ignore*batches_per_store_file

        for batch in tqdm(iterator):

            images_len.extend(batch['len'].tolist())
            images_crc.extend(batch['crc'])
            images_h.extend(batch['h'].tolist())
            images_w.extend(batch['w'].tolist())

            batch['input'] = batch['input'].cuda()
            outs = model.forward(batch, get_embeddings=True)["embeddings"]
            embeddings[(idx % batches_per_store_file) * batch_size:(idx % batches_per_store_file) * batch_size + outs.size(0), :] = outs.detach().cpu().numpy()

            idx += 1
            if not idx % batches_per_store_file:

                with open(os.path.join(sys.argv[1], get_embedding_file_name(idx // batches_per_store_file - 1)), 'wb') as f:
                    pickle.dump(embeddings, f)
                # print(idx // store_batch_size, idx, store_batch_size)
            elif idx - batches_to_ignore * batches_per_store_file >= len(dl):

                with open(os.path.join(sys.argv[1], get_embedding_file_name(idx // batches_per_store_file)), 'wb') as f:
                    pickle.dump(embeddings[:batch_size * ((idx - 1) % batches_per_store_file) + outs.size(0)], f)

        with open(os.path.join(sys.argv[1], IMAGES_METADATA_FILE_NAME), 'wb') as f:
            pickle.dump({"len": images_len, "crc": images_crc, "h": images_h, "w": images_w}, f)


if __name__ == '__main__':
    args = Dict2Class(config7.args)
    source_dir = sys.argv[1]
    desired_store_size = int(float(sys.argv[2]) * 1024)  #200
    images_to_skip = int(sys.argv[3])  #5751804
    batch_size = 60

    assert(not images_to_skip % batch_size)

    data_set = MultiDirDataset(source_dir, args.val_aug, normalize_imagenet_img, images_to_skip)
    print(f"Generating embeddings for the directory {source_dir}, {len(data_set)} images")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = Net(args)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(args.model_weights_file_name))

    get_embeddings(data_loader, model, args, desired_store_size, images_to_skip)
