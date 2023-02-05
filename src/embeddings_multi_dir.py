import os
import torch
import pickle
import tqdm
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from models import *
from configs import config1


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


def get_dir_name(file_id, dirs):

    for dir in dirs:
        if dir[2] <= file_id <= dir[3]:
            return dir[1]

    return None


def join(r, d):
    return r + "/" + d


DIRS_PROCESSED_CACHE = "scanned_dirs.pkl"
IMAGES_LIST = "images_file_names.pkl"


class MultiDirDataset(Dataset):

    def __init__(self, root_dir, aug=None, normalize=None):

        self.normalize = normalize
        self.aug = aug
        self.images = None
        self.image_dirs = None

        try:
            with open(join(root_dir, DIRS_PROCESSED_CACHE), 'rb') as f:
                s = pickle.load(f)
                self.image_dirs = s["dirs"]

            with open(join(root_dir, IMAGES_LIST), 'rb') as f:
                self.images = pickle.load(f)

        except FileNotFoundError:
            print("Error: list of directories/images is/are missing")

    def __len__(self):
        return len(self.images)

    def augment(self, img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def __getitem__(self, idx):

        img_path = join(get_dir_name(idx, self.image_dirs), self.images[idx] + ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.aug:
            image = self.augment(image)

        image = image.astype(np.float32)
        if self.normalize:
            image = self.normalize(image)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return {'input': image}

def get_embedding_file_name(id):
    return f"embeddings_{id}.pkl"


def get_embeddings(dl, model, args, store_batch_size):

    store_batch_size = (store_batch_size // batch_size)
    with torch.no_grad():
        embeddings = np.empty((store_batch_size * batch_size, args.embedding_size))
        iterator = iter(dl)
        idx = 0

        for batch in tqdm(iterator):

            batch['input'] = batch['input'].cuda()
            outs = model.forward(batch, get_embeddings=True)["embeddings"]
            embeddings[(idx % store_batch_size) * batch_size:(idx % store_batch_size) * batch_size + outs.size(0), :] = outs.detach().cpu().numpy()

            idx += 1
            if not idx % store_batch_size:

                with open(os.path.join(args.small_ds_dir, get_embedding_file_name(idx // store_batch_size - 1)), 'wb') as f:
                    pickle.dump(embeddings, f)
                # print(idx // store_batch_size, idx, store_batch_size)
            elif idx > len(dl):

                with open(os.path.join(args.small_ds_dir, get_embedding_file_name(idx // store_batch_size)), 'wb') as f:
                    pickle.dump(embeddings[:batch_size*((idx - 1) % store_batch_size) + outs.size(0)], f)


if __name__ == '__main__':
    args = Dict2Class(config1.args)

    albumentation = A.Compose([
            A.SmallestMaxSize(interpolation=cv2.INTER_AREA, max_size=512),
            A.CenterCrop(height=args.crop_size, width=args.crop_size, p=1.)
        ]
    )

    batch_size = 128
    data_set = MultiDirDataset(args.small_ds_dir, albumentation, normalize_imagenet_img)
    print(f"Generating embedding for the directory {args.small_ds_dir}, {len(data_set)} images")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        pin_memory_device='cuda:0'
    )

    model = Net(args)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(args.model_weights_file_name))

    get_embeddings(data_loader, model, args, 200*1024)
