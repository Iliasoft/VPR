import os
import sys
import pickle
import tqdm
import cv2
import numpy as np
import glob
import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from files import join
from src.models import *
from configs import config1, config2, config3, config4, config5, config6, config7


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


class DirDataset(Dataset):

    def __init__(self, root_dir, aug, normalize=None):
        self.root_dir = root_dir
        self.normalize = normalize
        self.aug = albumentation

        self.image_names = list(Path(root_dir).glob('**/*.jpg'))
        # print(self.image_names)

    def __len__(self):
        return len(self.image_names)

    def augment(self, img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def __getitem__(self, idx):
        img_path = str(self.image_names[idx])
        image = cv2.imread(img_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:

            print("Error", img_path)

        image = self.augment(image)

        if self.normalize:
            image = self.normalize(image)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return {'img_name': join(self.image_names[idx].parts[-2], self.image_names[idx].name), 'input': image}


def get_embeddings(dl, model, args):
    with torch.no_grad():
        embeddings = np.zeros((len(dl.dataset), args.embedding_size))
        img_names = []

        iterator = iter(dl)
        idx = 0
        for batch in tqdm(iterator):

            batch['input'] = batch['input'].cuda()
            outs = model.forward(batch, get_embeddings=True)["embeddings"]
            embeddings[idx * batch_size:idx * batch_size + outs.size(0), :] = outs.detach().cpu().numpy()
            img_names.extend(batch['img_name'])
            idx += 1

    return embeddings, img_names


if __name__ == '__main__':
    args = Dict2Class(config7.args)
    print("Generating img embeddings for the directory " + sys.argv[1])

    albumentation = args.class_aug

    batch_size = 24
    data_set = DirDataset(sys.argv[1], albumentation, normalize_imagenet_img)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        # pin_memory_device='cuda:0'
    )

    model = Net(args)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load("../" + args.model_weights_file_name))

    embeddings, img_names = get_embeddings(data_loader, model, args)

    with open(os.path.join(sys.argv[1], "embeddings.pkl"), 'wb') as f:
        pickle.dump(
            {"img_names": img_names, "embeddings": embeddings},
            f
        )
