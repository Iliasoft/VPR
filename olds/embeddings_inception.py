import os
import sys
import pickle
import tqdm
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from files import join
from models import *
from configs import config1, config2, config3, config4, config5, config6, config7
import glob

import os
import sys
import pickle
import tqdm
import cv2
import numpy as np
import glob
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from files import join
from models import *
from configs import config1, config2, config3, config4, config5, config6, config7
import torchvision.transforms as transforms
from torch.utils.model_zoo import load_url

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.inception import Inception3
from warnings import warn


class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else:
            warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x.view(x.size(0), -1)
        # 2048
        # x = self.fc(x_for_capt)
        # 1000 (num_classes)
        return x_for_capt


def beheaded_inception_v3(transform_input=True):
    model = BeheadedInception3(init_weights=True, transform_input=transform_input)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

    model.load_state_dict(load_url(inception_url))
    return model

def normalize_imagenet_img(img):

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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


def get_embeddings(dl, model):
    with torch.no_grad():
        embeddings = np.zeros((len(dl.dataset), 2048))
        img_names = []

        iterator = iter(dl)
        idx = 0
        for batch in tqdm(iterator):

            outs = model.forward(batch['input'].cuda())
            embeddings[idx * batch_size:idx * batch_size + outs.size(0), :] = outs.detach().cpu().numpy()
            img_names.extend(batch['img_name'])
            idx += 1

    return embeddings, img_names


if __name__ == '__main__':
    print("Generating img embeddings for the directory " + sys.argv[1])

    albumentation = A.Compose([
            A.Resize(299, 299)
        ]
    )
    batch_size = 32
    data_set = DirDataset(sys.argv[1], albumentation, normalize_imagenet_img)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    model = beheaded_inception_v3().train(False).to(device)

    embeddings, img_names = get_embeddings(data_loader, model)

    with open(os.path.join(sys.argv[1], "embeddings.pkl"), 'wb') as f:
        pickle.dump(
            {"img_names": img_names, "embeddings": embeddings},
            f
        )
