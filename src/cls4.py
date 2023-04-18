import math
import pickle
import time

import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from os import listdir
from os.path import isfile
from files import join
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


class PDClassifier(nn.Module):

    def __init__(self, img_features, n_classes):
        super(PDClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(img_features, img_features),
            nn.BatchNorm1d(img_features),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(img_features, n_classes)
        )
        print("2 layers x full img features, relu ahead of drop, dropped GoodPostcards., dropout = .05")
        '''
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear2.bias.data.fill_(0)
        '''

    def forward(self, x):
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    all_true_labels = np.array([], dtype='int16')
    all_preds = np.array([], dtype='int16')
    all_img_ids = []

    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)
        # if processed_size == 0:
        #    print(">>> ", outputs[0], torch.argmax(outputs[0]), labels[0], imgIds[0])

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

        all_true_labels = np.append(all_true_labels, labels.data.cpu().numpy().astype('int16'))
        all_preds = np.append(all_preds, preds.cpu().numpy().astype('int16'))

    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size

    f1b = f1_score(all_true_labels, all_preds, average='weighted')

    return val_loss, val_acc, f1b


def train(train_loader, val_loader, model, epochs, batch_size):

    best_accuracy = 0
    best_loss = np.inf
    best_f1 = 0
    history = []
    lr = 0.0125
    opt = optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.25, patience=2, min_lr=0.0001)
    #scheduler = StepLR(opt, step_size=70, gamma=0.5)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, opt)
        # IE: Добавил, чтобы отображать текущий LR

        test_loss, test_acc, f1 = eval_epoch(model, val_loader, criterion)
        if test_acc > best_accuracy:
            best_accuracy = test_acc

        if test_loss < best_loss:
            best_loss = test_loss

        if f1 > best_f1:
            best_f1 = f1
            # best_model_weights = copy.deepcopy(model.state_dict())
            best_model_weights = None

        scheduler.step(test_loss)
        #lr = scheduler.get_last_lr()[-1]
        lr = scheduler._last_lr[-1]

        #history.append((train_loss, train_acc, val_loss, val_acc))
        #tqdm.write(log_template.format(ep=epoch + 1, lr=lr, train_loss=train_loss, test_loss=test_loss, train_acc=train_acc, test_acc=test_acc, f1=f1))
        print(f"Epoch {epoch:02d} lr {lr:0.6f} train_loss {train_loss:0.3f} train_acc {train_acc:0.3f} test_loss {test_loss:0.3f} test_acc {test_acc:0.3f} f1 {f1:0.3f} T {((time.time() - epoch_start_time)/60):0.1f}")

    return history, best_accuracy, best_loss, best_model_weights


class PSDS(Dataset):
    EMBEDDINGS_FN = "embeddings.pkl"
    TRAIN_SUBSET_FN = "train_subset.pkl"
    TEST_SUBSET_FN = "test_subset.pkl"
    MAIN_DESCRIPTION_KEY = "main"

    def __init__(self, definition, is_test, train_share=0.8):
        self.root_dir = definition[PSDS.MAIN_DESCRIPTION_KEY]
        self.classes = []
        for cls in definition.keys():
            if cls == PSDS.MAIN_DESCRIPTION_KEY:
                continue
            self.classes.append(cls)

        #self.is_test = is_test
        #self.train_share = train_share
        #self.img_embeddings = {}
        #self.train_imgs = {}

        self.imgs = []

        # detect if images have embeddings

        try:
            with open(join(self.root_dir, PSDS.EMBEDDINGS_FN), 'rb') as f:
                img_embeddings = pickle.load(f)

        except FileNotFoundError:
            #generate embeddings
            assert False

        # detect of images are divided for test/train
        try:
            with open(join(self.root_dir, PSDS.TRAIN_SUBSET_FN), 'rb') as f:
                train_imgs = pickle.load(f)
            with open(join(self.root_dir, PSDS.TEST_SUBSET_FN), 'rb') as f:
                test_imgs = pickle.load(f)

        except FileNotFoundError:
            # generate split
            train_imgs = []
            test_imgs = []
            for cls in self.classes:
                cls_files = []
                for dir in definition[cls]:
                    dir_files = listdir(join(self.root_dir, dir))
                    for dir_file in dir_files:
                        if isfile(join(join(self.root_dir, dir), dir_file)) and dir_file[-4:] == ".jpg":
                            cls_files.append(join(dir, dir_file))

                test, train = train_test_split(cls_files, test_size=math.ceil(len(cls_files) * train_share))

                for img in train:
                    i = img_embeddings['img_names'].index(img)
                    train_imgs.append(
                        [torch.tensor(self.classes.index(cls)),
                        torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float)]
                    )

                for img in test:
                    i = img_embeddings['img_names'].index(img)
                    test_imgs.append(
                        [torch.tensor(self.classes.index(cls)),
                        torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float)]
                    )

                print(cls, len(cls_files), len(train), len(test))

            with open(join(self.root_dir, PSDS.TEST_SUBSET_FN), 'wb') as f:
                pickle.dump(test_imgs, f)

            with open(join(self.root_dir, PSDS.TRAIN_SUBSET_FN), 'wb') as f:
                pickle.dump(train_imgs, f)

        print(len(img_embeddings["img_names"]), len(test_imgs), len(train_imgs))
        #assert len(img_embeddings["img_names"]) == len(test_imgs) + len(train_imgs)

        self.imgs = test_imgs if is_test else train_imgs

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index][1], self.imgs[index][0]

    def get_classes_number(self):
        return len(self.classes)

    def get_classes(self):
        return self.classes


if __name__ == '__main__':

    ds_definition = {
        "main": "E:/AITests/EbaySet/parsingTest24/primers/",
        "2sides": ["2sided"],
        "skews": ("skews", "foreign_objects"),
        "goods": ["goods", "goods2", "ras", "ras2", "GoodPostcards"],
        "multis": ["multis", "multis2", "multis3", "multis4"]
    }

    test_ds = PSDS(ds_definition, True)
    train_ds = PSDS(ds_definition, False)
    batch_size = len(train_ds)
    num_workers = 2
    n_epochs = 100
    image_features_size = 512

    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = "cuDA"
    print("Training on", DEVICE)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None
    )

    model = PDClassifier(image_features_size, train_ds.get_classes_number()).to(DEVICE)

    history, best_accuracy, best_loss, best_model_weights = train(
        train_loader,
        test_loader,
        model,
        n_epochs,
        batch_size
    )
