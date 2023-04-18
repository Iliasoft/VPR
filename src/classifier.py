import math
import pickle
import time
import numpy as np
import shutil
import copy
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
from sklearn.metrics import confusion_matrix


class PDClassifier(nn.Module):

    def __init__(self, img_features, n_classes):
        super(PDClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(img_features, img_features),
            nn.ReLU(),
            nn.BatchNorm1d(img_features),
            nn.Dropout(p=0.5),
            nn.Linear(img_features, n_classes)
        )
        print("2 layers x full img features, relu ahead of drop")
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
    optimizer.zero_grad(set_to_none=True)

    for inputs, labels, _ in loader:

        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
        optimizer.zero_grad(set_to_none=True)

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
    for inputs, labels, _ in val_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

        all_true_labels = np.append(all_true_labels, labels.data.cpu().numpy().astype('int16'))
        all_preds = np.append(all_preds, preds.cpu().numpy().astype('int16'))

    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size

    f1b = f1_score(all_true_labels, all_preds, average='weighted')

    return val_loss, val_acc, f1b, confusion_matrix(all_true_labels, all_preds)


def predict_classes(model, val_loader):
    model.eval()
    all_preds = np.array([], dtype='int16')
    all_img_names = []
    for inputs, _, img_names in val_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        all_img_names.extend(img_names)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

        all_preds = np.append(all_preds, preds.cpu().numpy().astype('int16'))

    return all_img_names, all_preds


def train(train_loader, val_loader, model, epochs, batch_size):

    best_accuracy = 0
    best_loss = np.inf
    best_f1 = 0
    history = []
    best_loss_cfm = None
    lr = 0.0125
    opt = optim.AdamW(model.parameters(), lr=lr)
    best_model_weights = None
    test_loss = 0
    f1 = 0
    test_acc = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=2, min_lr=0.00001)
    #scheduler = StepLR(opt, step_size=70, gamma=0.5)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, opt)
        if epoch >= 70:
            test_loss, test_acc, f1, cfm = eval_epoch(model, val_loader, criterion)
            if test_acc > best_accuracy:
                best_accuracy = test_acc

            if test_loss < best_loss:
                best_loss = test_loss
                best_loss_cfm = cfm
                best_model_weights = copy.deepcopy(model.state_dict())

            if f1 > best_f1:
                best_f1 = f1

            scheduler.step(test_loss)
            lr = scheduler._last_lr[-1]
        else:
            scheduler.step(train_loss)

        print(f"Epoch {epoch:02d} lr {lr:0.6f} train_loss {train_loss:0.3f} train_acc {train_acc:0.3f} test_loss {test_loss:0.3f} test_acc {test_acc:0.3f} f1 {f1:0.3f} T {((time.time() - epoch_start_time)/60):0.1f}")

    return history, best_accuracy, best_loss, best_model_weights, best_loss_cfm


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

        self.imgs = []

        # detect if images have embeddings

        try:
            with open(join(self.root_dir, PSDS.EMBEDDINGS_FN), 'rb') as f:
                img_embeddings = pickle.load(f)

        except FileNotFoundError:
            # first you need to generate "embeddings.pkl"
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

                if train_share == 1:

                    test, train = [], cls_files
                else:
                    test, train = train_test_split(cls_files, test_size=math.ceil(len(cls_files) * train_share))

                for id, img in enumerate(train):
                    i = img_embeddings['img_names'].index(img)
                    train_imgs.append(
                        [
                            torch.tensor(self.classes.index(cls)),
                            torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float),
                            img_embeddings['img_names'][id]
                        ]
                    )

                for id, img in enumerate(test):
                    i = img_embeddings['img_names'].index(img)
                    test_imgs.append(
                        [
                            torch.tensor(self.classes.index(cls)),
                            torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float),
                            img_embeddings['img_names'][id]
                        ]
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
        return self.imgs[index][1], self.imgs[index][0], self.imgs[index][2]

    def get_classes_number(self):
        return len(self.classes)

    def get_classes(self):
        return self.classes


if __name__ == '__main__':

    ds_definition = {
        "main": "E:/AITests/EbaySet/parsingTest24/primers/",
        "2sides": ["2sided"],
        "skews": ["skews", "foreign_objects"],
        "goods": ["goods", "goods2", "goods3", "ras"],
        "multis": ["multis", "multis2", "multis3", "greetings"]
    }

    test_ds = PSDS(ds_definition, True, 0.95)
    train_ds = PSDS(ds_definition, False, 0.95)
    batch_size = len(train_ds)
    num_workers = 2
    n_epochs = 100
    image_features_size = 512

    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on", DEVICE)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )

    model = PDClassifier(image_features_size, train_ds.get_classes_number()).to(DEVICE)
    '''
    history, best_accuracy, best_loss, best_model_weights, confusion_mtrx = train(
        train_loader,
        test_loader,
        model,
        n_epochs,
        batch_size
    )
    
    print(confusion_mtrx)
    '''
    try_ds_definition = {
        "main": "E:/AITest2",
        "goods": ["goods4"]
    }

    try_ds_dir = try_ds_definition["main"]

    #torch.save(best_model_weights, join(try_ds_dir, "model_weights.pth"))
    model.load_state_dict(torch.load(join(try_ds_dir, "model_weights.pth")))

    try_ds = PSDS(try_ds_definition, False, train_share=1.0)
    try_loader = DataLoader(
        dataset=try_ds,
        batch_size=len(try_ds),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )

    img_names, img_predicted_cls = predict_classes(model, try_loader)
    target_dir = "E:/AITest2/try_ds/"
    for id, img in enumerate(img_names):
        try:
            shutil.copy(
                join(try_ds_dir, img),
                join(target_dir + str(img_predicted_cls[id]), img)
            )
        except:
            print("Error copying file:", join(try_ds_dir, img), join(target_dir + str(img_predicted_cls[id]), img))

