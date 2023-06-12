import math
import pickle
import random
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
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from collections import Counter

class PDClassifier(nn.Module):

    def __init__(self, img_features, n_classes):
        super(PDClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(img_features, img_features),
            nn.ReLU(),
            nn.BatchNorm1d(img_features),
            nn.Linear(img_features, img_features),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(img_features, n_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class PSDS(Dataset):
    EMBEDDINGS_FN = "embeddings.pkl"
    MAIN_DESCRIPTION_KEY = "main"

    def __init__(self, definition, is_test, id_number, train_share=0.8):

        self.TRAIN_SUBSET_FN = f"train_subset_{id_number}.pkl"
        self.TEST_SUBSET_FN = f"test_subset{id_number}.pkl"

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
            if is_test:
                with open(join(self.root_dir, self.TEST_SUBSET_FN), 'rb') as f:
                    test_imgs = pickle.load(f)
                    print(f"Test part: {len(test_imgs)}")
                    train_imgs = None
            else:
                with open(join(self.root_dir, self.TRAIN_SUBSET_FN), 'rb') as f:
                    train_imgs = pickle.load(f)
                    print(f"Train part: {len(train_imgs)}")
                    test_imgs = None

        except FileNotFoundError:
            # generate split
            train_imgs = []
            test_imgs = []
            img_classes = []
            cls_files = []
            for cls_label, cls in enumerate(self.classes):
                for dir in definition[cls]:
                    dir_files = listdir(join(self.root_dir, dir))
                    for dir_file in dir_files:
                        if isfile(join(join(self.root_dir, dir), dir_file)) and dir_file[-4:] == ".jpg":
                            cls_files.append(join(dir, dir_file))
                            img_classes.append(cls_label)

            if train_share == 1:
                test, train = [], cls_files
            elif train_share == 0:
                test, train = cls_files, []
            else:
                test, train = train_test_split(cls_files, test_size=math.ceil(len(cls_files) * train_share), shuffle=True, stratify=img_classes)

            for img in train:
                i = img_embeddings['img_names'].index(img)
                train_imgs.append(
                    [
                        torch.tensor(img_classes[cls_files.index(img)]),
                        torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float),
                        img_embeddings['img_names'][i]
                    ]
                )

            for img in test:
                i = img_embeddings['img_names'].index(img)
                test_imgs.append(
                    [
                        torch.tensor(img_classes[cls_files.index(img)]),
                        torch.tensor(img_embeddings['embeddings'][i], dtype=torch.float),
                        img_embeddings['img_names'][i]
                    ]
                )

            ###################################################################################
            # split statistics calculation:
            # lets count number of test items in each class
            test_cls_counter = {}

            for test_image in test:
                image_folder = test_image[:test_image.index("/")]
                for cls in definition.keys():
                    if cls == PSDS.MAIN_DESCRIPTION_KEY:
                        continue
                    elif image_folder in definition[cls]:
                        if cls in test_cls_counter:
                            test_cls_counter[cls] += 1
                        else:
                            test_cls_counter[cls] = 1
            train_cls_counter = {}

            for test_image in train:
                image_folder = test_image[:test_image.index("/")]
                for cls in definition.keys():
                    if cls == PSDS.MAIN_DESCRIPTION_KEY:
                        continue
                    elif image_folder in definition[cls]:
                        if cls in train_cls_counter:
                            train_cls_counter[cls] += 1
                        else:
                            train_cls_counter[cls] = 1

            print(f"DS Total {len(cls_files)}, splited to Train {len(train)} : {train_cls_counter}, Test {len(test)} : {test_cls_counter}")
            ####################################################################################
            with open(join(self.root_dir, self.TEST_SUBSET_FN), 'wb') as f:
                pickle.dump(test_imgs, f)

            with open(join(self.root_dir, self.TRAIN_SUBSET_FN), 'wb') as f:
                pickle.dump(train_imgs, f)

        self.imgs = test_imgs if is_test else train_imgs

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index][1], self.imgs[index][0], self.imgs[index][2]

    def get_classes_number(self):
        return len(self.classes)

    def get_classes(self):
        return self.classes


def accuracy_by_class(y_true, y_pred):

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    # Calculate the accuracy for each one of our classes
    #print("unique true classes", np.unique(y_true))
    preds_aggregated = Counter(y_true)
    for idx, cls in enumerate(np.unique(y_true)):
        # True negatives are all the samples that are not our current GT class (not the current row)
        # and were not predicted as the current class (not the current column)
        # true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]

        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[cls] = round(100 * true_positives / preds_aggregated[cls], 2)

    per_class_accuracies['avg'] = round(100 * accuracy_score(y_true, y_pred), 2)
    return per_class_accuracies


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    optimizer.zero_grad(set_to_none=True)
    per_class_accuracies = {}
    all_true_labels = np.array([], dtype='int16')
    all_preds = np.array([], dtype='int16')
    even_time = time.time()
    for inputs, labels, _ in loader:
        odd_time = time.time()
        # print(f"Train:reading from loader {abs(odd_time - even_time):.1f} seconds ")
        all_true_labels = np.hstack((all_true_labels, labels))

        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        even_time = time.time()
        # print(f"Train:uploaded train data to GPU {abs(odd_time - even_time):.1f} seconds ")
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        odd_item = time.time()
        # print(f"Train:model predicted outputs {abs(odd_item - even_time):.1f} seconds")
        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        #running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        optimizer.zero_grad(set_to_none=True)
        all_preds = np.hstack((all_preds, preds.detach().cpu()))

    train_loss = running_loss / processed_size
    #train_acc = running_corrects.cpu().numpy() / processed_size
    acc = accuracy_by_class(all_true_labels, all_preds)
    even_item = time.time()
    # print(f"Train:exiting train() {abs(odd_item - even_time):.1f} seconds")

    return train_loss, acc


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    per_class_accuracies = {}

    all_true_labels = np.array([], dtype='int16')
    all_preds = np.array([], dtype='int16')
    for inputs, labels, _ in val_loader:
        all_true_labels = np.hstack((all_true_labels, labels))

        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

        # all_true_labels = np.append(all_true_labels, labels.data.cpu().numpy().astype('int16'))
        # all_preds = np.append(all_preds, preds.cpu().numpy().astype('int16'))
        all_preds = np.hstack((all_preds, preds.cpu()))

    val_loss = running_loss / processed_size
    #val_acc = running_corrects.cpu().numpy() / processed_size
    f1b = f1_score(all_true_labels, all_preds, average='weighted')

    return val_loss, accuracy_by_class(all_true_labels, all_preds), f1b, confusion_matrix(all_true_labels, all_preds)


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


def train(train_loader, val_loader, model, epochs, lr, no_improvement_epochs_to_stop=50):

    best_accuracy = 0
    best_loss = np.inf
    best_loss_cfm = None
    best_model_weights = None
    epochs_with_no_improvement = 0
    test_loss = 0
    f1 = 0
    test_acc = {}

    opt = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    #scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=3, min_lr=0.00005)# was 1
    scheduler = StepLR(opt, step_size=105, gamma=0.5)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, opt)

        test_loss, test_acc, f1, cfm = eval_epoch(model, val_loader, criterion)

        if test_loss < best_loss:
            best_loss = test_loss
            best_loss_cfm = cfm
            # best_model_weights = copy.deepcopy(model.state_dict())
            epochs_with_no_improvement = 0
        else:
            epochs_with_no_improvement += 1

        if epoch != 0:
            lr = scheduler._last_lr[-1]

        scheduler.step()# scheduler.step(f1)
        if epochs_with_no_improvement >= no_improvement_epochs_to_stop:
            print(f"Reached {no_improvement_epochs_to_stop} epochs with no improvement, ending training")
            break

        print(f"Epoch {epoch:02d} lr {lr:0.6f} train_loss {train_loss:0.3f} train_acc {train_acc} test_loss {test_loss:0.3f} test_acc {test_acc} f1 {f1:0.3f} T {((time.time() - epoch_start_time) / 60):0.1f}")

    return best_accuracy, best_loss, best_model_weights, best_loss_cfm


if __name__ == '__main__':

    ds_definition = {
        "main": "d:/AITests/EbaySet/parsingTest24/primers/",
        "2sides": ["2sided" ],# 10k
        "skews": ["skews"],# 20K
        "ras": ["ras", "ras_2"], # 20K
        "goods": ["goods", "goods_2" ],# 24 K
        "multis": ["multis", "multis_2", "multis_3", "multis_greetings", "mutis_skewed_greetings"]# 35K
    }

    ds_definition = {
        "main": "d:/AITests/EbaySet/parsingTest24/primers/",
        "bads": ["skews", "skews_foreign_objects", "2sided", "2sided_augmented", "multis", "multis_2", "multis_3", "multis_greetings", "mutis_skewed_greetings"],#
        "goods": ["ras", "ras_2", "ras_augmented", "ras_2_augmented", "goods", "goods_2", "goods_augmented", "goods_2_augmented"],#
    }
    train_ds = PSDS(ds_definition, False, 0, 0.85)
    test_ds = PSDS(ds_definition, True, 0, 0.15)

    num_workers = 0
    n_epochs = 400
    image_features_size = 512

    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on", DEVICE)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=len(train_ds),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=len(test_ds),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )

    model = PDClassifier(image_features_size, train_ds.get_classes_number()).to(DEVICE)
    best_accuracy, best_loss, best_model_weights, confusion_mtrx = train(
        train_loader,
        test_loader,
        model,
        n_epochs,
        0.0005# 0.01 0.0005
    )
    
    print(confusion_mtrx)

    try_ds_definition = {
        "main": "d:/AITests/AITest_goods4",
        "goods": ["goods4"]
    }

    try_ds_dir = try_ds_definition["main"]
    torch.save(best_model_weights, join(try_ds_dir, "model_weights.pth"))
    #model.load_state_dict(torch.load(join(try_ds_dir, "model_weights.pth")))
    assert False

    try_ds = PSDS(try_ds_definition, False, 2, train_share=1.0)
    try_loader = DataLoader(
        dataset=try_ds,
        batch_size=len(try_ds),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=None,
        pin_memory=True
    )

    img_names, img_predicted_cls = predict_classes(model, try_loader)
    target_dir = "d:/AITests/AITest_goods4/try_ds/"
    for id, img in enumerate(img_names):
        try:
            shutil.copy(
                join(try_ds_dir, img),
                join(target_dir + str(img_predicted_cls[id]), img)
            )
        except:
            print("Error copying file:", join(try_ds_dir, img), join(target_dir + str(img_predicted_cls[id]), img))
