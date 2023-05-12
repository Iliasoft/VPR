import pickle
from collections import OrderedDict


import sklearn
from scipy import stats
from transformers import AdamW, get_cosine_schedule_with_warmup

# sys.argv = ['--config', 'config8']

# from conf import
from src.utils import *
from src.data import *
from src.models import *
from src.loss import *
import pandas as pd
import pytorch_lightning as pl


def fix_row(row):
    if len(str(row).split()) > 1:
        row = int(str(row).split()[0])
    return row


def setup():
    if args.seed == -1:
        args.seed = np.random.randint(0, 1000000)

    set_seed(args.seed)



    print('Load train DS from:', args.data_path + args.train_csv_fn)
    train = pd.read_csv(args.data_path + args.train_csv_fn)
    train["img_folder"] = args.img_path_train

    print('Load valid DS from:', args.data_path_2019 + args.valid_csv_fn)
    valid = pd.read_csv(args.data_path_2019 + args.valid_csv_fn)
    valid["img_folder"] = args.img_path_val
    valid['landmarks'] = valid['landmarks'].apply(lambda x: fix_row(x))
    valid['landmark_id'] = valid['landmarks'].fillna(-1)
    valid['landmarks'].fillna('', inplace=True)
    valid['landmark_id'] = valid['landmark_id'].astype(int)

    if args.data_path_2 is not None:
        train_2 = pd.read_csv(args.data_path_2 + args.train_2_csv_fn)
        train_2["img_folder"] = args.img_path_train_2
        if "gldv1" in args.data_path_2:
            train_2["landmark_id"] = train_2["landmark_id"] + train["landmark_id"].max()
        train = pd.concat([train, train_2], axis=0).reset_index(drop=True)

    # contains only  those records from train where  landmarkd_id exists  in valid  set
    train_filter = train[train.landmark_id.isin(valid.landmark_id)].reset_index()

    # unique list  of  all landmarks_ids
    landmark_ids = np.sort(train.landmark_id.unique())

    # num of classes goes from number of  landmark_ids
    args.n_classes = train.landmark_id.nunique()

    # dict of { landmark_id : index (class) }
    landmark_id2class = {lid: i for i, lid in enumerate(landmark_ids)}
    # copy
    landmark_id2class_val = landmark_id2class.copy()

    landmark_id2class_val[-1] = args.n_classes  # - 1  ####!!!!!! ALL UNKNOWN  LANDMARKS mapped to last  class

    train['target'] = train['landmark_id'].apply(lambda x: landmark_id2class[x])

    if args.class_weights == "log":
        val_counts = train.target.value_counts().sort_index().values
        class_weights = 1 / np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    valid['target'] = valid['landmark_id'].apply(lambda x: landmark_id2class_val.get(x, -1))
    valid = valid[valid.target > -1].reset_index(drop=True)

    allowed_classes = np.sort(valid[valid.target != args.n_classes].target.unique())

    train_filter['target'] = train_filter['landmark_id'].apply(lambda x: landmark_id2class_val.get(x, -1))

    # train = train.head(args.batch_size*2)

    return train, valid, train_filter, landmark_ids, landmark_id2class, landmark_id2class_val, class_weights, allowed_classes


class Model(pl.LightningModule):

    def __init__(self, hparams, tr_dl, val_dl, tr_filter_dl, train_filter, metric_crit, metric_crit_val,
                 allowed_classes):
        super(Model, self).__init__()

        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.tr_filter_dl = tr_filter_dl
        self.train_filter = train_filter
        self.metric_crit = metric_crit
        self.metric_crit_val = metric_crit_val
        self.allowed_classes = torch.Tensor(allowed_classes).long()

        self.params = hparams
        if args.distributed_backend == "ddp":
            self.num_train_steps = math.ceil(
                len(self.tr_dl) / (len(args.gpus.split(',')) * args.gradient_accumulation_steps))
        else:
            self.num_train_steps = math.ceil(len(self.tr_dl) / args.gradient_accumulation_steps)

        self.model = Net(args)

    def forward(self, x, get_embeddings=False):
        return self.model(x, get_embeddings)

    def configure_optimizers(self):

        if args.optimizer == "adamw":
            self.optimizer = AdamW([{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}],
                                   lr=self.params.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}], lr=self.params.lr,
                momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

        elif args.optimizer == "fused_sgd":
            import apex
            self.optimizer = apex.optimizers.FusedSGD(
                [{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}], lr=self.params.lr,
                momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

        if args.scheduler["method"] == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.num_train_steps * args.scheduler[
                                                                 "warmup_epochs"],
                                                             num_training_steps=int(
                                                                 self.num_train_steps * (args.max_epochs)))
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]
        elif args.scheduler["method"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler["step_size"],
                                                             gamma=args.scheduler["gamma"], last_epoch=-1)
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'epoch'}]
        elif args.scheduler["method"] == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode="max",
                                                                        patience=1, verbose=True)
            return [self.optimizer], [
                {'scheduler': self.scheduler, 'interval': 'epoch', 'reduce_on_plateau': True, 'monitor': 'val_gap_pp'}]
        else:
            self.scheduler = None
            return [self.optimizer]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

    def train_dataloader(self):
        return self.tr_dl

    def training_step(self, batch, batcn_num):
        input_dict, target_dict = batch

        output_dict = self.forward(input_dict)
        loss = loss_fn(self.metric_crit, target_dict, output_dict)

        if args.arcface_s is None:
            s = self.metric_crit.s.detach().cpu().numpy()
        elif args.arcface_s == -1:
            s = 0
        else:
            s = self.metric_crit.s

        if args.distributed_backend == "ddp":
            step = self.global_step * args.batch_size * len(args.gpus.split(',')) * args.gradient_accumulation_steps
        else:
            step = self.global_step * args.batch_size * args.gradient_accumulation_steps

        tb_dict = {'train_loss': loss, 'arcface_s': s, 'step': step}

        for i, param_group in enumerate(self.optimizer.param_groups):
            tb_dict[f'lr/lr{i}'] = param_group['lr']

        output = OrderedDict({
            'loss': loss,
            'log': tb_dict,

        })

        return output

    def validation_step(self, batch, batch_nb, dataset_idx):
        if dataset_idx == 0:
            input_dict, target_dict = batch
            output_dict = self.forward(input_dict, get_embeddings=True)
            loss = loss_fn(self.metric_crit_val, target_dict, output_dict, val=True)  # .data.cpu().numpy()

            logits = output_dict['logits']
            embeddings = output_dict['embeddings']

            preds_conf, preds = torch.max(logits.softmax(1), 1)

            allowed_classes = self.allowed_classes.to(logits.device)

            preds_conf_pp, preds_pp = torch.max(logits.gather(1, allowed_classes.repeat(logits.size(0), 1)).softmax(1),
                                                1)
            preds_pp = allowed_classes[preds_pp]

            targets = target_dict['target']

            output = dict({
                'idx': input_dict['idx'],
                'embeddings': embeddings,
                'val_loss': loss.view(1),
                'preds': preds,
                'preds_conf': preds_conf,
                'preds_pp': preds_pp,
                'preds_conf_pp': preds_conf_pp,
                'targets': targets,

            })

            return output
        else:
            input_dict, target_dict = batch
            targets = target_dict['target']
            output_dict = self.forward(input_dict, get_embeddings=True)
            embeddings = output_dict["embeddings"]
            output = dict({
                'idx': input_dict['idx'],
                'embeddings': embeddings,
                'targets': targets,
            })
            return output


def make_predictions(dl, _model_path, _threshold=0):
    model = Net(args).to(device)
    ds_len = len(dl.dataset)

    print('model device ', next(model.parameters()).is_cuda)
    print('Loading model from ', _model_path)
    print('DataSource len', ds_len)
    print('Model confidence threshold ', _threshold)

    emb_size = 512
    logits_file = f'{experiment_path}{args.experiment_name}_logits_r{ds_len}_e{args.max_epochs}.pth'

    model.load_state_dict(torch.load(_model_path))

    pred_result = {'idx': torch.zeros(0, dtype=int),
                   'target': torch.zeros(0, dtype=int),
                   'logits': torch.zeros(0, args.n_classes, dtype=float),
                   'max_pred_prob': torch.zeros(0, dtype=float),
                   'pred_class': torch.zeros(0, dtype=int),
                   'embeddings': torch.zeros(size=(0, emb_size), dtype=float),
                   'image_names': []
                   }

    model.cuda()
    model.eval()

    ldm_probabilities = []
    not_ldm_probabilities = []
    target_cleaned = []
    pred_class_cleaned = []

    with torch.no_grad():
        batch_id = 0
        for batch in tqdm(iter(dl)):
            input_dict, target_dict = batch

            input_dict['input'] = input_dict['input'].to(device)
            input_dict['idx'] = input_dict['idx'].to(device)
            target_dict['target'] = target_dict['target'].to(device)
            idx = input_dict['idx']

            output_dict = model.forward(input_dict, get_embeddings=True)  # ["embeddings"]
            embeddings = output_dict['embeddings']

            # row model output [batch_size, num_classes]
            logits = output_dict['logits']  # .softmax(1)

            # [ batch_size,] class predicted  for  every image  in batch
            max_pred_prob, pred_class = logits.max(1)  # predicted  classes argmax

            # [batch_size,] max prob for predicted for  each  sample in batch
            # max_pred_prob = logits.max(1).values

            target = target_dict['target'].to(device)

            for p_ind in range(len(max_pred_prob)):
                # img_ind = input_dict[p_ind]

                l = None
                if landmark_id2class_val[-1] == target[p_ind]:
                    #  this picture is  not  landmark
                    l = not_ldm_probabilities
                else:
                    l = ldm_probabilities

                l.append(max_pred_prob[p_ind].item())

            # pred_result['logits'] = torch.cat((pred_result['logits'].cpu(), logits.cpu()))
            pred_result['target'] = torch.cat((pred_result['target'].cpu(), target.cpu()))
            pred_result['idx'] = torch.cat((pred_result['idx'].cpu(), idx.cpu()))

            # maximum  predicted  probability  for  each  image
            # for  ex for  class 0 the  max prob was predicted 0.4 for  class 10
            # 0.4  goes  to pred_result['pred_prob']  and  10 goes to pred_result['pred_class']
            pred_result['max_pred_prob'] = torch.cat((pred_result['max_pred_prob'].cpu(), max_pred_prob.cpu()))
            pred_result['pred_class'] = torch.cat((pred_result['pred_class'].cpu(), pred_class.cpu()))
            pred_result['embeddings'] = torch.cat((pred_result['embeddings'].cpu(), embeddings.cpu()))
            # for im_ds_idx in idx:

            pred_result['image_names'].append(dl.dataset.image_names[idx.cpu()])
            assert len(pred_result['image_names']) == len(pred_result['idx'])

            for img_in_batch_idx in range(len(max_pred_prob)):
                if max_pred_prob[img_in_batch_idx] > _threshold:
                    target_cleaned.append(target[img_in_batch_idx].item())
                    pred_class_cleaned.append(pred_class[img_in_batch_idx].item())

    if len(target_cleaned) > 0:
        l = sorted(set(target_cleaned).union(pred_class_cleaned))
        confusion_matrix = sklearn.metrics.confusion_matrix(target_cleaned, pred_class_cleaned, labels=l)

        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        # TN = confusion_matrix.values.sum() - (FP + FN + TP)

        sk_prec = sklearn.metrics.precision_score(target_cleaned, pred_class_cleaned, average='macro')
        sk_recall = sklearn.metrics.recall_score(target_cleaned, pred_class_cleaned, average='macro')
        sk_acc = sklearn.metrics.accuracy_score(target_cleaned, pred_class_cleaned)

        print('Average sklearn  precision:', sk_prec)
        print('Average sklearn recall:', sk_recall)
        print('Average sklearn accuracy:', sk_acc)

    with open(f'{experiment_path}{args.experiment_name}_ldm_probabilities', 'wb') as f:
        pickle.dump(ldm_probabilities, f)

    with open(f'{experiment_path}{args.experiment_name}_not_ldm_probabilities', 'wb') as f:
        pickle.dump(not_ldm_probabilities, f)

    if len(ldm_probabilities) > 0:
        print("Median1:", np.median(ldm_probabilities))
        print("Mode1:", stats.mode(ldm_probabilities, keepdims=True))
    else:
        print('Mode1: is empty')
        print('Median1: is empty')

    if len(not_ldm_probabilities) > 0:
        print("Median2:", np.median(not_ldm_probabilities))
        print("Mode2:", stats.mode(not_ldm_probabilities, keepdims=True))
    else:
        print('Median2: is empty')
        print('Mode2: is empty')

    with open(logits_file, 'wb') as f:
        pickle.dump(pred_result, f)
        print(f'Predictions result saved into {logits_file}')

    return pred_result, sk_prec, sk_recall, sk_acc


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('device ', device)
    print(torch.cuda.device)

    train, valid, train_filter, landmark_ids, landmark_id2class, landmark_id2class_val, class_weights, allowed_classes = setup()

    if args.data_frac < 1.:
        train = train.sample(frac=args.data_frac)

    if args.loss == 'bce':
        metric_crit = nn.CrossEntropyLoss()
        metric_crit_val = nn.CrossEntropyLoss(weight=None, reduction="sum")
    else:
        metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
        metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit="bce", weight=None, reduction="sum")

    print(f'Load  train  {len(train)}')
    print(f'Load  valid  {len(valid)}')

    tr_ds = GLRDataset(train, normalization=args.normalization, preload=False, txt_embedding_fn=args.text_embeddings_fn,
                       aug=args.tr_aug)
    val_ds = GLRDataset(valid, normalization=args.normalization, preload=False,
                        txt_embedding_fn=args.text_embeddings_fn, aug=args.val_aug)

    print('batch_size', args.batch_size)
    # print (tr_ds['input'].size())
    pin_memory = True
    tr_dl = DataLoader(dataset=tr_ds, batch_size=args.batch_size, sampler=RandomSampler(tr_ds), collate_fn=collate_fn,
                       num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory)

    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size,
                        sampler=SequentialSampler(val_ds),
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=pin_memory)

    tr_filter_ds = GLRDataset(train_filter, normalization=args.normalization, aug=args.val_aug)
    tr_filter_dl = DataLoader(dataset=tr_filter_ds, batch_size=args.batch_size,
                              sampler=SequentialSampler(tr_filter_ds),
                              collate_fn=collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=pin_memory)

    experiment_path = args.model_path + args.experiment_name + '\\'
    model_path = f'{experiment_path}{args.experiment_name}_ckpt_{args.max_epochs}.pth'
    model_path = f'{experiment_path}{args.experiment_name}_ckpt_{args.max_epochs}.pth'

    # print('')
    # merics_result = []
    # for tr in np.arange(0.6, 0.7, 0.01):
    #     print(f"START THRESHOLD {tr:.4}")
    #
    #     pred_result, prec, recall, acc = make_predictions(val_dl, model_path, _threshold=tr)
    #     merics_result.append((tr, prec, recall, acc))

    pred_result, prec, recall, acc = make_predictions(val_dl, model_path)
    with open(f'E:/ftp/data/Models/config10/config10_vpr_pred_result.pkl', 'wb') as f:
        pickle.dump(pred_result, f)

    print('---->')

    exit()
