import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import genpareto

from utility.clustering import final_clusters, create_list_index, generate_cluster_with_em

import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter

import sys

import random



def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

def create_combined_datasets(params, d1, d2):
    seed = params['seed']
    perc_train_split_g = params['split_train']
    perc_val_split_g = params['split_val']
    log_file_path = params['log_path']

    np.random.seed(seed)

    seqs1, n_feats1 = create_sequences(d1, params['seq_len'])
    seqs2, n_feats2 = create_sequences(d2, params['seq_len'])

    labels1 = generate_cluster_with_em(seqs1, seed)
    labels2 = generate_cluster_with_em(seqs2, seed)

    train1_idx, val1_idx = np.where(labels1 == 0)[0], np.where(labels1 == 1)[0]
    train2_idx, val2_idx = np.where(labels2 == 0)[0], np.where(labels2 == 1)[0]

    train1, val1 = seqs1[train1_idx, :, :], seqs1[val1_idx, :, :]
    train2, val2 = seqs2[train2_idx, :, :], seqs2[val2_idx, :, :]

    if n_feats1 > n_feats2:
        params['n_feats'] = n_feats1
        dif = n_feats1 - n_feats2
        train2 = np.pad(train2, ((0, 0), (0, 0), (0, dif)), 'constant')
        val2 = np.pad(val2, ((0, 0), (0, 0), (0, dif)), 'constant')
    elif n_feats2 > n_feats1:
        params['n_feats'] = n_feats2
        dif = n_feats2 - n_feats1
        train1 = np.pad(train1, ((0, 0), (0, 0), (0, dif)), 'constant')
        val1 = np.pad(val1, ((0, 0), (0, 0), (0, dif)), 'constant')
    else:
        params['n_feats'] = n_feats1
        print('No padding')

    training = np.concatenate((train1, train2), axis=0)
    validation = np.concatenate((val1, val2), axis=0)

    np.random.shuffle(training)
    np.random.shuffle(validation)

    min_ = [training[:, :, i].min() for i in range(training.shape[2])]
    max_ = [training[:, :, i].max() for i in range(training.shape[2])]

    for i in range(training.shape[2]):
        training[:, :, i] = (training[:, :, i] - min_[i]) / (max_[i] - min_[i])
        validation[:, :, i] = (validation[:, :, i] - min_[i]) / (max_[i] - min_[i])

    _, train_g = train_test_split(training, test_size=perc_train_split_g, random_state=seed)
    _, val_g = train_test_split(validation, test_size=perc_val_split_g, random_state=seed)

    labels_train_g = np.ones(train_g.shape[0])
    labels_val_g = -1 * np.ones(val_g.shape[0])

    all_g = np.concatenate((train_g, val_g))
    all_labels_g = np.concatenate((labels_train_g, labels_val_g))

    temp = list(zip(all_g, all_labels_g))
    temp = shuffle(temp, random_state=seed)
    all_g, all_labels_g = zip(*temp)

    train_m, val_m = torch.Tensor(training), torch.Tensor(validation)
    all_g = torch.Tensor(all_g)
    all_labels_g = torch.Tensor(all_labels_g)
    train_g, val_g = torch.Tensor(train_g), torch.Tensor(val_g)

    with open(log_file_path, 'a') as filehandle:
        tot_train = train_m.shape
        tot_val = val_m.shape

        tot_train_G = train_g.shape
        tot_val_G = val_g.shape
        filehandle.write(f"Number of elements in training set M: {tot_train} \n")
        filehandle.write(f"Number of elements in validation set M: {tot_val} \n")

        filehandle.write(f"Number of elements in training set G: {tot_train_G} \n")
        filehandle.write(f"Number of elements in validation set G: {tot_val_G} \n")


    print("Number of elements in training set M:", tot_train, "\n")
    print("Number of elements in validation set M:", tot_val, "\n")
    print("Number of elements in training set G:", tot_train_G, "\n")
    print("Number of elements in validation set G:", tot_val_G, "\n")

    return train_m, val_m, all_g, train_g, val_g, all_labels_g

def create_sequences(df, seq_len=60):
    n_feats = df.shape[1]
    n_seqs = df.shape[0]//seq_len
    seqs = np.empty((n_seqs, seq_len, n_feats))

    for i in range(0, n_seqs):
        start_idx = i * seq_len
        end_idx = (i + 1) * seq_len
        sample = df.iloc[start_idx:end_idx].values
        seqs[i] = sample

    return seqs, n_feats

def create_clusters(seqs, params):
    labels, cluster_c, dend = final_clusters(seqs, params)
    idx_train, idx_val = create_list_index(cluster_c, dend, params)

    with open(params['idx_clustering_path'], "wb") as f:
        pickle.dump((labels, idx_train, idx_val), f)

    return labels, idx_train, idx_val

def create_train_val_sets(seqs, params):
    seed = params['seed']
    np.random.seed(seed)
    perc_split_m = params['perc_split_m']
    perc_train_split_g = params['split_train']
    perc_val_split_g = params['split_val']
    clustering = params['clustering']

    if clustering:
        if params['train_m']:
            labels, idx_train, idx_val = create_clusters(seqs, params)
        else:
            with open(params['idx_clustering_path'], "rb") as f:
                labels, idx_train, idx_val = pickle.load(f)

        train_m_idxs, val_m_idxs = [], []

        for idx in idx_train:
            train_m_idxs.extend(np.where(labels == idx)[0])

        for idx in idx_val:
            val_m_idxs.extend(np.where(labels == idx)[0])

        train_m, val_m = seqs[train_m_idxs, :, :], seqs[val_m_idxs, :, :]
    else:
        train_m, val_m = train_test_split(seqs, test_size=perc_split_m, random_state=seed)

    min_ = [train_m[:, :, i].min() for i in range(train_m.shape[2])]
    max_ = [train_m[:, :, i].max() for i in range(train_m.shape[2])]

    for i in range(train_m.shape[2]):
        train_m[:, :, i] = (train_m[:, :, i] - min_[i]) / (max_[i] - min_[i])
        val_m[:, :, i] = (val_m[:, :, i] - min_[i]) / (max_[i] - min_[i])

    _, train_g = train_test_split(train_m, test_size=perc_train_split_g, random_state=seed)
    _, val_g = train_test_split(val_m, test_size=perc_val_split_g, random_state=seed)

    labels_train_g = np.ones(train_g.shape[0])
    labels_val_g = -1 * np.ones(val_g.shape[0])

    all_g = np.concatenate((train_g, val_g))
    all_labels_g = np.concatenate((labels_train_g, labels_val_g))

    temp = list(zip(all_g, all_labels_g))
    temp = shuffle(temp, random_state=seed)
    all_g, all_labels_g = zip(*temp)

    train_m, val_m = torch.Tensor(train_m), torch.Tensor(val_m)
    all_g = torch.Tensor(all_g)
    all_labels_g = torch.Tensor(all_labels_g)
    train_g, val_g = torch.Tensor(train_g), torch.Tensor(val_g)

    return train_m, val_m, all_g, train_g, val_g, all_labels_g


def split_sequence(seqs, ratio=.5):
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition, we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        seqs: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    # x = (0,1,2,3)
    # y = (4,5,6,7)
    # tgt = (3,4,5,6)

    # y = (4,5,6,7)
    # x = (0,1,2,3)
    # tgt = (4,3,2,1)

    x_end = int(seqs.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    x = seqs[:, :x_end]
    # [bs, tgt_seq_len, num_features]
    y_shift = seqs[:, x_end - 1:-1]
    # [bs, tgt_seq_len, num_features]
    y = seqs[:, x_end:]

    return x, y_shift, y


def get_dataloaders(params, df=None, d2=None):
    if params['combined']:
        train_m, val_m, all_g, train_g, val_g, all_labels_g = create_combined_datasets(params, df, d2)
    else:
        sequences, n_feats = create_sequences(df, params['seq_len'])
        train_m, val_m, all_g, train_g, val_g, all_labels_g = create_train_val_sets(sequences, params)

    x_train_g, y_shift_train_g, y_train_g = split_sequence(all_g)

    x_train_m, y_shift_train_m, y_train_m = split_sequence(train_m)
    x_val_m, y_shift_val_m, y_val_m = split_sequence(val_m)

    x_train_g_test, y_shift_train_g_test, y_train_g_test = split_sequence(train_g)
    x_val_g_test, y_shift_val_g_test, y_val_g_test = split_sequence(val_g)

    dataset_g = TensorDataset(y_train_g, x_train_g, y_shift_train_g, y_train_g, all_labels_g)
    train_dataset_m = TensorDataset(x_train_m, y_shift_train_m, y_train_m)
    val_dataset_m = TensorDataset(x_val_m, y_shift_val_m, y_val_m)
    train_dataset_g = TensorDataset(y_train_g_test, x_train_g_test, y_shift_train_g_test, y_train_g_test)
    val_dataset_g = TensorDataset(y_val_g_test, x_val_g_test, y_shift_val_g_test, y_val_g_test)

    batch_size = params['batch_size']

    dataloader_g = DataLoader(dataset_g, batch_size=batch_size, shuffle=True)
    train_loader_m = DataLoader(train_dataset_m, batch_size=batch_size, shuffle=True)
    val_loader_m = DataLoader(val_dataset_m, batch_size=batch_size, shuffle=True)
    train_loader_g = DataLoader(train_dataset_g, batch_size=batch_size, shuffle=True)
    val_loader_g = DataLoader(val_dataset_g, batch_size=batch_size, shuffle=True)

    return dataloader_g, train_loader_m, val_loader_m, train_loader_g, val_loader_g, params


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True

        return False


class ReductionData:
    def __init__(self, patience=1, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def condition_data_dropping(self, loss):
        if loss < (self.min_loss - self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.counter = 0
                return True
        return False


def check_prompt_generator(model, generator, test_loader, params):
    device = params['device']
    criterion = torch.nn.MSELoss(reduction="sum")

    generator.eval()
    model.eval()
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for x_g, y_shift_g, y_shift_m, y_m, label in test_loader:

            x_g, y_shift_g = x_g.to(device), y_shift_g.to(device)
            y_shift_m, y_m = y_shift_m.to(device), y_m.to(device)

            pred_gen = generator.infer(x_g, y_shift_g.shape[1])
            pred_mod = model.infer_m(pred_gen, y_shift_m.shape[1])

            loss = criterion(pred_mod, y_m)

            test_loss += loss.item()
            if count == 0:
                y_true = y_m.cpu()
                y_pred = pred_mod.detach().cpu()
                x_true = x_g.cpu()
                x_pred = pred_gen.detach().cpu()
                labels = label
            else:
                y_true = torch.cat((y_true, y_m.cpu()))
                y_pred = torch.cat((y_pred, pred_mod.detach().cpu()))
                x_true = torch.cat((x_true, x_g.cpu()))
                x_pred = torch.cat((x_pred, pred_gen.detach().cpu()))
                labels = torch.cat((labels, label))

            count += 1
        avg_loss_test = test_loss / count

    return avg_loss_test, y_pred, y_true, x_pred, x_true, labels


def eliminate_data(model, generator, dataloader_g, params):
    log_file_path = params['log_path']
    n = params['len_training_set']

    avg_loss_test, y_pred, y_true, x_pred, x_true, labels = check_prompt_generator(model, generator,
                                                                                        dataloader_g, params)
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error_list = list(zip(list(range(len(error))), error, labels.tolist()))
    error_sorted = sorted(error_list, key=itemgetter(1), reverse=False)
    error_sorted = np.array(error_sorted)
    errors_for_fitting = error_sorted[:, 1]
    shape, loc, scale = genpareto.fit(errors_for_fitting)

    alpha = genpareto.ppf(0.8, shape, loc, scale)

    m = np.searchsorted(errors_for_fitting, alpha)

    if m == len(errors_for_fitting):
        with open(log_file_path, 'a') as filehandle:
            filehandle.write(f'No reduction needed \n')
        print("No reduction needed")
    else:
        perc = m / n * 100

        with open(log_file_path, 'a') as filehandle:
            filehandle.write(f'Reduction of {100 - perc} % of data \n')
        print(f'Reduction {100 - perc} % of data')

        if m <= 0.3 * n:
            m = 0.3 * n
            m = int(m)
            params['reduction'] = False
            perc = m / n * 100
            with open(log_file_path, 'a') as filehandle:
                filehandle.write(f'Last reduction of {100 - perc} % of data \n')
            print(f'Last reduction {100 - perc} % of data')

        selected_indices = error_sorted[:m + 1, 0].astype(int)
        dataset = dataloader_g.dataset
        subset_dataset = torch.utils.data.Subset(dataset, selected_indices)
        dataloader_g = DataLoader(subset_dataset, batch_size=params['batch_size'], shuffle=True)

    return dataloader_g
