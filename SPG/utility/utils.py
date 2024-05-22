import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utility.clustering import final_clusters, create_list_index

import torch
from torch.utils.data import TensorDataset, DataLoader


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
    return labels, idx_train, idx_val

def create_train_val_sets(seqs, params):
    seed = params['seed']
    np.random.seed(seed)
    perc_split_m = params['perc_split_m']
    perc_train_split_g = params['split_train']
    perc_val_split_g = params['split_val']
    clustering = params['clustering']

    if clustering:
        labels, idx_train, idx_val = create_clusters(seqs, params)

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

    all_g = np.concatenate((train_g, val_g))
    all_g = shuffle(all_g, random_state=seed)

    train_m, val_m = torch.Tensor(train_m), torch.Tensor(val_m)
    all_g = torch.Tensor(all_g)
    train_g, val_g = torch.Tensor(train_g), torch.Tensor(val_g)

    return train_m, val_m, all_g, train_g, val_g


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


def get_dataloaders(df, params):
    sequences, n_feats = create_sequences(df, params['seq_len'])
    train_m, val_m, all_g, train_g, val_g = create_train_val_sets(sequences, params)

    x_train_g, y_shift_train_g, y_train_g = split_sequence(all_g)

    x_train_m, y_shift_train_m, y_train_m = split_sequence(train_m)
    x_val_m, y_shift_val_m, y_val_m = split_sequence(val_m)

    x_train_g_test, y_shift_train_g_test, y_train_g_test = split_sequence(train_g)
    x_val_g_test, y_shift_val_g_test, y_val_g_test = split_sequence(val_g)

    dataset_g = TensorDataset(y_train_g, x_train_g, y_shift_train_g, y_train_g)
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









