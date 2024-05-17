import pickle
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator


def find_copy(y_pred, y_true, x_pred, x_true, train, threshold, path):
    # print(y_pred.shape)
    # print(x_pred.shape)
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error = np.array(list(zip(range(len(error)), error)))
    count, idx = 0, 0
    error_x = []

    if train:
        kl = KneeLocator(range(0, len(error)), sorted(error[:, 1], reverse=True), curve="concave",
                         direction="decreasing")
        threshold = kl.elbow

    # print("threshold", threshold)
    # th = threshold * np.ones(len(error))
    
    for err in error:
        if err[-1] >= threshold:
            count += 1
            tmp_error = ((x_pred[idx] - x_true[idx]) ** 2).sum(dim=-1)
            tmp_error = tmp_error.sum(dim=-1)
            error_x.append(tmp_error)
        idx += 1

    error = sorted(error, key=lambda a: a[1])
    error = np.array(error)

    # print(error[:10])
    # tmp_true = y_true[error[:2,0]][0][0]
    # tmp_pred = y_pred[error[:2,0]][0][0]

    count = 0
    sort = sorted(error[:, 1], reverse=True)
    if train:
        th = sort[threshold]
    else: 
        th = threshold
    print("threshold", th)
    for err in sort:
        if err <= th:
            count += 1

    # print("y true\n", y_true[error[:2,0]][0][0][:10])
    # print("y pred\n", y_pred[error[:2,0]][0][0][:10])
    # print(((tmp_true - tmp_pred)**2).sum())

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.plot(range(0, len(error)), th * np.ones(len(error)), color='red')
    plt.plot(range(0, len(error)), sorted(error[:, 1], reverse=True), color='blue')
  
    #plt.axvline(threshold, ymin=min(sorted(error[:, 1], reverse=True)), ymax=max(sorted(error[:, 1], reverse=True)))
    plt.savefig(path)

    return count, error_x, th


def statistics(params):
    path_pred_train = params['path_y_prediction_train']
    path_pred_val = params['path_y_prediction_val']
    path_saved_train = params['path_out_log_train']
    path_saved_val = params['path_out_log_val']
    log_file_path = params['log_path']

    with open(path_pred_train, "rb") as f:
        y_pred_t, y_true_t, x_pred_t, x_true_t = pickle.load(f)

    with open(path_pred_val, "rb") as f:
        y_pred_v, y_true_v, x_pred_v, x_true_v = pickle.load(f)

    # train
    train = 1
    threshold = 0
    count_train, error_x_train, threshold = find_copy(y_pred_t, y_true_t, x_pred_t, x_true_t, train, threshold,
                                                      path_saved_train)
    print(f"Count copy find in training set: {count_train} \n")

    # val
    train = 0
    count_val, error_x_val, threshold = find_copy(y_pred_v, y_true_v, x_pred_v, x_true_v, train, threshold,
                                                  path_saved_val)
    print(f"Count copy find in validation set: {count_val} \n")

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"Count copy find in training set: {count_train} \n")
        filehandle.write(f"Count copy find in validation set: {count_val} \n")

    return count_train, count_val


