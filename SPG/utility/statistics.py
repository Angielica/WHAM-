import pickle
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from scipy.signal import savgol_filter
from kneebow.rotor import Rotor

from operator import itemgetter


def find_elbow(data):
    rotor = Rotor()
    rotor.fit_rotate(data)

    return rotor.get_elbow_index()

def find_low_k_copy(y_pred, y_true, labels, K=5):
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error_list = list(zip(list(range(len(error))), error, labels.tolist()))

    error_sorted = sorted(error_list, key=itemgetter(1), reverse=True)
    error_sorted = np.array(error_sorted)

    copy = error_sorted[:K]

    count_train = len(np.where(copy[:, 2] == 1.)[0])
    count_val = len(copy) - count_train

    return count_train, count_val

def find_copy_on_y_all(y_pred, y_true, labels, path_err, path_smooth):
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error_list = list(zip(list(range(len(error))), error, labels.tolist()))
    error_sorted = sorted(error_list, key=itemgetter(1), reverse=True)
    error_sorted = np.array(error_sorted)

    y = error_sorted[:, 1]
    labels = error_sorted[:, 2]
    x = range(0, len(error_sorted))

    window_size = max(5, len(y) // 5)

    polynomial_order = 2
    smoothed_y = savgol_filter(y, window_size, polynomial_order)


    y1 = np.diff(smoothed_y) / np.diff(x)
    y2 = np.diff(y1) / np.diff(x)[:-1]

    # searching the inflection point
    idx = np.abs(y2).argmin()
    right_x = x[idx:]
    right_y = smoothed_y[idx:]

    right_elbow = find_elbow(np.array(list(zip(right_x, -right_y))))
    high_thr_x = x[right_elbow + idx]
    high_thr_y = y[right_elbow + idx]

    threshold = high_thr_y

    error_list = np.array(error_list)
    copy = error_list[np.where(error_list[:, 1] < threshold)[0]]

    count_train = len(np.where(copy[:, 2] == 1.)[0])
    count_val = len(copy) - count_train

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.title("Evaluation: reconstruction error")
    plt.plot(range(0, len(error_list)), threshold * np.ones(len(error_list)), color='red')
    plt.plot(range(0, len(error_list)), sorted(error_list[:, 1], reverse=True), color='blue')
    plt.savefig(path_err)
    plt.show()

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.title("Evaluation: smooth reconstruction error")
    plt.plot(range(0, len(error_list)), threshold * np.ones(len(error_list)), color='red')
    plt.plot(range(0, len(error_list)), smoothed_y, color='orange')
    plt.savefig(path_smooth)
    plt.show()
    return count_train, count_val, threshold


def find_copy_on_y(y_pred, y_true, train, threshold, path_err, path_smooth):
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error = np.array(list(zip(range(len(error)), error)))
    count, idx = 0, 0

    y = sorted(error[:, 1], reverse=True)
    x = range(0, len(error))

    window_size = max(5, len(y) // 5)

    polynomial_order = 2
    smoothed_y = savgol_filter(y, window_size, polynomial_order)

    if train:
        y1 = np.diff(smoothed_y) / np.diff(x)
        y2 = np.diff(y1) / np.diff(x)[:-1]

        # searching the inflection point
        idx = np.abs(y2).argmin()
        right_x = x[idx:]
        right_y = smoothed_y[idx:]

        right_elbow = find_elbow(np.array(list(zip(right_x, -right_y))))
        high_thr_x = x[right_elbow + idx]
        high_thr_y = y[right_elbow + idx]

        threshold = high_thr_y

    count = len(np.where(error[:, 1] < threshold)[0])

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.title("Evaluation: reconstruction error")
    plt.plot(range(0, len(error)), threshold * np.ones(len(error)), color='red')
    plt.plot(range(0, len(error)), sorted(error[:, 1], reverse=True), color='blue')
    plt.savefig(path_err)
    plt.show()

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.title("Evaluation: smooth reconstruction error")
    plt.plot(range(0, len(error)), threshold * np.ones(len(error)), color='red')
    plt.plot(range(0, len(error)), smoothed_y, color='orange')
    plt.savefig(path_smooth)
    plt.show()
    return count, threshold


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
    
    # for err in error:
    #     if err[-1] >= threshold:
    #         count += 1
    #         tmp_error = ((x_pred[idx] - x_true[idx]) ** 2).sum(dim=-1)
    #         tmp_error = tmp_error.sum(dim=-1)
    #         error_x.append(tmp_error)
    #     idx += 1

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


def find_k_copy(y_pred, y_true, x_pred, x_true, train, threshold, path):
    # print(y_pred.shape)
    # print(x_pred.shape)
    error = (y_pred - y_true) ** 2
    error = error.sum(dim=-1)
    error = error.sum(dim=-1)
    error = error.clone().detach().cpu().numpy()

    error = np.array(list(zip(range(len(error)), error)))
    count, idx = 0, 0
    error_x = []

    # print("threshold", threshold)
    # th = threshold * np.ones(len(error))

    # for err in error:
    #     if err[-1] >= threshold:
    #         count += 1
    #         tmp_error = ((x_pred[idx] - x_true[idx]) ** 2).sum(dim=-1)
    #         tmp_error = tmp_error.sum(dim=-1)
    #         error_x.append(tmp_error)
    #     idx += 1

    error = sorted(error, key=lambda a: a[1])
    error = np.array(error)

    # print(error[:10])
    # tmp_true = y_true[error[:2,0]][0][0]
    # tmp_pred = y_pred[error[:2,0]][0][0]

    th_vec = []
    count_train = []
    count_val = []
    if train:
        threshold = sorted(error[:, 1], reverse=True)
        for k in range(1, 15):
            th = threshold[-k]
            if k == 1:
                th_vec.append(th)
            elif th != th_vec[k - 2]:
                th_vec.append(th)
            else:
                count_train.pop()
            count_train.append(k)
        print(f"count train : {count_train}")
        count = count_train
    else:
        for th in threshold:
            count = 0
            sort = sorted(error[:, 1], reverse=True)
            for err in sort:
                if err <= th:
                    count += 1
            count_val.append(count)
        print(f"count val : {count_val}")
        count = count_val

    print("threshold", th_vec)

    # print("y true\n", y_true[error[:2,0]][0][0][:10])
    # print("y pred\n", y_pred[error[:2,0]][0][0][:10])
    # print(((tmp_true - tmp_pred)**2).sum())

    plt.figure()
    plt.grid()
    plt.yscale("log")
    plt.plot(range(0, len(error)), th * np.ones(len(error)), color='red')
    plt.plot(range(0, len(error)), sorted(error[:, 1], reverse=True), color='blue')

    # plt.axvline(threshold, ymin=min(sorted(error[:, 1], reverse=True)), ymax=max(sorted(error[:, 1], reverse=True)))
    plt.savefig(path)

    return count, error_x, th_vec


def statistics(params):
    path_pred_train = params['path_y_prediction_train']
    path_pred_val = params['path_y_prediction_val']
    path_saved_train = params['path_out_log_train']
    path_saved_val = params['path_out_log_val']
    path_saved_train_smooth = params['path_out_log_train_smooth']
    path_saved_val_smooth = params['path_out_log_val_smooth']
    log_file_path = params['log_path']

    with open(path_pred_train, "rb") as f:
        y_pred_t, y_true_t, x_pred_t, x_true_t = pickle.load(f)

    with open(path_pred_val, "rb") as f:
        y_pred_v, y_true_v, x_pred_v, x_true_v = pickle.load(f)

    print(f'Shape y train: {y_true_t.shape}, shape y val: {y_true_v.shape}')

    # train
    train = 1
    threshold = 0
    count_train, threshold = find_copy_on_y(y_pred_t, y_true_t, train, threshold, path_saved_train, path_saved_train_smooth)
    print(f"Count copy find in training set: {count_train} \n")

    # val
    train = 0
    count_val, threshold = find_copy_on_y(y_pred_v, y_true_v, train, threshold, path_saved_val, path_saved_val_smooth)
    print(f"Count copy find in validation set: {count_val} \n")

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"Count copy find in training set: {count_train} \n")
        filehandle.write(f"Count copy find in validation set: {count_val} \n")

    return count_train, count_val


def statistics_all(params):
    path_pred_g = params['path_y_prediction_g']
    path_saved_g = params['path_out_log_g']
    path_saved_g_smooth = params['path_out_log_g_smooth']
    log_file_path = params['log_path']

    with open(path_pred_g, "rb") as f:
        y_pred_t, y_true_t, x_pred_t, x_true_t, labels = pickle.load(f)

    shape_train = len(np.where(labels == 1.)[0])
    shape_val = len(np.where(labels == -1.)[0])

    print(f'Shape y (G): {y_true_t.shape}')
    print(f'shape D+: {shape_train}, shape D-: {shape_val}')

    if params['compute_elbow_metric']:
        count_train, count_val, threshold = find_copy_on_y_all(y_pred_t, y_true_t, labels, path_saved_g, path_saved_g_smooth)
    else:
        count_train, count_val = find_low_k_copy(y_pred_t, y_true_t, labels, K=params['k'])

    print(f"Count copy find in training set: {count_train} \n")
    print(f"Count copy find in validation set: {count_val} \n")

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"Count copy find in training set: {count_train} \n")
        filehandle.write(f"Count copy find in validation set: {count_val} \n")

    return count_train, count_val


