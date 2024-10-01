import math
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
import time
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
from kneed import KneeLocator

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

import pickle


def divide_with_hierarchical_clustering(sequences, params):
    path_dendograms = params["path_dendograms"]

    X = sequences.copy()
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    w_linkage = linkage(X, method="complete")
    plt.figure(figsize=(15, 5))
    plt.title("Dendogram of Hierarchical Clustering")
    dendrogram(w_linkage)
    plt.savefig(path_dendograms)
    # plt.show()

    search = True
    i = 2
    total = X.shape[0]
    while search:
        tot = 0
        tmp = []
        idx = []
        n_train = []
        n_val_m = []
        n_val_g = []
        condition = False

        clusters = fcluster(w_linkage, i, criterion='maxclust')
        for j in range(1, i + 1):
            tmp.append(len(np.where(clusters == j)[0]))
            idx.append(np.where(clusters == j))
        for k in range(len(tmp)):
            if not condition:
                if tmp[k] < 0.30 * total:
                    tot += tmp[k]
                    n_train.append(k + 1)
                    if 0.30 * total <= tot <= 0.35 * total:
                        condition = True
                        tot = 0
                    elif tot > 0.35 * total:
                        break
                elif 0.30 * total <= tmp[k] <= 0.35 * total and 0.30 * total <= tot + tmp[k] <= 0.35 * total:
                    n_train.append(k + 1)
                    condition = True
                    tot = 0
                else:
                    break
            else:
                if k == len(tmp) - 1:
                    break
                k += 1
                if tmp[k] < 0.30 * total:
                    tot += tmp[k]
                    n_val_m.append(k + 1)
                    if 0.30 * total <= tot <= 0.35 * total:
                        search = False
                        break
                elif 0.30 * total <= tmp[k] <= 0.35 * total and 0.30 * total <= tot + tmp[k] <= 0.35 * total:
                    n_val_m.append(k + 1)
                    search = False
                    break
                else:
                    break
        if search:
            i += 1

    for t in range(1, i + 1):
        if t not in n_train and t not in n_val_m:
            n_val_g.append(t)

    idx_train = []
    idx_val_m = []
    idx_val_g = []
    for n in n_train:
        idx_train.extend(np.where(clusters == n)[0])
    for m in n_val_m:
        idx_val_m.extend(np.where(clusters == m)[0])
    for mm in n_val_g:
        idx_val_g.extend(np.where(clusters == mm)[0])

    with open(params['idx_clustering_path'], "wb") as f:
        pickle.dump((clusters, idx_train, idx_val_m, idx_val_g), f)

    return clusters, idx_train, idx_val_m, idx_val_g


def find_num_clusters(sequences, params):
    max = math.ceil(math.sqrt(len(sequences))) 
    range_n_clusters = list(x for x in range(8, max))
    sse = []
    seed = params['seed']
    log_file_path = params['log_path']
    start_time = time.time()

    with open(log_file_path, 'a') as filehandle:
        filehandle.write('Start searching optimal number of clusters \n')

    print("Start searching optimal number of clusters \n")

    for num_clusters in range_n_clusters:
        km = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=seed)
        labels = km.fit_predict(sequences)
        inertia = km.inertia_
        sse.append(inertia)

        with open(log_file_path, 'a') as filehandle:
            filehandle.write("Per n_clusters = {0}, inertia coefficient equal to {1} \n".format(num_clusters, inertia))

        print("Per n_clusters = {0}, inertia coefficient equal to {1} \n".format(num_clusters, inertia))

    plt.style.use("fivethirtyeight")
    plt.plot(range(8, max), sse)
    plt.xticks(range(8, max))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    # plt.show()

    kl = KneeLocator(range(8, max), sse, curve="convex", direction="decreasing")

    cluster_count = kl.elbow

    if cluster_count == None:
        cluster_count = params['n_clusters']

    end_time = time.time() - start_time

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"End searching optimal number of clusters {end_time} \n")
        filehandle.write(f"Optimal number of clusters: {cluster_count} \n")

    print(f"End searching optimal number of clusters {end_time} \n")
    print(f"Optimal number of clusters: {cluster_count} \n")

    return cluster_count


def create_clusters(sequences, cluster_count, params):
    path_distribution = params['plot_cluster_distribution_path']
    path_clusters = params['plot_clusters_path']
    seed = params['seed']

    km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", random_state=seed)
    labels = km.fit_predict(sequences)

    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(sequences))))
    plot_count = math.ceil(math.sqrt(cluster_count))

    fig, axs = plt.subplots(math.ceil(cluster_count / plot_count), plot_count, figsize=(25, 25))
    fig.suptitle('Clusters')
    row_i = 0
    column_j = 0
    # For each label there is,
    # plots every series with that label
    mean = []
    for label in set(labels):
        cluster = []
        # print(label)
        for i in range(len(labels)):
            if (labels[i] == label):
                # axs[row_i, column_j].plot(data[i],c="gray",alpha=0.4)
                cluster.append(sequences[i])
        if len(cluster) > 0:
            tmp_mean = np.average(np.vstack(cluster), axis=0)
            # print(tmp_mean)
            mean.append(tmp_mean)
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(label))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig(path_clusters)
    # plt.show()

    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n, cluster_c)
    plt.savefig(path_distribution)
    # plt.show()

    return labels, cluster_c, mean


def final_clusters(sequences, params):
    path_dendograms = params["path_dendograms"]
    cluster_count = find_num_clusters(sequences, params)

    labels, cluster_c, mean = create_clusters(sequences, cluster_count, params)

    matrix = np.zeros((cluster_count, cluster_count))
    for i in range(0, cluster_count):
        for j in range(0, cluster_count):
            matrix[i][j] = wasserstein_distance(mean[i], mean[j])

    dist_matrix = squareform(matrix)

    w_linkage = linkage(dist_matrix, method="complete")
    plt.figure(figsize=(15, 5))
    plt.title("Dendogram of Hierarchical Clustering")
    dend = dendrogram(w_linkage)
    plt.savefig(path_dendograms)
    # plt.show()

    clusters = fcluster(w_linkage, 2, criterion='maxclust')

    return labels, clusters, cluster_c

def create_list_index(clusters, cluster_c, params):
    log_file_path = params['log_path']
    tot_train = 0
    tot_val = 0
    idx_train = np.where(clusters == 1)
    idx_val = np.where(clusters == 2)

    for i in range(len(cluster_c)):
        if i in idx_train[0]:
            tot_train += cluster_c[i]
        else:
            tot_val += cluster_c[i]
    total = tot_train + tot_val

    if tot_train < 0.3 * total:
        tmp = idx_train
        idx_train = idx_val
        idx_val = tmp

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"Number of elements in training set M: {tot_train} \n")
        filehandle.write(f"Number of elements in validation set M: {tot_val} \n")
        filehandle.write(f"Clusters in training set: {idx_train} \n")
        filehandle.write(f"Clusters in validation set: {idx_val} \n")

    print("Number of elements in training set M:", tot_train, "\n")
    print("Number of elements in validation set M:", total - tot_train, "\n")

    print("Clusters in training set", idx_train)
    print("Clusters in validation set", idx_val)

    return idx_train, idx_val, tot_train, total


def create_list(cluster_c, dend, params):
    split_train = params['perc_split_m']
    log_file_path = params['log_path']

    total = 0
    for i in range(len(cluster_c)):
        total += cluster_c[i]

    num_el_train = split_train * total

    idx_train = []
    idx_val = []
    tot_train = 0

    for el in reversed(dend['ivl']):
        if tot_train < num_el_train:
            tot_train += cluster_c[int(el)]
            idx_train.append(int(el))
            if (tot_train / total - split_train) > 0.05:
                tot_train -= cluster_c[int(el)]
                idx_train.pop()
                idx_val.append(int(el))
        else:
            idx_val.append(int(el))

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"Number of elements in training set M: {tot_train} \n")
        filehandle.write(f"Number of elements in validation set M: {total-tot_train} \n")
        filehandle.write(f"Clusters in training set: {idx_train} \n")
        filehandle.write(f"Clusters in validation set: {idx_val} \n")

    print("Number of elements in training set M:", tot_train, "\n")
    print("Number of elements in validation set M:", total - tot_train, "\n")

    print("Clusters in training set", idx_train)
    print("Clusters in validation set", idx_val)

    return idx_train, idx_val


def generate_cluster_with_em(data, seed):
    X = data.copy()
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    em = GaussianMixture(n_components=2, random_state=seed)
    labels = em.fit_predict(X)

    return labels



