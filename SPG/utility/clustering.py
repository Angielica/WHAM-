import math
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
import time
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
from kneed import KneeLocator


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
        # cluster_c = [len(labels[labels == i]) for i in range(num_clusters)]
        # cluster_labels = km.labels_
        inertia = km.inertia_
        sse.append(inertia)

        with open(log_file_path, 'a') as filehandle:
            filehandle.write("Per n_clusters = {0}, il coefficiente di inertia è pari a {1} \n".format(num_clusters, inertia))

        print("Per n_clusters = {0}, il coefficiente di inertia è pari a {1} \n".format(num_clusters, inertia))

    plt.style.use("fivethirtyeight")
    plt.plot(range(8, max), sse)
    plt.xticks(range(8, max))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(8, max), sse, curve="convex", direction="decreasing")

    cluster_count = kl.elbow

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
    plt.show()

    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n, cluster_c)
    plt.savefig(path_distribution)
    plt.show()

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
    plt.show()

    return labels, cluster_c, dend


def create_list_index(cluster_c, dend, params):
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



