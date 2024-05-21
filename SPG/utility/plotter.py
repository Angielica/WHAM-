import matplotlib.pyplot as plt
import math
import numpy as np


def plot_clusters(seqs, labels, params):
    n_clusters = params['n_clusters']

    plot_count = math.ceil(math.sqrt(n_clusters))

    fig, axs = plt.subplots(math.ceil(n_clusters / plot_count), plot_count, figsize=(25, 25))
    fig.suptitle('Clusters')

    row_i = 0
    column_j = 0

    # For each label there is,
    # plots every series with that label

    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if (labels[i] == label):
                # axs[row_i, column_j].plot(data[i],c="gray",alpha=0.4)
                cluster.append(seqs[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cl " + str(label))

        column_j += 1

        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0

    plt.savefig(params['plot_clusters_path'], bbox_inches='tight')
    plt.show()


def plot_cluster_distribution(labels, params):
    n_clusters = params['n_clusters']
    cluster_c = [len(labels[labels == i]) for i in range(n_clusters)]
    cluster_n = ["Cluster " + str(i) for i in range(n_clusters)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n, cluster_c)
    plt.savefig(params['plot_cluster_distribution_path'], bbox_inches='tight')
    plt.show()


def plot_loss(loss_values, val_loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()


def plot_loss_log(loss_values, val_loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()

def plot_loss_training(loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()

def plot_loss_training_log(loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.yscale('log')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()