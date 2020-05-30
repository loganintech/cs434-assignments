import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

sns.set()

import argparse

from utils import load_data
from decompose import PCA
from clustering import KMeans


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--pca', default=0, type=int,
                        help='set to 1 if we desire running pca, otherwise 0')
    parser.add_argument('--kmeans', default=1, type=int,
                        help='set to 1 if we desire running kmeans, otherwise 0')

    parser.add_argument('--pca_retain_ratio', default=.9, type=float)
    parser.add_argument('--kmeans_max_k', default=10, type=int) #default was originally 15
    parser.add_argument('--kmeans_max_iter', default=20, type=int)
    parser.add_argument('--root_dir', default='./data/', type=str)
    args = parser.parse_args()

    return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    plots_per_fig = 2

    ks_sses_keys = list(range(0, len(y_vs_x)))
    js = list(range(0, len(ks_sses_keys), plots_per_fig))

    for j in js:
        pp = ks_sses_keys[j:j + plots_per_fig]
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(len(pp), 1, figure=fig)
        i = 0
        for k in pp:
            ax = fig.add_subplot(gs[i, :])
            ax.set_ylabel('%s (k=%d)' % (y_label, k))
            ax.set_xlabel(x_label)
            ax.plot(range(1, len(y_vs_x[k]) + 1), [x for x in y_vs_x[k]], linewidth=2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1

        fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

    print('Saved at : %s' % fld)


def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(fld, 'plot.png'))

    print('Saved at : %s' % fld)


def visualize(x_train, y_train):
    #pass
    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################        
    c1 = []
    for i in range(y_train.shape[0]):
        if (y_train[i] == 0):
            c1.append(0)
        elif (y_train[i] == 1):
            c1.append(1)
        elif (y_train[i] == 2):
            c1.append(2)
        elif (y_train[i] == 3):
            c1.append(3)
        elif (y_train[i] == 4):
            c1.append(4)
        elif (y_train[i] == 5):
            c1.append(5)
        elif (y_train[i] == 6):
            c1.append(6)    
    
    plt.scatter(x_train[:,0], x_train[:,1], c=c1, marker="o", picker=True)
    plt.show()


def apply_kmeans(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################
    #run it five times, then compute the average of sses_vs_iter, then plot.
    for k in range(1, kmeans_max_k):
        t = []
        for i in range(5):
            kmeans = KMeans(k, kmeans_max_iter)
            sse_vs_iter = kmeans.fit(x_train)
            t.append(sse_vs_iter)
            #train_sses_vs_iter.append(sse_vs_iter)
            train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
            train_sses_vs_k.append(min(sse_vs_iter))
        avg_list = np.average(t, axis=0)
        train_sses_vs_iter.append(avg_list)

    x_axis = list(range(20))
    #plt.plot(train_sses_vs_iter[6])
    #plt.savefig("AVERAGE_sse_vs_iter_k=6")
    #plt.show()

    for i in range(len(train_sses_vs_iter))):
        plt.plot(train_sses_vs_iter[i])
        plt.show()
        

    plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
                     save_path='plot_sse_vs_k_subplots_%d'%do_pca)
    plot_y_vs_x(train_sses_vs_k, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d'%do_pca)
    plot_y_vs_x(train_purities_vs_k, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d'%do_pca)



if __name__ == '__main__':
    args = load_args()
    #x_train, y_train, x_test, y_test = load_data(args.root_dir)
    x_train, y_train = load_data(args.root_dir)

    if args.pca == 1:
        pca = PCA(args.pca_retain_ratio)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
       # x_test = pca.transform(x_test)
        visualize(x_train, y_train)

    if args.kmeans == 1:
        apply_kmeans(args.pca, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)

    print('Done')
