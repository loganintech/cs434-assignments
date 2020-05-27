import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def load_data(rootdir='./data/'):
    print('load data \n')
    c = './data/x_train.txt'
    c2 = './data/y_train.txt'
    #x_train = np.loadtxt(rootdir + 'x_train.txt', dtype=str).astype(float)
    x_train = np.loadtxt(c, dtype=str).astype(float)
    #y_train = np.loadtxt(rootdir + 'y_train.txt', dtype=str).astype(int)
    y_train = np.loadtxt(c2, dtype=str).astype(int)
    #x_test = np.loadtxt(rootdir + 'x_test.txt', dtype=str).astype(float)
    #y_test = np.loadtxt(rootdir + 'y_test.txt', dtype=str).astype(int)

    print('x_train: [%d, %d], y_train:[%d,]' % (x_train.shape[0], x_train.shape[1], y_train.shape[0]))

    return x_train, y_train#, x_test, y_test
