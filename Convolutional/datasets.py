from __future__ import print_function

import os
import os.path
import pickle

import numpy as np
import tensorflow as tf


def run_all():
    cpath = os.getcwd()
    try:
        os.chdir(os.getcwd() + '/datasets')
    except:
        os.mkdir('datasets')
        os.chdir(os.getcwd() + '/datasets')
    print('\nDownloading the Cifar100 dataset')
    data = tf.keras.datasets.cifar100.load_data(
        label_mode='fine'
    )
    data_train, data_test = data
    X_train, Y_train = data_train
    X_test, Y_test = data_test
    indexes = np.arange(100)
    # np.random.shuffle(indexes)
    all_sets = []
    for i in range(20):
        labels = indexes[i * 5:(i + 1) * 5]
        train_index = []
        for l in labels:
            train_index += list(np.where(Y_train == l)[0])

        test_index = []
        for l in labels:
            test_index += list(np.where(Y_test == l)[0])

        bxtrain, bytrain = X_train[train_index], Y_train[train_index]
        bxtest, bytest = X_test[test_index], Y_test[test_index]

        cset = [bxtrain, bytrain, bxtest, bytest]
        all_sets.append(cset)

    pickle.dump(all_sets, open('split_cifar_100.pkl', 'wb'))

    print('\nDownloading the Cifar10 dataset')
    data = tf.keras.datasets.cifar10.load_data()
    data_train, data_test = data
    X_train, Y_train = data_train
    X_test, Y_test = data_test
    indexes = np.arange(10)
    # np.random.shuffle(indexes)
    all_sets = []
    set_labels = np.arange(10)  # [2,0,1,5,3,7,6,4,8,9]
    for i in range(5):
        labels = [set_labels[2 * i], set_labels[2 * (i) + 1]]
        train_index = []
        for l in labels:
            train_index += list(np.where(Y_train == l)[0])

        test_index = []
        for l in labels:
            test_index += list(np.where(Y_test == l)[0])

        bxtrain, bytrain = X_train[train_index], Y_train[train_index]
        bxtest, bytest = X_test[test_index], Y_test[test_index]

        cset = [bxtrain, bytrain, bxtest, bytest]
        all_sets.append(cset)

    pickle.dump(all_sets, open('split_cifar_10.pkl', 'wb'))

    os.chdir(cpath)
