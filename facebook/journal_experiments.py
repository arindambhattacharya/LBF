import copy
import os
import pickle
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")  # shouldn't do this


def ca1(data):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    model = SGDClassifier(loss="log")
    model.fit(X_init, Y_init)

    my_bc = bc.BloomClassifier(model)

    outfile = open("outputs/fb_calbf1_run1.txt", "w")
    start = time.time()
    my_bc.initialize(X_init, Y_init, m=50)

    init_time = (time.time() - start) / len(X_init)
    init_fp = my_bc.get_fpr(X_init, Y_init)
    init_mem = my_bc.get_size()

    insert_fps = []
    insert_mems = []
    insert_times = []

    insert_fps.append(init_fp)
    insert_mems.append(init_mem)
    insert_times.append(init_time)

    entire_X = X_init
    entire_Y = Y_init

    for X_insert, Y_insert in zip(X_inserts, Y_inserts):
        entire_X = np.concatenate((entire_X, X_insert))
        entire_Y = np.concatenate((entire_Y, Y_insert))

        start = time.time()
        model.partial_fit(X_insert, Y_insert)
        my_bc.add_data(X_insert, Y_insert, model)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_bc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_bc.get_size())

    return (insert_fps, insert_times, insert_mems)


def ca2(data):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    model = SGDClassifier(loss="log")
    model.fit(X_init, Y_init)

    my_bc = bc.BloomClassifier(model)

    outfile = open("outputs/fb_calbf1_run1.txt", "w")
    start = time.time()
    my_bc.initialize(X_init, Y_init, m=50)

    init_time = (time.time() - start) / len(X_init)
    init_fp = my_bc.get_fpr(X_init, Y_init)
    init_mem = my_bc.get_size()

    insert_fps = []
    insert_mems = []
    insert_times = []

    insert_fps.append(init_fp)
    insert_mems.append(init_mem)
    insert_times.append(init_time)

    entire_X = X_init
    entire_Y = Y_init

    for X_insert, Y_insert in zip(X_inserts, Y_inserts):
        entire_X = np.concatenate((entire_X, X_insert))
        entire_Y = np.concatenate((entire_Y, Y_insert))

        start = time.time()
        model = SGDClassifier(loss="log")
        model.fit(X_insert, Y_insert)
        my_bc.add_data(X_insert, Y_insert, model)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_bc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_bc.get_size())

    return (insert_fps, insert_times, insert_mems)


def ia(data):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import dpbf_classification as dc

    X_init, Y_init, X_inserts, Y_inserts = data

    model = SGDClassifier(loss="log")
    model.fit(X_init, Y_init)

    my_dc = dc.dpbf_logistic(model)

    outfile = open("outputs/fb_calbf1_run1.txt", "w")
    start = time.time()
    my_dc.initialize(X_init, Y_init, m=50)

    init_time = (time.time() - start) / len(X_init)
    init_fp = my_dc.get_fpr(X_init, Y_init)
    init_mem = my_dc.get_size()

    insert_fps = []
    insert_mems = []
    insert_times = []

    insert_fps.append(init_fp)
    insert_mems.append(init_mem)
    insert_times.append(init_time)

    entire_X = X_init
    entire_Y = Y_init

    for X_insert, Y_insert in zip(X_inserts, Y_inserts):
        entire_X = np.concatenate((entire_X, X_insert))
        entire_Y = np.concatenate((entire_Y, Y_insert))

        start = time.time()
        for x, y in zip(X_insert, Y_insert):
            if Y_insert:
                my_dc.insert(x)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_dc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_dc.get_size())

    return (insert_fps, insert_times, insert_mems)


if __name__ == "__main__":
    ds = pd.read_pickle("facebook_checkin.pkl")
    X = ds.drop(["place_id"], axis=1)
    X = (X - X.mean()) / X.std()
    X = X.to_numpy()
    Y = ds["place_id"].to_numpy()
    N = len(X) // 2
    shuffle_indices = np.arange(len(X))
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]
    X_init = X[:N]
    Y_init = Y[:N]
    X_inserts = np.array_split(X[N:], 10)
    Y_inserts = np.array_split(Y[N:], 10)

    data = (X_init, Y_init, X_inserts, Y_inserts)
    ca1_fps, ca1_times, ca1_mems = ca1(data)
    ca2_fps, ca2_times, ca2_mems = ca2(data)
    ia_fps, ia_times, ia_mems = ia(data)

    plt.plot(ca1_fps, label="CA 1")
    plt.plot(ca2_fps, label="CA 2")
    plt.plot(ia_fps, label="IA")
    plt.title("FPS")
    plt.show()

    plt.plot(ca1_times, label="CA 1")
    plt.plot(ca2_times, label="CA 2")
    plt.plot(ia_times, label="IA")
    plt.title("Time")
    plt.show()

    plt.plot(ca1_mems, label="CA 1")
    plt.plot(ca2_mems, label="CA 2")
    plt.plot(ia_mems, label="IA")
    plt.title("Memory")
    plt.show()
