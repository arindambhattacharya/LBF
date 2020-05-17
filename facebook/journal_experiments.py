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

sns.set(context="talk")


def ca1(data):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    model = SGDClassifier(loss="log")
    model.fit(X_init, Y_init)
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 1

    my_bc = bc.BloomClassifier(model)
    start = time.time()
    my_bc.initialize(X_init, Y_init, m=model_fp, p=0.001)

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
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 1
    my_bc = bc.BloomClassifier(model)

    start = time.time()
    my_bc.initialize(X_init, Y_init, m=model_fp, p=0.001)

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
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 10
    my_dc = dc.dpbf_logistic(model)

    start = time.time()
    print(model_fp)
    my_dc.initialize(X_init, Y_init, n=model_fp, p=0.001)

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
            if y:
                my_dc.insert(x)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_dc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_dc.get_size())

    return (insert_fps, insert_times, insert_mems)


def base(data):
    X_init, Y_init, X_inserts, Y_inserts = data
    start = time.time()

    sys.path.append("bloom_classifier")
    from bloom_classifier import bloom_filter

    my_bf = bloom_filter.BloomFilter(n=len(X_init), p=0.001)
    start = time.time()
    for x, y in zip(X_init, Y_init):
        if y:
            my_bf.insert(x)
    init_time = (time.time() - start) / len(X_init)
    init_fp = sum([my_bf.check(x) for x, y in zip(X_init, Y_init) if not y])
    init_mem = my_bf.m

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
            if y:
                my_bf.insert(x)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(
            sum([my_bf.check(x) for x, y in zip(entire_X, entire_Y) if not y])
        )
        insert_mems.append(my_bf.m)

    return (insert_fps, insert_times, insert_mems)


if __name__ == "__main__":
    ds = pd.read_pickle("facebook_checkin.pkl")
    X = ds.drop(["place_id"], axis=1)
    X = (X - X.mean()) / X.std()
    X = X.to_numpy()
    Y = ds["place_id"].to_numpy()
    N = len(X) // 2

    ca1_fps, ca1_times, ca1_mems = [], [], []
    ca2_fps, ca2_times, ca2_mems = [], [], []
    ia_fps, ia_times, ia_mems = [], [], []
    base_fps, base_times, base_mems = [], [], []

    for i in range(6):
        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        X_init = X[:N]
        Y_init = Y[:N]
        X_inserts = np.array_split(X[N:], 10)
        Y_inserts = np.array_split(Y[N:], 10)

        data = (X_init, Y_init, X_inserts, Y_inserts)

        print("Running CA1")
        fps, times, mems = ca1(data)
        if i:
            ca1_fps.append(fps)
            ca1_times.append(times)
            ca1_mems.append(mems)

        print("Running CA2")
        fps, times, mems = ca2(data)
        if i:
            ca2_fps.append(fps)
            ca2_times.append(times)
            ca2_mems.append(mems)

        print("Running IA")
        fps, times, mems = ia(data)
        if i:
            ia_fps.append(fps)
            ia_times.append(times)
            ia_mems.append(mems)

        print("Running Base")
        fps, times, mems = ia(data)
        if i:
            base_fps.append(fps)
            base_times.append(times)
            base_mems.append(mems)

    ca1_fps = np.array(ca1_fps)
    ca1_times = np.array(ca1_times)
    ca1_mems = np.array(ca1_mems)
    ca2_fps = np.array(ca2_fps)
    ca2_times = np.array(ca2_times)
    ca2_mems = np.array(ca2_mems)
    ia_fps = np.array(ia_fps)
    ia_times = np.array(ia_times)
    ia_mems = np.array(ia_mems)
    base_fps = np.array(base_fps)
    base_times = np.array(base_times)
    base_mems = np.array(base_mems)

    plt.figure()
    plt.plot(ca1_fps.mean(axis=0), label="CA 1")
    plt.plot(ca2_fps.mean(axis=0), label="CA 2")
    plt.plot(ia_fps.mean(axis=0), label="IA")
    plt.plot(base_fps.mean(axis=0), label="Base")
    plt.title("FPS")
    plt.legend()
    plt.savefig("./plots/fpr.png")

    plt.figure()
    plt.plot(ca1_times.mean(axis=0), label="CA 1")
    plt.plot(ca2_times.mean(axis=0), label="CA 2")
    plt.plot(ia_times.mean(axis=0), label="IA")
    plt.plot(base_times.mean(axis=0), label="Base")
    plt.title("Time")
    plt.legend()
    plt.savefig("./plots/time.png")

    plt.figure()
    plt.plot(ca1_mems.mean(axis=0), label="CA 1")
    plt.plot(ca2_mems.mean(axis=0), label="CA 2")
    plt.plot(ia_mems.mean(axis=0), label="IA")
    plt.plot(base_mems.mean(axis=0), label="Base")
    plt.title("Memory")
    plt.legend()
    plt.savefig("./plots/mem.png")
