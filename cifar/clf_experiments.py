import copy
import os
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as skds
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")  # shouldn't do this


def ca1(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    if clf == "SVM":
        model = SGDClassifier(
            loss="hinge", warm_start=True, class_weight={0: 9, 1: 1}, penalty="none"
        )
    elif clf == "NN":
        model = MLPClassifier((3176, 1024, 512, 256, 128, 64, 32), warm_start=True)
    elif clf == "LR":
        model = LogisticRegression(
            warm_start=True, penalty="none", class_weight={0: 9, 1: 1}
        )

    start = time.time()
    model.fit(X_init, Y_init)
    # model_fp = max(np.sum([model.predict(X_init[Y_init == 0])]), 1000)

    my_bc = bc.BloomClassifier(model)
    # my_bc.initialize(X_init, Y_init, n=int(model_fp), p=1e-4)
    my_bc.initialize(X_init, Y_init, n=1000, p=1e-4)

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
        model.fit(X_insert, Y_insert)
        my_bc.add_data(X_insert, Y_insert, model)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_bc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_bc.get_size())

    return (insert_fps, insert_times, insert_mems)


def ca2(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    if clf == "SVM":
        model = SGDClassifier(
            loss="hinge", warm_start=False, class_weight={0: 9, 1: 1}, penalty="none"
        )
    elif clf == "NN":
        model = MLPClassifier((3176, 1024, 512, 256, 128, 64, 32), warm_start=False)
    elif clf == "LR":
        model = LogisticRegression(
            warm_start=False, penalty="none", class_weight={0: 9, 1: 1}
        )

    start = time.time()
    model.fit(X_init, Y_init)
    # model_fp = max(np.sum([model.predict(X_init[Y_init == 0])]), 1000)
    my_bc = bc.BloomClassifier(model)

    # my_bc.initialize(X_init, Y_init, n=int(model_fp), p=1e-4)
    my_bc.initialize(X_init, Y_init, n=1000, p=1e-4)

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
        if clf == "SVM":
            model = SGDClassifier(
                loss="hinge",
                warm_start=False,
                class_weight={0: 9, 1: 1},
                penalty="none",
            )
        elif clf == "NN":
            model = MLPClassifier((3176, 1024, 512, 256, 128, 64, 32), warm_start=False)
        elif clf == "LR":
            model = LogisticRegression(
                warm_start=False, penalty="none", class_weight={0: 9, 1: 1}
            )
        model.fit(X_insert, Y_insert)
        my_bc.add_data(X_insert, Y_insert, model)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_bc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_bc.get_size())

    return (insert_fps, insert_times, insert_mems)


def ia(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import dpbf_classification as dc

    X_init, Y_init, X_inserts, Y_inserts = data

    if clf == "SVM":
        model = SGDClassifier(
            loss="hinge", warm_start=False, class_weight={0: 9, 1: 1}, penalty="none"
        )
    elif clf == "NN":
        model = MLPClassifier((3176, 1024, 512, 256, 128, 64, 32), warm_start=False)
    elif clf == "LR":
        model = LogisticRegression(
            warm_start=False, penalty="none", class_weight={0: 9, 1: 1}
        )

    start = time.time()
    model.fit(X_init, Y_init)
    # model_fp = max(np.sum([model.predict(X_init[Y_init == 0])]), 100)
    my_dc = dc.dpbf_logistic(model)

    # my_dc.initialize(X_init, Y_init, n=int(model_fp), p=1e-4)
    my_dc.initialize(X_init, Y_init, n=100, p=1e-2)

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
        print('Inserting new batch...')
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


def base(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    if clf == "SVM":
        model = SGDClassifier(
            loss="hinge", warm_start=False, class_weight={0: 9, 1: 1}, penalty="none"
        )
    elif clf == "NN":
        model = MLPClassifier((3176, 1024, 512, 256, 128, 64, 32), warm_start=False)
    elif clf == "LR":
        model = LogisticRegression(
            warm_start=False, penalty="none", class_weight={0: 9, 1: 1}
        )

    model.fit(X_init, Y_init)
    # model_fp = max(np.sum([model.predict(X_init[Y_init == 0])]), 1000)
    my_bc = bc.BloomClassifier(model)

    start = time.time()
    # my_bc.initialize(X_init, Y_init, n=int(model_fp), p=1e-4)
    my_bc.initialize(X_init, Y_init, n=1000, p=1e-4)

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
        my_bc.add_data(X_insert, Y_insert)
        insert_times.append((time.time() - start) / len(X_insert))
        insert_fps.append(my_bc.get_fpr(entire_X, entire_Y))
        insert_mems.append(my_bc.get_size())

    return (insert_fps, insert_times, insert_mems)


if __name__ == "__main__":
    X, Y = skds.fetch_openml("cifar_10", return_X_y=True)
    Y = np.array([int(y) for y in Y])
    X = X[Y < 2]
    Y = Y[Y < 2]
    N = len(X) // 2

    df = pd.DataFrame()

    shuffle_indices = np.arange(len(X))
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]
    X_init = X[:N]
    Y_init = Y[:N]
    X_inserts = np.array_split(X[N:], 10)
    Y_inserts = np.array_split(Y[N:], 10)

    for i in range(2):
        data = (X_init, Y_init, X_inserts, Y_inserts)
        clfs = ["LR", "SVM", "NN"]

        for clf in clfs:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Running {clf}")

            print("Running IA")
            fps, times, mems = ia(data, clf)
            if i:
                for j, (fp, t, mem) in enumerate(zip(fps, times, mems)):
                    df = df.append(
                        {
                            "Method": "IA-LBF",
                            "Run": i,
                            "Batch": j,
                            "FPS": fp,
                            "Time": t,
                            "Memory": mem,
                            "Classifier": clf,
                        },
                        ignore_index=True,
                    )

            print("Running CA1")
            fps, times, mems = ca1(data, clf)
            if i:
                for j, (fp, t, mem) in enumerate(zip(fps, times, mems)):
                    df = df.append(
                        {
                            "Method": "CA-LBF I",
                            "Run": i,
                            "Batch": j,
                            "FPS": fp,
                            "Time": t,
                            "Memory": mem,
                            "Classifier": clf,
                        },
                        ignore_index=True,
                    )

            print("Running CA2")
            fps, times, mems = ca2(data, clf)
            if i:
                for j, (fp, t, mem) in enumerate(zip(fps, times, mems)):
                    df = df.append(
                        {
                            "Method": "CA-LBF II",
                            "Run": i,
                            "Batch": j,
                            "FPS": fp,
                            "Time": t,
                            "Memory": mem,
                            "Classifier": clf,
                        },
                        ignore_index=True,
                    )

            print("Running Base")
            fps, times, mems = base(data, clf)
            if i:
                for j, (fp, t, mem) in enumerate(zip(fps, times, mems)):
                    df = df.append(
                        {
                            "Method": "LBF",
                            "Run": i,
                            "Batch": j,
                            "FPS": fp,
                            "Time": t,
                            "Memory": mem,
                            "Classifier": clf,
                        },
                        ignore_index=True,
                    )
    df.to_csv("./outputs/cifar_clf_output.csv")

    print("Plotting")
    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )
    g = sns.relplot(
        "Batch",
        "value",
        col="variable",
        row="Method",
        hue="Classifier",
        kind="line",
        data=melted_df,
        markers=True,
        facet_kws={"sharey": False},
        ci=None,
    )
    g.savefig("plots/cifar_clfs.png")
