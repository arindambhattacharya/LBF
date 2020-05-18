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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")  # shouldn't do this


def ca1(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    try:
        model = clf(n_estimators=10, max_depth=2, warm_start=True)
    except Exception:
        model = clf(warm_start=True)

    model.fit(X_init, Y_init)
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 1

    my_bc = bc.BloomClassifier(model)
    start = time.time()
    my_bc.initialize(X_init, Y_init, n=model_fp, p=1e-4)

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

    try:
        model = clf(n_estimators=10, max_depth=2, warm_start=False)
    except Exception:
        model = clf(warm_start=False)
    model.fit(X_init, Y_init)
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 1
    my_bc = bc.BloomClassifier(model)

    start = time.time()
    my_bc.initialize(X_init, Y_init, n=model_fp, p=1e-4)

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
        try:
            model = clf(n_estimators=10, max_depth=2, warm_start=False)
        except Exception:
            model = clf(warm_start=False)
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

    try:
        model = clf(n_estimators=10, max_depth=2, warm_start=False)
    except Exception:
        model = clf(warm_start=False)
    model.fit(X_init, Y_init)
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])])
    my_dc = dc.dpbf_logistic(model)

    start = time.time()
    my_dc.initialize(X_init, Y_init, n=1024, p=1e-4)

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


def base(data, clf):
    sys.path.append("bloom_classifier")
    sys.path.append("dpbf_classifier")

    import bloom_classifier as bc

    X_init, Y_init, X_inserts, Y_inserts = data

    try:
        model = clf(n_estimators=10, max_depth=2, warm_start=False)
    except Exception:
        model = clf(warm_start=False)
    model.fit(X_init, Y_init)
    model_fp = len([1 for x in X_init[Y_init == 0] if model.predict([x])]) + 1
    my_bc = bc.BloomClassifier(model)

    start = time.time()
    my_bc.initialize(X_init, Y_init, n=model_fp, p=1e-4)

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
    ds = pd.read_pickle("facebook_checkin.pkl")
    X = ds.drop(["place_id"], axis=1)
    X = X.to_numpy()
    Y = ds["place_id"].to_numpy()
    N = len(X) // 2

    df = pd.DataFrame()

    for i in range(10):
        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        X_init = X[:N]
        Y_init = Y[:N]
        X_inserts = np.array_split(X[N:], 10)
        Y_inserts = np.array_split(Y[N:], 10)

        data = (X_init, Y_init, X_inserts, Y_inserts)
        clfs = dict(
            zip(
                ["LR", "NN", "RF"],
                [LogisticRegression, MLPClassifier, RandomForestClassifier],
            )
        )
        for name, clf in clfs.items():
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Running {name}")
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
                            "Classifier": name,
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
                            "Classifier": name,
                        },
                        ignore_index=True,
                    )

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
                            "Classifier": name,
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
                            "Classifier": name,
                        },
                        ignore_index=True,
                    )

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
        ci=95,
    )
    g.savefig("plots/fb_clfs.png")
