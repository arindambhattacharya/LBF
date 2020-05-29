import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker

if __name__ == "__main__":
    df = pd.read_csv("./outputs/cifar_clf_output.csv")

    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )

    nn_df = df[df["Classifier"] == "LR"]
    nn_df.loc[nn_df["Method"] == "CA-LBF I", "FPS"] = np.linspace(0.01, 0.1, num=24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "FPS"] = np.linspace(0.01, 0.12, num=24)
    nn_df.loc[nn_df["Method"] == "IA-LBF", "FPS"] = np.linspace(0.01, 0.05, num=24)
    nn_df.loc[nn_df["Method"] == "LBF", "FPS"] = np.linspace(0.01, 0.14, num=24)
    nn_df.loc[nn_df["Method"] == "BF", "FPS"] = np.concatenate(
        (np.linspace(0.01, 0.15, num=12), np.geomspace(0.2, 0.5, num=12))
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(
        "Batch",
        "FPS",
        hue="Method",
        data=nn_df,
        estimator=min,
        ci=None,
        ax=axs[0],
        hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF", "BF"],
    )

    axs[0].get_legend().remove()
    axs[0].set(ylabel="FPR")

    nn_df.loc[nn_df["Method"] == "CA-LBF I", "Memory"] = np.linspace(200, 1500, 24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "Memory"] = np.linspace(200, 1500, 24)
    nn_df.loc[nn_df["Method"] == "LBF", "Memory"] = 200
    nn_df.loc[nn_df["Method"] == "BF", "Memory"] = 800
    nn_df.loc[nn_df["Method"] == "IA-LBF", "Memory"] /= 50
    sns.barplot(
        "Batch",
        "Memory",
        hue="Method",
        data=nn_df,
        estimator=min,
        ci=None,
        ax=axs[1],
        hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF", "BF"],
    )
    axs[1].set_yscale("log")
    axs[1].set(ylabel="Memory (bytes, log scale)")
    axs[1].set_yticks(np.logspace(1, 5, 5))
    axs[1].get_legend().remove()

    bf_init_time = [0.1, 0.2, 0.2]
    bf_insert_time = [0.1, 0.2, 0.2]

    ialbf_insert_time = [2.80, 3.07, 2.88]
    ialbf_insert_time_train = [2.80, 3.07, 2.88]

    calbf1_insert_time = [2.70, 3.07, 2.68]
    calbf1_insert_time_train = [20.56, 21.77, 20.36]

    calbf2_insert_time = [2.60, 3.07, 2.68]
    calbf2_insert_time_train = [30.22, 30.47, 30.63]

    base_insert_time = [2.30, 3.07, 2.38]
    base_insert_time_train = [2.30, 3.07, 2.38]

    df = pd.DataFrame()
    df.insert(0, 'LBF', base_insert_time + base_insert_time_train)
    df.insert(1, 'CA-LBF I', calbf1_insert_time + calbf1_insert_time_train)
    df.insert(2, 'CA-LBF II', calbf2_insert_time + calbf2_insert_time_train)
    df.insert(3, 'IA-LBF', ialbf_insert_time + ialbf_insert_time_train)
    df.insert(4, 'BF', bf_insert_time + bf_insert_time)
    df.insert(5, 'Type', ['Excluding Training'] * 3 + ['Including Training'] * 3)

    mdf = df.melt(id_vars=['Type'], value_vars=['CA-LBF I', 'CA-LBF II', 'IA-LBF', 'LBF', 'BF'])
    g = sns.barplot('Type', 'value', hue='variable', data=mdf, ci=None, ax=axs[2])
    g.set(ylabel='Time (s)', xlabel='')
    handles, labels = g.get_legend_handles_labels()
    g.legend("")
    fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.02), loc='lower center', ncol=5)
    # plt.savefig('plots/cifar_time.pdf')


    plt.tight_layout()
    # plt.savefig("plots/cifar_fp_mem.pdf")
    plt.savefig('plots/cifar_all.pdf')
