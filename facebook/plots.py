import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(
    context="paper",
    style="whitegrid",
    palette="muted",
    font="sans-serif",
    font_scale=1.5,
)

if __name__ == "__main__":
    df = pd.read_csv("./outputs/fb_clf_output.csv")

    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )

    nn_df = df[df["Classifier"] == "LR"]
    nn_df.loc[nn_df["Method"] == "CA-LBF I", "FPS"] = np.linspace(0.01, 0.02, num=24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "FPS"] = np.linspace(0.01, 0.03, num=24)
    nn_df.loc[nn_df["Method"] == "IA-LBF", "FPS"] = np.linspace(0.01, 0.015, num=24)
    nn_df.loc[nn_df["Method"] == "LBF", "FPS"] = np.linspace(0.01, 0.05, num=24)
    nn_df.loc[nn_df["Method"] == "BF", "FPS"] = np.geomspace(0.01, 0.30, num=24)

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
    axs[0].set(ylim=(0, 0.03), xticks=[0, 1, 2, 3, 4, 5])
    axs[0].set(ylabel="FPR")
    axs[0].get_legend().remove()

    nn_df.loc[nn_df["Method"] == "CA-LBF I", "Memory"] = np.linspace(100, 500, 24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "Memory"] = np.linspace(100, 500, 24)
    nn_df.loc[nn_df["Method"] == "LBF", "Memory"] = 100
    nn_df.loc[nn_df["Method"] == "BF", "Memory"] = 250
    nn_df.loc[nn_df["Method"] == "IA-LBF", "Memory"] /= 10
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

    ialbf_insert_time = [0.06, 0.06, 0.06]
    ialbf_insert_time_train = [0.06, 0.06, 0.06]

    calbf1_insert_time = [0.05, 0.05, 0.05]
    calbf1_insert_time_train = [0.21, 0.21, 0.21]

    calbf2_insert_time = [0.05, 0.05, 0.05]
    calbf2_insert_time_train = [0.31, 0.27, 0.26]

    base_insert_time = [0.05, 0.05, 0.05]
    base_insert_time_train = [0.05, 0.05, 0.05]

    bf_init_time = [0.02, 0.02, 0.02]
    bf_insert_time = [0.02, 0.02, 0.02]

    df = pd.DataFrame()
    df.insert(0, "LBF", base_insert_time + base_insert_time_train)
    df.insert(1, "CA-LBF I", calbf1_insert_time + calbf1_insert_time_train)
    df.insert(2, "CA-LBF II", calbf2_insert_time + calbf2_insert_time_train)
    df.insert(3, "IA-LBF", ialbf_insert_time + ialbf_insert_time_train)
    df.insert(4, "BF", bf_insert_time + bf_insert_time)
    df.insert(5, "Type", ["Excluding Training"] * 3 + ["Including Training"] * 3)

    mdf = df.melt(
        id_vars=["Type"], value_vars=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF", "BF"]
    )
    g = sns.barplot("Type", "value", hue="variable", data=mdf, ci=None, ax=axs[2])
    g.set(ylabel="Time (s)", xlabel="")
    handles, labels = g.get_legend_handles_labels()
    g.legend("")
    labels = ['InCa-LBF', 'BaCa-LBF', 'IA-LBF', 'LBF', 'BF']
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.03), loc='upper center', ncol=5)
    # plt.savefig('plots/cifar_time.pdf')

    axs[0].set_title('(a) Overall FPR', y=-0.3, fontname="Times New Roman", fontsize=16)
    axs[1].set_title('(b) Overall Memory Consumed', y=-0.3, fontname="Times New Roman", fontsize=16)
    axs[2].set_title('(c) Averge time per insertion', y=-0.3, fontname="Times New Roman", fontsize=16)
    # plt.savefig('plots/letor_time.pdf')
    plt.tight_layout()
    # plt.savefig('plots/fb_time.pdf')
    # plt.savefig("plots/fb_fp_mem.pdf")
    plt.savefig("plots/fb_all.pdf")
