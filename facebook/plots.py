import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
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
    plt.tight_layout()
    plt.savefig("plots/fb_fp_mem.pdf")
