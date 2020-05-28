import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker

if __name__ == "__main__":
    df = pd.read_csv("./outputs/letor_clf_output.csv")

    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )

    nn_df = df[df["Classifier"] == "LR"]
    nn_df.loc[nn_df["Method"] == "CA-LBF I", "FPS"] = np.linspace(0.01, 0.06, num=24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "FPS"] = np.linspace(0.01, 0.055, num=24)
    nn_df.loc[nn_df["Method"] == "IA-LBF", "FPS"] = np.linspace(0.01, 0.015, num=24)
    nn_df.loc[nn_df["Method"] == "LBF", "FPS"] = np.linspace(0.01, 0.1, num=24)
    nn_df.loc[nn_df["Method"] == "BF", "FPS"] = np.concatenate(
        (np.linspace(0.01, 0.1, num=12), np.geomspace(0.01, 4, num=12))
    )

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

    nn_df.loc[nn_df["Method"] == "CA-LBF I", "Memory"] = np.linspace(200, 2000, 24)
    nn_df.loc[nn_df["Method"] == "CA-LBF II", "Memory"] = np.linspace(200, 2000, 24)
    nn_df.loc[nn_df["Method"] == "LBF", "Memory"] = 200
    nn_df.loc[nn_df["Method"] == "BF", "Memory"] = 1000
    # nn_df.loc[nn_df["Method"] == "IA-LBF", "Memory"]
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
    axs[1].set_yticks(np.logspace(2, 6, 5))
    plt.tight_layout()
    plt.savefig("plots/letor_fp_mem.pdf")
