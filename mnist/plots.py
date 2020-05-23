import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("./outputs/mnist_clf_output.csv")
    print("Plotting")
    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )
    
    nn_df = df[df["Classifier"] == "NN"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(
        "Batch",
        "FPS",
        hue="Method",
        data=nn_df,
        estimator=min,
        ci=None,
        ax=axs[0],
        markers=True,
        hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    )
    sns.lineplot(
        "Batch",
        "Memory",
        hue="Method",
        data=nn_df,
        estimator=min,
        ci=None,
        ax=axs[1],
        markers=True,
        hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    )
    plt.tight_layout()
    plt.savefig("plots/mnist_fp_mem.pdf")


