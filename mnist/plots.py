import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv("./outputs/mnist_clf_output.csv")
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
    g.savefig("plots/mnist_clfs.pdf")
