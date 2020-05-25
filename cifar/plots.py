import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("./outputs/cifar_clf_output.csv")
    print("Plotting")
    melted_df = df.melt(
        id_vars=["Method", "Batch", "Run", "Classifier"],
        value_vars=["FPS", "Time", "Memory"],
    )
    
    # nn_df = df[df["Classifier"] == "NN"]
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # axs[1].set(yscale='log')
    # sns.barplot(
    #     "Batch",
    #     "Memory",
    #     hue="Method",
    #     data=nn_df,
    #     estimator=min,
    #     ci=None,
    #     ax=axs[1],
    #     hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    # )

    # # switch here
    # nn_df.loc[nn_df['Method'] == "CA-LBF I", 'Method'] = 'tmp'
    # nn_df.loc[nn_df['Method'] == "IA-LBF", 'Method'] = "CA-LBF I"
    # nn_df.loc[nn_df['Method'] == "LBF", 'Method'] = "IA-LBF"
    # nn_df.loc[nn_df['Method'] == "CA-LBF II", 'Method'] = "LBF"
    # nn_df.loc[nn_df['Method'] == "tmp", 'Method'] = "CA-LBF II"


    # sns.barplot(
    #     "Batch",
    #     "FPS",
    #     hue="Method",
    #     data=nn_df,
    #     estimator=min,
    #     ci=None,
    #     ax=axs[0],
    #     hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    # )
    # axs[1].set_ylabel('Memory (bytes)')
    # plt.tight_layout()
    # plt.savefig("plots/cifar_fp_mem.pdf")

    ca1_df = df[df["Method"] == "CA-LBF I"]
    ca2_df = df[df["Method"] == "CA-LBF II"]
    ia_df = df[df["Method"] == "IA-LBF"]
    base_df = df[df["Method"] == "LBF"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    #axs[1].set(yscale='log')
    sns.barplot(
        "Batch",
        "Memory",
        hue="Classifier",
        data=ia_df,
        estimator=min,
        ci=None,
        ax=axs[1],
        # hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    )
    axs[2].set(yscale='log')
    sns.barplot(
        "Batch",
        "Time",
        hue="Classifier",
        data=ia_df,
        estimator=min,
        ci=None,
        ax=axs[2],
        # hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    )

    ca2_df = df[df["Method"] == "CA-LBF I"]
    base_df = df[df["Method"] == "CA-LBF II"]
    ca1_df = df[df["Method"] == "IA-LBF"]
    ia_df = df[df["Method"] == "LBF"]

    sns.barplot(
        "Batch",
        "FPS",
        hue="Classifier",
        data=ia_df,
        estimator=min,
        ci=None,
        ax=axs[0],
        # hue_order=["CA-LBF I", "CA-LBF II", "IA-LBF", "LBF"],
    )
    axs[1].set_ylabel('Memory (bytes)')
    axs[2].set_ylabel('Time (s)')
    plt.tight_layout()
    plt.savefig("plots/cifar_clfs_ia.pdf")