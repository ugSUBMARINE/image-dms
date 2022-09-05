import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("seaborn-whitegrid")


def plot_reruns(protein_name: str, result_path: list[str] | None = None) -> None:
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance
    compared to Gelman et al. 'Neural networks to learn protein sequence–function
    relationships from deep mutational scanning data'
    :parameter
        - protein_name:
          how the protein is named in the result and test files
        - result_path:
          list of file paths to the different training runs
    :return
        None"""
    if result_path is None:
        result_path = "result_files/rr3_results/{}_results.csv".format(
            protein_name.lower()
        )

    # get the sequence convolution data to compare the runs to
    g_data = pd.read_csv(
        "nononsense/{}_test_formatted.txt".format(protein_name), delimiter=","
    )
    g_mses = g_data["mse"].values
    g_pearsons = g_data["pearsonr"].values
    g_spearmans = g_data["spearmanr"].values

    # split them into their respective runs
    g_split_mses = np.median(np.split(g_mses, 3), axis=0)
    g_split_pearsons = np.median(np.split(g_pearsons, 3), axis=0)
    g_split_spearmans = np.median(np.split(g_spearmans, 3), axis=0)

    # read data and convert it to ndarrays
    data = pd.read_csv(result_path, delimiter=",")
    architecture = np.unique(data["architecture"])[0]
    mse = np.asarray(data["mse"])
    pearson = np.asarray(data["pearson_r"])
    spearman = np.asarray(data["spearman_r"])

    # split for not augmented and augmented
    ana_mse = np.asarray(np.split(mse, 2))
    ana_pearson = np.asarray(np.split(pearson, 2))
    ana_spearman = np.asarray(np.split(spearman, 2))

    fig, axs = plt.subplots(2, 3)
    # different training set sizes
    set_sizes = [50, 100, 250, 500, 1000, 2000, 6000]
    # iterates over not augmented and augmented data separately
    for c, (m, p, s) in enumerate(zip(ana_mse, ana_pearson, ana_spearman)):
        # run split
        rs_mse = np.split(m, 3)
        rs_pearson = np.split(p, 3)
        rs_spearman = np.split(s, 3)

        # lists for calculating the medians
        rs_mse_m = []
        rs_pearson_m = []
        rs_spearman_m = []
        for i in range(len(rs_mse)):
            rs_mse_m += [rs_mse[i]]
            rs_pearson_m += [rs_pearson[i]]
            rs_spearman_m += [rs_spearman[i]]

        # all medians
        mse_medians = np.median(np.asarray(rs_mse_m), axis=0)
        pearson_medians = np.median(np.asarray(rs_pearson_m), axis=0)
        spearman_medians = np.median(np.asarray(rs_spearman_m), axis=0)

        # split medians in their different settings
        s_mse_medians = np.asarray(np.split(mse_medians, 3))
        s_pearson_medians = np.asarray(np.split(pearson_medians, 3))
        s_spearman_medians = np.asarray(np.split(spearman_medians, 3))

        # relative performance against sequence convolution
        r_mse = 200 - (s_mse_medians / g_split_mses) * 100
        r_pearson = (s_pearson_medians / g_split_pearsons) * 100
        r_spearman = (s_spearman_medians / g_split_spearmans) * 100

        add = ["", "aug"]
        settings = [
            architecture,
            f"{architecture} transfer no train conv",
            f"{architecture} transfer train conv",
        ]
        for j in range(len(s_mse_medians)):
            label_str = f"{add[c]} {settings[j]}"
            axs[0, 0].plot(
                set_sizes,
                s_mse_medians[j],
                label=label_str,
                marker="x",
            )
            axs[0, 1].plot(
                set_sizes,
                s_pearson_medians[j],
                label=label_str,
                marker="x",
            )
            axs[0, 2].plot(
                set_sizes,
                s_spearman_medians[j],
                label=label_str,
                marker="x",
            )
            axs[1, 0].plot(
                set_sizes,
                r_mse[j],
                label=label_str,
                marker="x",
            )
            axs[1, 1].plot(
                set_sizes,
                r_pearson[j],
                label=label_str,
                marker="x",
            )
            axs[1, 2].plot(
                set_sizes,
                r_spearman[j],
                label=label_str,
                marker="x",
            )

    # plot the data that the trainings should be compared to
    axs[0, 0].plot(
        set_sizes, g_split_mses, label="sequence convolution", marker="o", color="black"
    )
    axs[0, 1].plot(
        set_sizes,
        g_split_pearsons,
        label="sequence convolution",
        marker="o",
        color="black",
    )
    axs[0, 2].plot(
        set_sizes,
        g_split_spearmans,
        label="sequence convolution",
        marker="o",
        color="black",
    )

    # plot a dashed line where 100% would be in the relative performance plots
    axs[1, 0].plot(
        set_sizes,
        np.ones(len(set_sizes)) * 100,
        linestyle="dashdot",
        color="black",
        label="break_even",
    )
    axs[1, 1].plot(
        set_sizes,
        np.ones(len(set_sizes)) * 100,
        linestyle="dashdot",
        color="black",
        label="break_even",
    )
    axs[1, 2].plot(
        set_sizes,
        np.ones(len(set_sizes)) * 100,
        linestyle="dashdot",
        color="black",
        label="break_even",
    )

    # setting one legend for all plots on the right side
    box = axs[0, 2].get_position()
    axs[0, 2].set_position([box.x0, box.y0, box.width, box.height])
    axs[0, 2].legend(loc="center left", bbox_to_anchor=(1, -0.1))

    # define the appearance of the plots
    x_label = "train set size"
    axs[0, 0].set(
        xscale="log",
        yticks=np.arange(0.0, 6.0, 0.5),
        ylabel="MSE",
        xlabel=x_label,
        title="Median MSE",
    )
    axs[0, 1].set(
        xscale="log",
        yticks=np.arange(-0.3, 1.1, 0.2),
        ylabel="Correlation Coefficient",
        xlabel=x_label,
        title="Median PearsonR",
    )
    axs[0, 2].set(
        xscale="log",
        yticks=np.arange(-0.3, 1.1, 0.2),
        ylabel="Correlation Coefficient",
        xlabel=x_label,
        title="Median SpearmanR",
    )
    axs[1, 0].set(
        xscale="log",
        yticks=np.arange(-220, 190, 20),
        ylabel="Relative performance in %",
        xlabel=x_label,
        title="Relative Performance MSE",
    )
    axs[1, 1].set(
        xscale="log",
        yticks=np.arange(-130, 190, 20),
        ylabel="Relative performance in %",
        xlabel=x_label,
        title="Relative Performance PearsonR",
    )
    axs[1, 2].set(
        xscale="log",
        yticks=np.arange(-130, 190, 20),
        ylabel="Relative performance in %",
        xlabel=x_label,
        title="Relative Performance SpearmanR",
    )

    plt.show()


if __name__ == "__main__":
    prot = "pab1"
    """
    plot_reruns(prot,
            result_path="./result_files/"
                    "DenseNet_results/{}_results.csv".format(prot))
    """
    plot_reruns(
        prot,
        result_path=f"./result_files/rr5/{prot}_results.csv",
    )
