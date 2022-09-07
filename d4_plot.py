import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

plt.style.use("bmh")


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
    LINES_TO_PLOT = 7
    ERROR_ALPHA = 0.2
    COLORS = ["blue", "orange", "green", "red", "violet", "brown", "black"]

    # get the sequence convolution data to compare the runs to
    g_data = pd.read_csv(
        "nononsense/{}_test_formatted.txt".format(protein_name), delimiter=","
    )
    g_mses = g_data["mse"].values
    g_pearsons = g_data["pearsonr"].values
    g_spearmans = g_data["spearmanr"].values

    # split them into their respective runs
    g_mses_split = np.asarray(np.split(g_mses, 3))
    g_pearsons_split = np.asarray(np.split(g_pearsons, 3))
    g_spearmans_split = np.asarray(np.split(g_spearmans, 3))

    # mean per training set size
    g_split_mses = np.median(g_mses_split, axis=0)
    g_split_pearsons = np.median(g_pearsons_split, axis=0)
    g_split_spearmans = np.median(g_spearmans_split, axis=0)

    # errors per training set size
    g_mses_err = stats.sem(g_mses_split, axis=0)
    g_pearsons_err = stats.sem(g_pearsons_split, axis=0)
    g_spearmans_err = stats.sem(g_spearmans_split, axis=0)

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
    sep_fig_mse, sep_axs_mse = plt.subplots(LINES_TO_PLOT, 1)
    sep_fig_pearson, sep_axs_pearson = plt.subplots(LINES_TO_PLOT, 1)
    sep_fig_spearman, sep_axs_spearman = plt.subplots(LINES_TO_PLOT, 1)
    sep_count = 0

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

        rs_mse_m = np.asarray(rs_mse_m)
        rs_pearson_m = np.asarray(rs_pearson_m)
        rs_spearman_m = np.asarray(rs_spearman_m)

        # all medians
        mse_medians = np.median(rs_mse_m, axis=0)
        pearson_medians = np.median(rs_pearson_m, axis=0)
        spearman_medians = np.median(rs_spearman_m, axis=0)

        # all errors
        mse_err = stats.sem(rs_mse_m, axis=0)
        pearson_err = stats.sem(rs_pearson_m, axis=0)
        spearman_err = stats.sem(rs_spearman_m, axis=0)

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
                color=COLORS[sep_count],
            )
            axs[0, 1].plot(
                set_sizes,
                s_pearson_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            axs[0, 2].plot(
                set_sizes,
                s_spearman_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            axs[1, 0].plot(
                set_sizes,
                r_mse[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            axs[1, 1].plot(
                set_sizes,
                r_pearson[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            axs[1, 2].plot(
                set_sizes,
                r_spearman[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )

            sep_axs_mse[sep_count].plot(
                set_sizes,
                s_mse_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_mse[sep_count].fill_between(
                set_sizes,
                s_mse_medians[j] - mse_err[j],
                s_mse_medians[j] + mse_err[j],
                alpha=ERROR_ALPHA,
                color=COLORS[sep_count],
            )

            sep_axs_pearson[sep_count].plot(
                set_sizes,
                s_pearson_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_pearson[sep_count].fill_between(
                set_sizes,
                s_pearson_medians[j] - pearson_err[j],
                s_pearson_medians[j] + pearson_err[j],
                alpha=ERROR_ALPHA,
                color=COLORS[sep_count],
            )

            sep_axs_spearman[sep_count].plot(
                set_sizes,
                s_spearman_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_spearman[sep_count].fill_between(
                set_sizes,
                s_spearman_medians[j] - spearman_err[j],
                s_spearman_medians[j] + spearman_err[j],
                alpha=ERROR_ALPHA,
                color=COLORS[sep_count],
            )
            sep_count += 1

    # plot the data that the trainings should be compared to
    g_label = "sequence convolution"
    axs[0, 0].plot(
        set_sizes,
        g_split_mses,
        label=g_label,
        marker="o",
        color="black",
    )
    axs[0, 1].plot(
        set_sizes,
        g_split_pearsons,
        label=g_label,
        marker="o",
        color="black",
    )
    axs[0, 2].plot(
        set_sizes,
        g_split_spearmans,
        label=g_label,
        marker="o",
        color="black",
    )

    sep_axs_mse[-1].plot(
        set_sizes,
        g_split_mses,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_mse[-1].fill_between(
        set_sizes,
        g_split_mses - g_mses_err,
        g_split_mses + g_mses_err,
        alpha=ERROR_ALPHA,
        color="black",
    )
    sep_axs_pearson[-1].plot(
        set_sizes,
        g_split_pearsons,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_pearson[-1].fill_between(
        set_sizes,
        g_split_pearsons - g_pearsons_err,
        g_split_pearsons + g_pearsons_err,
        alpha=ERROR_ALPHA,
        color="black",
    )
    sep_axs_spearman[-1].plot(
        set_sizes,
        g_split_spearmans,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_spearman[-1].fill_between(
        set_sizes,
        g_split_spearmans - g_spearmans_err,
        g_split_spearmans + g_spearmans_err,
        alpha=ERROR_ALPHA,
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

    for i in range(LINES_TO_PLOT):
        sep_axs_mse[i].set(xscale="log")
        sep_axs_mse[i].legend(loc="upper right")
        sep_axs_pearson[i].set(xscale="log")
        sep_axs_pearson[i].legend(loc="lower right")
        sep_axs_spearman[i].set(xscale="log")
        sep_axs_spearman[i].legend(loc="lower right")
    sep_axs_mse[0].set(title="MeanSquaredError")
    sep_axs_pearson[0].set(title="Pearson Correlation Coefficient")
    sep_axs_spearman[0].set(title="Spearman Correlation Coefficient")

    plt.show()


if __name__ == "__main__":
    prot = "gb1"
    """
    plot_reruns(prot,
            result_path="./result_files/"
                    "DenseNet_results/{}_results.csv".format(prot))
    """
    plot_reruns(
        prot,
        result_path=f"./result_files/rr5/dense_net2/{prot}_results.csv",
    )
