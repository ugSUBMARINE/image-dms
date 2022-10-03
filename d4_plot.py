import itertools
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from d4_predict import recall_calc

plt.style.use("bmh")
plt.rcParams.update({"font.size": 15})

SET_SIZES = [50, 100, 250, 500, 1000, 2000, 6000]
MSE_RANGE = np.arange(0.0, 6.5, 0.5)
PEARSON_RANGE = np.arange(-0.3, 1.1, 0.2)
SPEARMAN_RANGE = np.arange(-0.3, 1.1, 0.2)
SETTINGS = [
    "base",
    "transfer no train",
    "transfer train",
    "aug",
    "aug transfer no train",
    "aug transfer train",
]


def get_data(
    data_file_path: str, num_replica: int = 3, num_settings: int = 3
) -> tuple[
    str,
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
]:
    """reads the data of a result file and calculates the median of all runs/settings
    :parameter
        - data_file_path:
          path to the result file
        - num_replica:
          number of runs (repetitions) done
        - num_settings:
          number of different settings (eg. nothing - no train conv - train conv)
    :return
        - architecture
          name of the used architecture
        - inter_mse
          all median mse of each training set and setting
        - inter_pearson
          all median pearson r of each training set and setting
        - inter_spearman
          all median spearman r of each training set and setting
    """
    data = pd.read_csv(data_file_path, delimiter=",")
    architecture = np.unique(data["architecture"])[0]
    mse = np.asarray(data["mse"])
    pearson = np.asarray(data["pearson_r"])
    spearman = np.asarray(data["spearman_r"])

    # split for not augmented and augmented
    ana_mse = np.asarray(np.split(mse, 2))
    ana_pearson = np.asarray(np.split(pearson, 2))
    ana_spearman = np.asarray(np.split(spearman, 2))

    inter_mse = []
    inter_pearson = []
    inter_spearman = []

    for c, (m, p, s) in enumerate(zip(ana_mse, ana_pearson, ana_spearman)):
        # run split
        rs_mse = np.split(m, num_replica)
        rs_pearson = np.split(p, num_replica)
        rs_spearman = np.split(s, num_replica)

        # lists for calculating the medians
        rs_mse_m = []
        rs_pearson_m = []
        rs_spearman_m = []
        for i in range(len(rs_mse)):
            rs_mse_m.append(rs_mse[i])
            rs_pearson_m.append(rs_pearson[i])
            rs_spearman_m.append(rs_spearman[i])

        rs_mse_m = np.asarray(rs_mse_m)
        rs_pearson_m = np.asarray(rs_pearson_m)
        rs_spearman_m = np.asarray(rs_spearman_m)

        # all medians
        mse_medians = np.median(rs_mse_m, axis=0)
        pearson_medians = np.median(rs_pearson_m, axis=0)
        spearman_medians = np.median(rs_spearman_m, axis=0)

        # split medians in their different settings
        inter_mse += np.split(mse_medians, num_settings)
        inter_pearson += np.split(pearson_medians, num_settings)
        inter_spearman += np.split(spearman_medians, num_settings)

    inter_mse = np.asarray(inter_mse)
    inter_pearson = np.asarray(inter_pearson)
    inter_spearman = np.asarray(inter_spearman)

    return architecture, inter_mse, inter_pearson, inter_spearman


def comparison_plot(
    result_path0: str, result_path1: str, save_fig: bool = False
) -> None:
    """plots comparisons between all medians of all settings
    :parameter
        - result_path0:
          file path to the results for the first architecture
        - result_path1:
          file path to the results for the second architecture
        - save_fig:
          whether to save the plot
    """
    num_settings = len(SETTINGS)
    # getting the data from the data files
    first_architecture, first_mse, first_pearson, first_spearman = get_data(
        result_path0
    )
    second_architecture, second_mse, second_pearson, second_spearman = get_data(
        result_path1
    )
    if first_architecture == second_architecture:
        first_architecture = first_architecture + " 1"
        second_architecture = second_architecture + " 2"

    fig, ax = plt.subplots(3, num_settings)
    for i in range(num_settings):
        ax[0, i].plot(
            SET_SIZES,
            first_mse[i],
            label=f"{first_architecture}",
            color="forestgreen",
            marker="x",
        )
        ax[0, i].plot(
            SET_SIZES,
            second_mse[i],
            label=f"{second_architecture}",
            color="firebrick",
            marker="x",
        )
        ax[1, i].plot(
            SET_SIZES,
            first_pearson[i],
            label=f"{first_architecture}",
            color="forestgreen",
            marker="x",
        )
        ax[1, i].plot(
            SET_SIZES,
            second_pearson[i],
            label=f"{second_architecture}",
            color="firebrick",
            marker="x",
        )
        ax[2, i].plot(
            SET_SIZES,
            first_spearman[i],
            label=f"{first_architecture}",
            color="forestgreen",
            marker="x",
        )
        ax[2, i].plot(
            SET_SIZES,
            second_spearman[i],
            label=f"{second_architecture}",
            color="firebrick",
            marker="x",
        )

        # setting the appearance
        ax[0, i].set(xscale="log", yticks=MSE_RANGE)
        ax[1, i].set(xscale="log", yticks=PEARSON_RANGE)
        ax[2, i].set(xscale="log", yticks=SPEARMAN_RANGE)
        if i == 0:
            ax[0, i].set_ylabel("Median MSE")
            ax[1, i].set_ylabel("Median PearsonR")
            ax[2, i].set_ylabel("Median SpearmanR")
        ax[2, i].set_xlabel("train set size")
        ax[0, i].set_title(SETTINGS[i])

    # setting one legend for all plots on the right side
    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=2)
    fig.set_size_inches(32, 18)
    fig.savefig(f"./{os.path.split(result_path0)[-1].split('.')[0]}.png")
    plt.show()


def compare_best(
    result_path0,
    result_path1,
    result_path2,
    result_path3,
    result_path4,
    result_path5,
    data_oi=2,
):
    parameters = locals()
    data = []
    architectures = []
    proteins = []
    for i in list(parameters.values())[:-1]:
        architectures.append(os.path.split(os.path.split(i)[0])[1])
        data.append(get_data(i))
        proteins.append(os.path.split(i)[-1].split("_")[0])
    fig, ax = plt.subplots(3, 3)
    for i in range(6):
        if i % 2 == 0:
            col = "forestgreen"
        else:
            col = "firebrick"
        # to plot dense and simple on the same column per protein
        i_plot = i // 2
        ax[0, i_plot].plot(
            SET_SIZES,
            data[i][1][data_oi],
            label=architectures[i],
            marker="x",
            color=col,
        )
        ax[1, i_plot].plot(
            SET_SIZES,
            data[i][2][data_oi],
            label=architectures[i],
            marker="x",
            color=col,
        )
        ax[2, i_plot].plot(
            SET_SIZES,
            data[i][3][data_oi],
            label=architectures[i],
            marker="x",
            color=col,
        )
        # setting the appearance
        ax[0, i_plot].set(xscale="log", yticks=MSE_RANGE)
        ax[1, i_plot].set(xscale="log", yticks=PEARSON_RANGE)
        ax[2, i_plot].set(xscale="log", yticks=SPEARMAN_RANGE)
        if i == 0:
            ax[0, i_plot].set_ylabel("Median MSE")
            ax[1, i_plot].set_ylabel("Median PearsonR")
            ax[2, i_plot].set_ylabel("Median SpearmanR")
        ax[2, i_plot].set_xlabel("train set size")
        ax[0, i_plot].set_title(proteins[i])

    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=2)
    fig.set_size_inches(32, 18)
    # fig.savefig(f"compare_best{SETTINGS[data_oi]}.png")
    plt.show()


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

    # min and maxes
    g_mses_err_min = np.min(g_mses_split, axis=0)
    g_pearsons_err_min = np.min(g_pearsons_split, axis=0)
    g_spearmans_err_min = np.min(g_spearmans_split, axis=0)
    g_mses_err_max = np.max(g_mses_split, axis=0)
    g_pearsons_err_max = np.max(g_pearsons_split, axis=0)
    g_spearmans_err_max = np.max(g_spearmans_split, axis=0)

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
    sep_fig_mse, sep_axs_mse = plt.subplots(3, 3)
    sep_fig_pearson, sep_axs_pearson = plt.subplots(3, 3)
    sep_fig_spearman, sep_axs_spearman = plt.subplots(3, 3)
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
            rs_mse_m.append(rs_mse[i])
            rs_pearson_m.append(rs_pearson[i])
            rs_spearman_m.append(rs_spearman[i])

        rs_mse_m = np.asarray(rs_mse_m)
        rs_pearson_m = np.asarray(rs_pearson_m)
        rs_spearman_m = np.asarray(rs_spearman_m)

        # all medians
        mse_medians = np.median(rs_mse_m, axis=0)
        pearson_medians = np.median(rs_pearson_m, axis=0)
        spearman_medians = np.median(rs_spearman_m, axis=0)

        # min and maxes
        mse_err_min = np.asarray(np.split(np.min(rs_mse_m, axis=0), 3))
        pearson_err_min = np.asarray(np.split(np.min(rs_pearson_m, axis=0), 3))
        spearman_err_min = np.asarray(np.split(np.min(rs_spearman_m, axis=0), 3))
        mse_err_max = np.asarray(np.split(np.max(rs_mse_m, axis=0), 3))
        pearson_err_max = np.asarray(np.split(np.max(rs_pearson_m, axis=0), 3))
        spearman_err_max = np.asarray(np.split(np.max(rs_spearman_m, axis=0), 3))

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
            f"{architecture} transfer no train",
            f"{architecture} transfer train",
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

            row_num = 0
            if sep_count > 2:
                row_num = 1
            sep_axs_mse[row_num, j].plot(
                set_sizes,
                s_mse_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_mse[row_num, j].fill_between(
                set_sizes,
                mse_err_min[j],
                mse_err_max[j],
                alpha=ERROR_ALPHA,
                color=COLORS[sep_count],
            )

            sep_axs_pearson[row_num, j].plot(
                set_sizes,
                s_pearson_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_pearson[row_num, j].fill_between(
                set_sizes,
                pearson_err_min[j],
                pearson_err_max[j],
                alpha=ERROR_ALPHA,
                color=COLORS[sep_count],
            )

            sep_axs_spearman[row_num, j].plot(
                set_sizes,
                s_spearman_medians[j],
                label=label_str,
                marker="x",
                color=COLORS[sep_count],
            )
            sep_axs_spearman[row_num, j].fill_between(
                set_sizes,
                spearman_err_min[j],
                spearman_err_max[j],
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

    sep_axs_mse[2, 1].plot(
        set_sizes,
        g_split_mses,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_mse[2, 1].fill_between(
        set_sizes,
        g_mses_err_min,
        g_mses_err_max,
        alpha=ERROR_ALPHA,
        color="black",
    )
    sep_axs_pearson[2, 1].plot(
        set_sizes,
        g_split_pearsons,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_pearson[2, 1].fill_between(
        set_sizes,
        g_pearsons_err_min,
        g_pearsons_err_max,
        alpha=ERROR_ALPHA,
        color="black",
    )
    sep_axs_spearman[2, 1].plot(
        set_sizes,
        g_split_spearmans,
        label=g_label,
        marker="o",
        color="black",
    )
    sep_axs_spearman[2, 1].fill_between(
        set_sizes,
        g_spearmans_err_min,
        g_spearmans_err_max,
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
    # box = axs[0, 2].get_position()
    # axs[0, 2].set_position([box.x0, box.y0, box.width, box.height])
    # axs[0, 2].legend(loc="center left", bbox_to_anchor=(1, -0.1))
    leg_lines, leg_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=4)

    # define the appearance of the plots
    x_label = "train set size"
    mse_range = np.arange(0.0, 6.0, 0.5)
    pearson_range = np.arange(-0.3, 1.1, 0.2)
    spearman_range = np.arange(-0.3, 1.1, 0.2)
    axs[0, 0].set(
        xscale="log",
        yticks=mse_range,
        ylabel="MSE",
        xlabel=x_label,
        title="Median MSE",
    )
    axs[0, 1].set(
        xscale="log",
        yticks=pearson_range,
        ylabel="Correlation Coefficient",
        xlabel=x_label,
        title="Median PearsonR",
    )
    axs[0, 2].set(
        xscale="log",
        yticks=spearman_range,
        ylabel="Correlation Coefficient",
        xlabel=x_label,
        title="Median SpearmanR",
    )
    axs[1, 0].set(
        xscale="log",
        yticks=np.arange(-520, 190, 50),
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

    for i in list(itertools.product([0, 1, 2], repeat=2)):
        sep_axs_mse[i].set(xscale="log", yticks=mse_range, ylabel="MSE", xlabel=x_label)
        sep_axs_mse[i].legend(loc="upper right")
        sep_axs_pearson[i].set(
            xscale="log", yticks=pearson_range, ylabel="PearsonR", xlabel=x_label
        )
        sep_axs_pearson[i].legend(loc="lower right")
        sep_axs_spearman[i].set(
            xscale="log", yticks=spearman_range, ylabel="SpearmanR", xlabel=x_label
        )
        sep_axs_spearman[i].legend(loc="lower right")
    sep_axs_mse[0, 1].set(title="MeanSquaredError")
    sep_axs_pearson[0, 1].set(title="Pearson Correlation Coefficient")
    sep_axs_spearman[0, 1].set(title="Spearman Correlation Coefficient")
    sep_axs_mse[-1, 0].axis("off")
    sep_axs_pearson[-1, 0].axis("off")
    sep_axs_spearman[-1, 0].axis("off")
    sep_axs_mse[-1, -1].axis("off")
    sep_axs_pearson[-1, -1].axis("off")
    sep_axs_spearman[-1, -1].axis("off")

    plt.show()


def recall_plot() -> None:
    """plots all recall results
    :parameter
        - None
    :return
        - None
    """
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    # name of used proteins
    proteins = ["avgfp", "gb1", "pab1"]
    # dicts that specify the subplot position of each
    # protein-architecture combination
    protein_position = dict(zip(proteins, np.arange(len(proteins))))
    architecture_position = dict(zip(["simple_model_imp", "dense_net2"], np.arange(2)))

    split_data = np.split(
        pd.read_csv("result_files/rr5/recall/fract_splits_results.csv", delimiter=","),
        6,
    )
    whole_data = pd.read_csv("result_files/rr5/recall/whole_results.csv", delimiter=",")
    filenames_whole = np.asarray(whole_data["name"])

    # calculating the number of samples in the test.txt files
    prot_filelen = []
    for i in filenames_whole:
        if proteins[0] in i:
            prot_filelen.append(
                [
                    pd.read_csv(
                        os.path.join(
                            "result_files/rr5/recall/recall_whole_splits/",
                            i + "_splits0",
                            "test.txt",
                        ),
                        delimiter=",",
                    ).shape[0],
                    proteins[0],
                ]
            )
        elif proteins[1] in i:
            prot_filelen.append(
                [
                    pd.read_csv(
                        os.path.join(
                            "result_files/rr5/recall/recall_whole_splits/",
                            i + "_splits0",
                            "test.txt",
                        ),
                        delimiter=",",
                    ).shape[0],
                    proteins[1],
                ]
            )
        elif proteins[2] in i:
            prot_filelen.append(
                [
                    pd.read_csv(
                        os.path.join(
                            "result_files/rr5/recall/recall_whole_splits/",
                            i + "_splits0",
                            "test.txt",
                        ),
                        delimiter=",",
                    ).shape[0],
                    proteins[2],
                ]
            )
    fl = np.unique(prot_filelen, axis=0)
    # dict specifying the number of test samples per protein
    test_num = dict(zip(fl[:, 1], fl[:, 0].astype(int)))

    for ci, i in enumerate(split_data):
        # names of the runs
        filename_i = np.asarray(i["name"])
        # train data size of each run
        sizes_i = np.asarray(i["train_data_size"])
        # architecture of the runs in i
        architecture_i = np.asarray(i["architecture"])[0]
        # protein of the runs in i
        protein_name_i = filename_i[0].split("_")[1]

        # setting the number of test samples to a max of 5000
        tni = test_num[protein_name_i]
        if 5000 - tni >= 0:
            test_num_i = tni
        else:
            test_num_i = 5000

        # plotting all fraction results
        for cj, j in enumerate(filename_i):
            model_path = os.path.join(
                "result_files/saved_models/recall_fract_ds/", architecture_i, j
            )
            split_path = os.path.join(
                "result_files/rr5/recall/recall_fract_splits/",
                architecture_i,
                j + "_splits0",
                "test.txt",
            )
            j_n, j_rp, j_rr, j_rb = recall_calc(
                protein_name_i,
                split_path,
                model_path,
                steps=50,
                test_size=test_num_i,
            )
            ax[
                architecture_position[architecture_i],
                protein_position[protein_name_i],
            ].plot(j_n, j_rp, label=f"fract {sizes_i[cj]}")
        ax[
            architecture_position[architecture_i],
            protein_position[protein_name_i],
        ].plot(j_n, j_rr, label="random")
        ax[
            architecture_position[architecture_i],
            protein_position[protein_name_i],
        ].plot(j_n, j_rb, label="best case")

        # setting labels and titles for the plots as well as the log scale
        ax[
            architecture_position[architecture_i], protein_position[protein_name_i]
        ].set(xscale="log")

        if protein_name_i == proteins[0]:
            ax[
                architecture_position[architecture_i],
                protein_position[protein_name_i],
            ].set_ylabel("Recall Percentage Top 100")
        if architecture_position[architecture_i] == 1:
            ax[1, protein_position[protein_name_i]].set_xlabel("Budget")
        if architecture_position[architecture_i] == 0:
            ax[0, protein_position[protein_name_i]].set_title(protein_name_i)

        # plotting all whole results
        protein_name_w = filenames_whole[ci].split("_")[1]
        architecture_w = whole_data["architecture"].iloc[ci]
        whole_model_path = os.path.join(
            "result_files/saved_models/recall_whole_ds/", filenames_whole[ci]
        )
        whole_split_path = os.path.join(
            "result_files/rr5/recall/recall_whole_splits/",
            filenames_whole[ci] + "_splits0",
            "test.txt",
        )
        w_n, w_rp, _, _ = recall_calc(
            protein_name_w,
            whole_split_path,
            whole_model_path,
            steps=50,
            test_size=test_num_i,
        )
        ax[
            architecture_position[architecture_w], protein_position[protein_name_w]
        ].plot(w_n, w_rp, label="whole data")

    # setting one legend for all plots on the right side
    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(SET_SIZES) + 3)
    plt.show()


if __name__ == "__main__":
    prot = "pab1"
    """
    plot_reruns(prot,
            result_path="./result_files/"
                    "DenseNet_results/{}_results.csv".format(prot))
    """
    """
    plot_reruns(
        prot,
        result_path=f"./result_files/rr5/simple_model_imp/{prot}_results.csv",
    )
    """
    """
    plot_reruns(
        prot,
        result_path=f"./result_files/rr5/simple_model_imp/{prot}_results.csv",
    )
    """
    """
    prot = "pab1"
    comparison_plot(
        f"./result_files/rr5/dense_net2/{prot}_results.csv",
        f"./result_files/rr5/simple_model_imp/{prot}_results.csv",
    )
    """
    """
    compare_best(
        "./result_files/rr5/dense_net2/pab1_results.csv",
        "./result_files/rr5/simple_model_imp/pab1_results.csv",
        "./result_files/rr5/dense_net2/gb1_results.csv",
        "./result_files/rr5/simple_model_imp/gb1_results.csv",
        "./result_files/rr5/dense_net2/avgfp_results.csv",
        "./result_files/rr5/simple_model_imp/avgfp_results.csv",
        data_oi=5,
    )
    """
    recall_plot()
