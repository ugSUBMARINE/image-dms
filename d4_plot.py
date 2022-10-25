import itertools
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from d4_predict import recall_calc, predict_score
from d4_utils import protein_settings, run_dict, aa_dict

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


def comparison_plot(paths: list, save_fig: bool = False) -> None:
    """plots comparisons between all medians of all settings
    :parameter
        - paths:
          list of file paths of the result files
        - save_fig:
          whether to save the plot
    """
    num_settings = len(SETTINGS)
    # getting the data from the data files

    fig, ax = plt.subplots(3, num_settings, figsize=(32, 18))
    arch_used = []
    used_architectures = []
    for j in range(len(paths)):
        architecture, mse, pearson, spearman = get_data(paths[j])
        if architecture in used_architectures:
            c = 1
            while architecture + "_" + str(c) in architecture:
                c += 1
            architecture = architecture + "_"  + str(c)
        used_architectures.append(architecture)
        for i in range(num_settings):
            ax[0, i].plot(
                SET_SIZES,
                mse[i],
                label=f"{architecture}",
                # color="forestgreen",
                marker="x",
            )
            ax[1, i].plot(
                SET_SIZES,
                pearson[i],
                label=f"{architecture}",
                # color="forestgreen",
                marker="x",
            )
            ax[2, i].plot(
                SET_SIZES,
                spearman[i],
                label=f"{architecture}",
                # color="forestgreen",
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
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(paths))
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_fig:
        fig.savefig(f"./{os.path.split(paths[0])[-1].split('.')[0]}.png")
    plt.show()


def compare_best(
    result_path0,
    result_path1,
    result_path2,
    result_path3,
    result_path4,
    result_path5,
    data_oi=2,
    save_fig=False,
):
    """comparison plot of all proteins for two different networks
    :parameter
        - result_path0:
          path to results of eg pab1 simple_model_imp
        - result_path1:
          path to results of eg pab1 dense_net2
        - same alternation goes on with different proteins and the other result paths
        - data_oi:
          index for SETTINGS use
        - save_fig:
          whether to save the plot
    :return
        - None
    """
    parameters = locals()
    data = []
    architectures = []
    proteins = []
    for i in list(parameters.values())[:-2]:
        architectures.append(os.path.split(os.path.split(i)[0])[1])
        data.append(get_data(i))
        proteins.append(os.path.split(i)[-1].split("_")[0])
    fig, ax = plt.subplots(3, 3, figsize=(32, 18))
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
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_fig:
        fig.savefig(f"compare_best{SETTINGS[data_oi]}.png")
    plt.show()


def plot_reruns(
    protein_name: str, result_path: list[str] | None = None, save_fig: bool = False
) -> None:
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance
    compared to Gelman et al. 'Neural networks to learn protein sequence–function
    relationships from deep mutational scanning data'
    :parameter
        - protein_name:
          how the protein is named in the result and test files
        - result_path:
          list of file paths to the different training runs
        - save_fig:
          whether to save the plot or not
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

    fig, axs = plt.subplots(2, 3, figsize=(32, 18))
    sep_fig_mse, sep_axs_mse = plt.subplots(3, 3, figsize=(32, 18))
    sep_fig_pearson, sep_axs_pearson = plt.subplots(3, 3, figsize=(32, 18))
    sep_fig_spearman, sep_axs_spearman = plt.subplots(3, 3, figsize=(32, 18))
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

    # setting one legend for all plots
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
    fig.tight_layout(pad=6, w_pad=1.5, h_pad=2)
    sep_fig_mse.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    sep_fig_pearson.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    sep_fig_spearman.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)

    if save_fig:
        fig.savefig(f"plot_reruns{protein_name}.png")
        sep_fig_mse.savefig(f"plot_reruns_mse{protein_name}.png")
        sep_fig_pearson.savefig(f"plot_reruns_pearson{protein_name}.png")
        sep_fig_spearman.savefig(f"plot_reruns_spearman{protein_name}.png")
    plt.show()


def recall_plot() -> None:
    """plots all recall results
    :parameter
        - None
    :return
        - None
    """
    fig, ax = plt.subplots(2, 3, figsize=(32, 18))
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
        ax[architecture_position[architecture_i], protein_position[protein_name_i]].set(
            xscale="log"
        )

        if protein_name_i == proteins[0]:
            ax[
                architecture_position[architecture_i],
                protein_position[protein_name_i],
            ].set_ylabel(f"Recall Percentage Top 100\n{architecture_i}")
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


def sm_effect_heatmap(protein: str, trained_models: list[str]) -> None:
    """plots heat maps of predicted single mutation effects for each sequence position
    :parameter
        - protein:
          name of the protein the models were trained on
        - trained_models:
          file paths to the trained models
    :return
        - None
    """
    # getting all the necessary data of the protein from protein_settings
    data = pd.read_csv(f"nononsense/nononsense_{protein}.tsv", delimiter="\t")
    # bool to later select only single mutations
    mut_bool = data["num_mutations"] == 1
    protein_attributes = protein_settings(protein)
    offset = int(protein_attributes["offset"])
    seq = protein_attributes["sequence"]
    aas = list(aa_dict.values())
    aa_positsion = dict(zip(list(aa_dict.values()), np.arange(len(aa_dict))))
    to_fill = np.full((len(aa_dict), len(seq)), np.nan)
    # variants of the dms dataset
    voi = list(data[mut_bool]["variant"])

    # titles for the plots
    titles = [
        "pretrained",
        50,
        100,
        250,
        500,
        1000,
        2000,
        6000,
        "whole dataset",
        "ground truth",
    ]

    # use all models to predict all scores for all (single) mutations in the dataset
    all_scores = []
    # 20 x N maps filled with 200 - to later be filled with predicted scores
    pre_maps = []
    for i in trained_models:
        score = predict_score(
            protein_pdb=f"./datasets/{protein}.pdb",
            protein_seq=list(seq),
            variant_s=voi,
            model_filepath=i,
            dist_th=20,
            algn_path=f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
            algn_base=protein,
            first_ind=offset,
        )
        all_scores.append(score)
        pre_maps.append(np.full((len(aa_dict), len(seq)), np.nan))

    # fill each pre_maps with the predicted score
    for ci, (i, j) in enumerate(
        zip(list(data[mut_bool]["variant"]), list(data[mut_bool]["score"]))
    ):
        ind = int(i[1:-1]) - offset
        aa_i = i[-1]
        to_fill[aa_positsion[aa_i], ind] = j
        for k in range(len(trained_models)):
            pre_maps[k][aa_positsion[aa_i], ind] = all_scores[k][ci]

    # add ground truth to pre_maps
    pre_maps.append(to_fill)
    # plot all heat maps
    opts = {"vmin": -6, "vmax": 1}
    fig, ax = plt.subplots(4, 3, figsize=(9, 6), layout="compressed")
    for i in range(len(titles)):
        if i < 3:
            row = 0
            col = i
        elif i >= 3 and i < 6:
            row = 1
            col = i - 3
        elif i >= 6 and i < 9:
            row = 2
            col = i - 6
        else:
            row = 3
            col = 1

        a = ax[row, col].imshow(pre_maps[i], **opts)
        ax[row, col].set_title(titles[i])
        if np.sum([row == 2, col == 0, col == 2]) == 2 or all([row == 3, col == 1]):
            ax[row, col].set_xlabel("sequence position")
        if col == 0 or all([col == 1, row == 3]):
            ax[row, col].set_ylabel("amino acids")
        ax[row, col].set_yticks(np.arange(len(aas)), aas, size="x-small", ha="center")
        plt.colorbar(a, ax=ax[row, col], shrink=0.5)
    ax[3, 0].axis("off")
    ax[3, 2].axis("off")
    plt.show()


if __name__ == "__main__":
    pass
    """ 
    plot_reruns(
        "pab1",
        "./result_files/rr5/sep_conv_mix/pab1_results.csv",
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

    # recall_plot()
    prot = "pab1"
    comparison_plot(
        [
            f"result_files/rr5/simple_model_imp/{prot}_results.csv",
            f"result_files/rr5/dense_net2/{prot}_results.csv",
            f"result_files/rr5/sep_conv_mix/{prot}_results.csv",
            f"result_files/rr5/sep_conv_res/{prot}_results.csv",
        ]
    )

    """
    protein = "gb1"
    trained_models = [
        "result_files/saved_models/simple_model_imp_pretrained_gb1/gb1_fr_50_05_09_2022_190713/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_125924/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130124/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130313/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130554/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130725/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130925/",
        "result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_131308/",
        "result_files/saved_models/recall_whole_ds/nononsense_gb1_27_09_2022_155847/",
    ]

    sm_effect_heatmap(protein, trained_models)
    """
