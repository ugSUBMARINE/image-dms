import itertools
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from d4_predict import recall_calc, predict_score
from d4_utils import protein_settings, run_dict, aa_dict

# plt.style.use("bmh")
# plt.rcParams.update({"font.size": 12})
CM = 1 / 2.54
SET_SIZES = [50, 100, 250, 500, 1000, 2000, 6000]
COLORS = ["blue", "orange", "green", "red", "violet", "brown", "black"]
NN_COLORS = {
    "simple_model_imp": "royalblue",
    "dense_net2": "firebrick",
    "sep_conv_mix": "forestgreen",
}
MSE_RANGE = np.arange(0.0, 6.5, 0.5)
PEARSON_RANGE = np.arange(-0.3, 1.1, 0.2)
SPEARMAN_RANGE = np.arange(-0.50, 1.1, 0.2)

RELATIVE_PR = np.arange(-1.5, 1.95, 0.2)
RELATIVE_SR = np.arange(-1.5, 1.95, 0.20)
RELATIVE_MSE = np.arange(-5.20, 1.90, 0.50)
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


def get_gitter_data(protein_name):
    # get the sequence convolution data to compare the runs to
    g_data = pd.read_csv(f"nononsense/{protein_name}_test_formatted.txt", delimiter=",")
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

    return g_split_mses, g_split_pearsons, g_split_spearmans


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
            architecture = architecture + "_" + str(c)
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


def best_setting_comp(
    file_paths: list[str], setting_index: int = 2, save_fig: bool = False
) -> None:
    """plots the mse, PearsonR and SpearmanR for all 3 proteins for a chosen setting
    :parameter
        - file_paths:
          parent file paths where results of specific architecture are stored
        - setting_index:
          index of the setting of interest like listed in SETTINGS
        - save_fig:
          whether to save the plot or not
    :return
        - None
    """
    proteins = ["gb1", "pab1", "avgfp"]

    fig, ax = plt.subplots(3, 3, figsize=(32, 18))
    # column position of each protein
    protein_position = dict(zip(proteins, np.arange(len(proteins))))
    for i in file_paths:
        for p in proteins:
            # read the result csv
            arch, p_mse, p_p, p_sp = get_data(os.path.join(i, p + "_results.csv"))

            # plotting
            ax[0, protein_position[p]].plot(
                SET_SIZES, p_mse[setting_index], label=arch, marker="x"
            )
            ax[1, protein_position[p]].plot(
                SET_SIZES, p_p[setting_index], label=arch, marker="x"
            )
            ax[2, protein_position[p]].plot(
                SET_SIZES, p_sp[setting_index], label=arch, marker="x"
            )

            # appearance settings
            ax[0, protein_position[p]].set(xscale="log", yticks=MSE_RANGE)
            ax[1, protein_position[p]].set(xscale="log", yticks=PEARSON_RANGE)
            ax[2, protein_position[p]].set(xscale="log", yticks=SPEARMAN_RANGE)
            if protein_position[p] == 0:
                ax[0, 0].set_ylabel("Median MSE")
                ax[1, 0].set_ylabel("Median PearsonR")
                ax[2, 0].set_ylabel("Median SpearmanR")
            ax[2, protein_position[p]].set_xlabel("train set size")
            ax[0, protein_position[p]].set_title(p)

    # legend handeling
    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(file_paths))
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_fig:
        fig.savefig("best_setting_comp.png")
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
        result_path = "pub_result_files/rr3_results/{}_results.csv".format(
            protein_name.lower()
        )
    ERROR_ALPHA = 0.2

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


def sm_effect_heatmap(
    protein: str, trained_models: list[str], save_fig: bool = False
) -> None:
    """plots heat maps of predicted single mutation effects for each sequence position
    :parameter
        - protein:
          name of the protein the models were trained on
        - trained_models:
          file paths to the trained models
        - save_fig:
          whether to save to plot or not
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
    aas = sorted(list(aa_dict.values()))
    aa_positsion = dict(zip(aas, np.arange(len(aa_dict))))
    to_fill = np.full((len(aa_dict), len(seq)), np.nan)
    # variants of the dms dataset
    voi = list(data[mut_bool]["variant"])
    ground_truth = list(data[mut_bool]["score"])
    # adding wild type sequence position with score 0 to variants
    for i in range(offset, len(seq) + offset):
        voi.append(f"X{i}{seq[i -offset]}")
        ground_truth.append(0)

    # titles for the plots
    titles = [
        "Pretrained",
        50,
        100,
        250,
        500,
        1000,
        2000,
        6000,
        "Whole Dataset",
        "Ground Truth",
    ]

    """
    # uncomment do write new scores to file
    # calculates all scores with all supplied models
    all_scores = []
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

    t = open("pub_result_files/plot_data/sm_effect_heatmap.csv", "w+")
    t.write(",".join(np.asarray(titles[:-1], dtype=str)) + "\n")
    for i in range(len(all_scores[0])):
        t.write(",".join([str(k[i]) for k in all_scores])+"\n")
    t.close()
    """

    # all model to predictions of all scores for all (single) mutations in the dataset
    all_scores = pd.read_csv(
        "pub_result_files/plot_data/sm_effect_heatmap.csv", delimiter=","
    )

    # 20 x N maps filled with nan - to later be filled with predicted scores
    pre_maps = []
    for i in titles[:-1]:
        pre_maps.append(np.full((len(aa_dict), len(seq)), np.nan))

    # fill each pre_maps with the predicted score
    for ci, (i, j) in enumerate(zip(voi, ground_truth)):
        ind = int(i[1:-1]) - offset
        aa_i = i[-1]
        to_fill[aa_positsion[aa_i], ind] = j
        for ck, k in enumerate(titles[:-1]):
            pre_maps[ck][aa_positsion[aa_i], ind] = all_scores[str(k)].iloc[ci]

    # add ground truth to pre_maps
    pre_maps.append(to_fill)
    # plot all heat maps
    opts = {"vmin": -6, "vmax": 1}
    # uncomment for difference to ground truth
    # opts = {"vmin": 0, "vmax": 6}
    fig, ax = plt.subplots(
        10, figsize=(10.5 * CM, 29.7 * CM), layout="compressed", sharey=True
    )
    for i in range(len(titles)):
        a = ax[i].imshow(pre_maps[i], **opts)
        # uncomment do plot difference to ground truth
        # a = ax[i].imshow(np.abs(pre_maps[i] - to_fill), **opts)
        ax[i].set_title(titles[i])
        ax[i].set_yticks([])
    ax[-1].set_xlabel("Sequence Position")
    # set colorbar
    cbar_ax = fig.add_axes([0.85, 0.4, 0.05, 0.2])
    fig.colorbar(a, cax=cbar_ax)
    fig.add_subplot(111, frameon=False)
    # shared ylabel
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel("Amino Acids", labelpad=-60)
    if save_fig:
        fig.savefig(f"{protein}_sm_effect_heatmap.png")
    plt.show()


def generalization(
    doi: str = "pearson_r", num_runs: int = 3, save_fig: bool = False
) -> None:
    """plot generalization results
    :parameter
        - doi:
          data column of interest in the result file
        - num_runs:
          number of replicas deon
        - save_fig:
          True to save the plot as generalization.png
    :return
        - None
    """
    # differnt settings used during training
    settings = [
        "base",
        "pre training",
        "data augmentation",
        "pre training+\ndata augmentation",
    ]
    settings = ["B", "PT", "DA", "PT+DA"]
    # get date
    data = pd.read_csv("pub_result_files/rr5/generalization/results.csv", delimiter=",")
    architectures = np.unique(data["architecture"])
    num_settings = len(settings)
    num_architectures = len(architectures)

    # position of each setting in the plot
    setting_pos = np.arange(num_settings)

    # plot each architectures median and the std as errorbars
    fig, ax = plt.subplots(1, 1, figsize=(21 * CM, 14.8 * CM))
    for ci, i in enumerate(architectures):
        i_data = np.asarray(np.split(data[data["architecture"] == i][doi], num_runs))
        i_median = np.median(i_data, axis=0)
        i_std = i_data.std(axis=0)
        i_color = NN_COLORS[i]
        # shift errorbars so they don't overlap
        offset = 0
        if ci == 0:
            offset = -0.1
        elif ci == 2:
            offset = +0.1

        """
        ax.errorbar(
            setting_pos + offset, i_median, yerr=i_std, fmt="o", capsize=5, label=i
        )
        """
        ax.scatter(setting_pos + offset, i_median, color=i_color, marker="x")
        _, cl, _ = ax.errorbar(
            setting_pos + offset,
            i_median,
            yerr=np.max(i_data, axis=0) - i_median,
            fmt="s",
            label=i,
            lolims=True,
            color=i_color,
            linestyle="none",
        )
        for i in cl:
            i.set_marker("_")
            i.set_markersize(10)

        _, cl, _ = ax.errorbar(
            setting_pos + offset,
            i_median,
            yerr=i_median - np.min(i_data, axis=0),
            fmt="s",
            uplims=True,
            color=i_color,
            linestyle="none",
        )
        for i in cl:
            i.set_marker("_")
            i.set_markersize(10)

    ax.set_xticks(setting_pos, settings)
    ax.set_ylabel("PearsonR")
    yt = np.arange(-0.25, 1.25, 0.25)
    ax.set_yticks(yt, yt)
    fig.legend(loc="lower center", ncol=num_architectures)
    fig.tight_layout(pad=5, w_pad=1.5, h_pad=1.5)
    if save_fig:
        fig.savefig("generalization.png")
    plt.show()


def rr_metrics_combined(
    data_type, architecture, save_fig: bool = False, show_fig: bool = True
):
    first_protein_name = "pab1"
    second_protein_name = "gb1"
    third_protein_name = "avgfp"

    first_data = get_data(
        f"pub_result_files/rr5/{architecture}/{first_protein_name}_results.csv"
    )[1:]
    second_data = get_data(
        f"pub_result_files/rr5/{architecture}/{second_protein_name}_results.csv"
    )[1:]
    third_data = get_data(
        f"pub_result_files/rr5/{architecture}/{third_protein_name}_results.csv"
    )[1:]

    first_g_data = get_gitter_data(first_protein_name)
    second_g_data = get_gitter_data(second_protein_name)
    third_g_data = get_gitter_data(third_protein_name)

    if data_type == "p":
        d_range = PEARSON_RANGE
        rp_range = RELATIVE_PR
        d_name = "PearsonR"
        data_id = 1
    elif data_type == "s":
        d_range = SPEARMAN_RANGE
        rp_range = RELATIVE_SR
        d_name = "SpearmanR"
        data_id = 2
    else:
        d_range = MSE_RANGE
        rp_range = RELATIVE_MSE
        d_name = "MSE"
        data_id = 0

    fig, ax = plt.subplots(2, 3, figsize=(42 * CM, 29.7 * CM))  # , sharex=True)
    for ci, i in enumerate(SETTINGS):
        rel_data = [
            2 - (first_data[data_id][ci] / first_g_data[data_id]),
            2 - (second_data[data_id][ci] / second_g_data[data_id]),
            2 - (third_data[data_id][ci] / third_g_data[data_id]),
        ]
        if data_type == "p" or data_type == "s":
            rel_data = [
                first_data[data_id][ci] / first_g_data[data_id],
                second_data[data_id][ci] / second_g_data[data_id],
                third_data[data_id][ci] / third_g_data[data_id],
            ]
        ax[0, 0].plot(
            SET_SIZES,
            first_data[data_id][ci],
            label=i,
            marker="x",
            color=COLORS[ci],
        )
        ax[1, 0].plot(SET_SIZES, rel_data[0], label=i, marker="x", color=COLORS[ci])

        ax[0, 1].plot(
            SET_SIZES,
            second_data[data_id][ci],
            label=i,
            marker="x",
            color=COLORS[ci],
        )
        ax[1, 1].plot(SET_SIZES, rel_data[1], label=i, marker="x", color=COLORS[ci])

        ax[0, 2].plot(
            SET_SIZES,
            third_data[data_id][ci],
            label=i,
            marker="x",
            color=COLORS[ci],
        )
        ax[1, 2].plot(SET_SIZES, rel_data[2], label=i, marker="x", color=COLORS[ci])

    ax[0, 0].plot(
        SET_SIZES,
        first_g_data[data_id],
        label="sequence convolution",
        marker="^",
        color="black",
    )
    ax[1, 0].plot(SET_SIZES, np.ones(len(SET_SIZES)), color="black", linestyle="--")
    ax[0, 1].plot(
        SET_SIZES,
        second_g_data[data_id],
        label="sequence convolution",
        marker="^",
        color="black",
    )
    ax[1, 1].plot(SET_SIZES, np.ones(len(SET_SIZES)), color="black", linestyle="--")
    ax[0, 2].plot(
        SET_SIZES,
        third_g_data[data_id],
        label="sequence convolution",
        marker="^",
        color="black",
    )
    ax[1, 2].plot(SET_SIZES, np.ones(len(SET_SIZES)), color="black", linestyle="--")

    ax[0, 0].set(xscale="log", yticks=d_range, title=first_protein_name)
    ax[0, 0].set_ylabel(d_name, fontsize=12)
    ax[1, 0].set(
        xscale="log",
        yticks=rp_range,
    )
    ax[1, 0].set_xlabel("Dataset Size", fontsize=12)
    ax[1, 0].set_ylabel("Relative Performance Factor", fontsize=12)
    ax[0, 1].set(
        xscale="log",
        yticks=d_range,
        title=second_protein_name,
    )
    ax[1, 1].set(
        xscale="log",
        yticks=rp_range,
    )
    ax[1, 1].set_xlabel("Dataset Size", fontsize=12)
    ax[0, 2].set(
        xscale="log",
        yticks=d_range,
        title=third_protein_name,
    )
    ax[1, 2].set(
        xscale="log",
        yticks=rp_range,
    )
    ax[1, 2].set_xlabel("Dataset Size", fontsize=12)
    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(SETTINGS) + 1)
    plt.subplots_adjust(hspace=0.08, wspace=0.12)
    if save_fig:
        fig.savefig(f"{architecture}_{d_name}.png")
    if show_fig:
        plt.show()


def generate_recall_data() -> None:
    """plots all recall results
    :parameter
        - None
    :return
        - None
    """

    def save_data(protein_n, arch, size, data, budget_size):
        d = open("pub_result_files/plot_data/recall.txt", "a")
        d.write(
            f"{protein_n}+{arch}+{size}+{','.join(np.asarray(data, dtype=str).tolist())}+{','.join(np.asarray(budget_size, dtype=str).tolist())}\n"
        )
        d.close()

    # name of used proteins
    proteins = ["gb1", "pab1", "avgfp"]
    architectures = ["simple_model_imp", "dense_net2", "sep_conv_mix"]
    num_proteins = len(proteins)
    num_architectures = len(architectures)
    # dicts that specify the subplot position of each
    # protein-architecture combination
    protein_position = dict(zip(proteins, np.arange(num_proteins)))
    architecture_position = dict(zip(architectures, np.arange(num_architectures)))

    split_data = np.split(
        pd.read_csv(
            "pub_result_files/rr5/recall/fract_splits_results.csv", delimiter=","
        ),
        num_proteins * num_architectures,
    )
    whole_data = pd.read_csv(
        "pub_result_files/rr5/recall/whole_results.csv", delimiter=","
    )
    filenames_whole = np.asarray(whole_data["name"])

    # calculating the number of samples in the test.txt files
    prot_filelen = []
    for i in filenames_whole:
        if proteins[0] in i:
            prot_filelen.append(
                [
                    pd.read_csv(
                        os.path.join(
                            "pub_result_files/rr5/recall/recall_whole_splits/",
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
                            "pub_result_files/rr5/recall/recall_whole_splits/",
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
                            "pub_result_files/rr5/recall/recall_whole_splits/",
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

    fig, ax = plt.subplots(
        num_architectures, num_proteins, figsize=(21 * CM, 14.8 * CM)
    )
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
                "pub_result_files/saved_models/recall_fract_ds/", architecture_i, j
            )
            split_path = os.path.join(
                "pub_result_files/rr5/recall/recall_fract_splits/",
                architecture_i,
                j + "_splits0",
                "test.txt",
            )
            j_n, j_rp, j_rr, j_rb = recall_calc(
                protein_name_i,
                split_path,
                model_path,
                steps=10,
                test_size=test_num_i,
            )
            save_data(protein_name_i, architecture_i, sizes_i[cj], j_rp, j_n)
        save_data(protein_name_i, architecture_i, "random", j_rr, j_n)
        save_data(protein_name_i, architecture_i, "best_case", j_rb, j_n)

        # plotting all whole results
        protein_name_w = filenames_whole[ci].split("_")[1]
        architecture_w = whole_data["architecture"].iloc[ci]
        whole_model_path = os.path.join(
            "pub_result_files/saved_models/recall_whole_ds/", filenames_whole[ci]
        )
        whole_split_path = os.path.join(
            "pub_result_files/rr5/recall/recall_whole_splits/",
            filenames_whole[ci] + "_splits0",
            "test.txt",
        )
        w_n, w_rp, _, _ = recall_calc(
            protein_name_w,
            whole_split_path,
            whole_model_path,
            steps=10,
            test_size=test_num_i,
        )
        save_data(protein_name_w, architecture_w, "whole", w_rp, w_n)
        print(f"{architecture_i} - protein {ci} done")


def plot_recall(show_fig: bool = True, save_fig: bool = False):
    """plots recall results
    :parameter
        - show_fig:
          True to show plot
        - save_fig:
          True to save plot
    :return
        - None
    """
    architectures = ["simple_model_imp", "dense_net2", "sep_conv_mix"]
    proteins = ["pab1", "gb1", "avgfp"]
    num_arch = len(architectures)
    num_prot = len(proteins)
    a_pos = dict(zip(architectures, np.arange(num_arch)))
    p_pos = dict(zip(proteins, np.arange(num_prot)))

    fig, ax = plt.subplots(num_arch, num_prot)
    file = open("pub_result_files/plot_data/recall.txt", "r")

    c = 0.0
    for i in file:
        # proteins architectures size data steps
        i = i.split("+")
        i_size = i[2]
        i_protein = i[0]
        i_architecture = i[1]
        i_data = np.asarray(i[3].split(","), dtype=float)
        i_steps = np.asarray(i[4].split(","), dtype=float)
        i_0pos = a_pos[i_architecture]
        i_1pos = p_pos[i_protein]
        c_a = 1
        c_s = "-"
        c_c = NN_COLORS[i_architecture]
        if i_size.isdigit():
            c_a = 0.1 + c * 0.1
        elif i_size == "random":
            c_c = "silver"
            c_s = "--"
        elif i_size == "best_case":
            c_c = "black"
        elif i_size == "whole":
            ax[i_0pos, i_1pos].set(xscale="log")
            if i_1pos == 0:
                ax[i_0pos, i_1pos].set_ylabel(f"Recall\n{i_architecture}")
            if i_0pos == num_arch - 1:
                ax[i_0pos, i_1pos].set_xlabel("Budget")
            if i_0pos == 0:
                ax[i_0pos, i_1pos].set_title(i_protein)

            c = 0.0

        ax[i_0pos, i_1pos].plot(i_steps, i_data, label=i_size, color=c_c, alpha=c_a)
        th = 0.6
        if i_size == "50":
            th_pos = np.argmin(np.abs(i_data - th))
            ax[i_0pos, i_1pos].scatter(
                i_steps[th_pos], i_data[th_pos], marker="^", color="black"
            )
        if i_size == "500":
            th_pos = np.argmin(np.abs(i_data - th))
            ax[i_0pos, i_1pos].scatter(
                i_steps[th_pos], i_data[th_pos], marker="d", color="black"
            )
        if i_size == "6000":
            th_pos = np.argmin(np.abs(i_data - th))
            ax[i_0pos, i_1pos].scatter(
                i_steps[th_pos], i_data[th_pos], marker="s", color="black"
            )

        c += 1

    file.close()
    leg_lines, leg_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(leg_lines, leg_labels, loc="lower center", ncol=len(SETTINGS) + 4)
    if save_fig:
        fig.savefig(f"recall.png")
    if show_fig:
        plt.show()


if __name__ == "__main__":
    pass
    """
    plot_reruns(
        "pab1",
        "~/PycharmProjects/dms/pub_result_files/rr5/sep_conv_mix/pab1_results.csv",
    )
    """

    """
    prot = "pab1"
    comparison_plot(
        [
            f"pub_result_files/rr5/simple_model_imp/{prot}_results.csv",
            f"pub_result_files/rr5/dense_net2/{prot}_results.csv",
            f"pub_result_files/rr5/sep_conv_mix/{prot}_results.csv",
        ]
    )
    """
    """
    best_setting_comp(
        [
            "pub_result_files/rr5/simple_model_imp",
            "pub_result_files/rr5/dense_net2",
            "pub_result_files/rr5/sep_conv_mix",
        ],
        5,
    )
    """

    """
    trained_models = [
        "pub_result_files/saved_models/dense_net2_pretrained_gb1/gb1_fr_50_02_09_2022_132955/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_140133/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_140349/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_140632/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_140930/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_141220/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_141553/",
        "pub_result_files/saved_models/recall_fract_ds/dense_net2/nononsense_gb1_28_09_2022_142206/",
        "pub_result_files/saved_models/recall_whole_ds/nononsense_gb1_27_09_2022_144604/",
    ]
    """
    """
    trained_models = [
        "pub_result_files/saved_models/simple_model_imp_pretrained_gb1/gb1_fr_50_05_09_2022_190713/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_125924/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130124/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130313/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130554/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130725/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_130925/",
        "pub_result_files/saved_models/recall_fract_ds/simple_model_imp/nononsense_gb1_28_09_2022_131308/",
        "pub_result_files/saved_models/recall_whole_ds/nononsense_gb1_27_09_2022_155847/",
    ]
    """
    """
    trained_models = [
        "pub_result_files/saved_models/sep_conv_mix_pretrained_gb1/gb1_fr_50_07_10_2022_214400/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_114754/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_115133/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_115607/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_115829/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_120040/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_120331/",
        "pub_result_files/saved_models/recall_fract_ds/sep_conv_mix/nononsense_gb1_05_11_2022_121023/",
        "pub_result_files/saved_models/recall_whole_ds/nononsense_gb1_04_11_2022_115442/",
    ]
    """
    # sm_effect_heatmap("gb1", trained_models, save_fig=True)

    """
    for k in ["simple_model_imp" , "dense_net2", "sep_conv_mix"]:
        for i in ["p", "s", "m"]:
            rr_metrics_combined(i, k, save_fig=True, show_fig=False)
    
    """
    # generate_recall_data()
    # plot_recall()
    # generalization(save_fig=True)
