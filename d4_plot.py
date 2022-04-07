import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def read_res_files(names=None, set_sizes=None, res_paths=None, protein_name=None):
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance compared to Gelman et al.
        'Neural networks to learn protein sequence–function relationships from deep mutational scanning data'\n
        :parameter
            names: list of strings or None, (optional - default None)\n
            how the different trainings are names\n
            set_sizes: list of ints or None, (optional - default None)\n
            training set sizes\n
            res_paths list of string or None, (optional - default None)\n
            list of file paths to the different training runs\n
            protein_name: str or None (default None)\n
            how the protein is named in the result and test files\n
        :return
            None"""
    # name of the different training settings for the legend
    if names is None:
        names = ["normal", "normal transfer no train conv", "normal transfer train conv",
                 "aug normal", "aug normal transfer no train conv", "aug normal transfer train conv"]

    # train set sizes
    if set_sizes is None:
        set_sizes = [50, 100, 250, 500, 1000, 2000, 6000]

    # paths to the result files if protein name is given and no other path is specified
    if res_paths is None and protein_name is not None:
        res_paths = ["nononsense/first_split_run/logs_results_cnn/{}_results.csv".format(protein_name),
                     "nononsense/second_split_run/logs_results_cnn/{}_results.csv".format(protein_name),
                     "nononsense/third_split_run/logs_results_cnn/{}_results.csv".format(protein_name)]

    # extracting the data from the files
    mses = []
    pearsons = []
    spearmans = []
    for i in res_paths:
        i_data = pd.read_csv(i, delimiter=",")
        mses += [i_data["mse"].values]
        pearsons += [i_data["pearson_r"].values]
        spearmans += [i_data["spearman_r"].values]

    mses = np.asarray(mses)
    pearsons = np.asarray(pearsons)
    spearmans = np.asarray(spearmans)

    # splitting the data in the respective training runs
    split_mses = np.split(np.median(mses, axis=0), 6)
    split_pearsons = np.split(np.median(pearsons, axis=0), 6)
    split_spearmans = np.split(np.median(spearmans, axis=0), 6)

    # test whether ConvMixer results are available for the protein
    cm_available = True
    try:
        # get conv mixer data
        cm_data = pd.read_csv("nononsense/logs_results_convmixer/{}_results.csv".format(protein_name), delimiter=",")[4:]
    except FileNotFoundError:
        cm_available = False

    if cm_available and protein_name == "avgfp":
        cm_available = False

    if cm_available:
        cm_mses = cm_data["mse"].values
        cm_pearsons = cm_data["pearson_r"].values
        cm_spearmans = cm_data["spearman_r"].values

        # split them into their respective runs
        # augmented ConvMixer
        acm_split_mses = np.median(np.split(cm_mses, 6)[:3], axis=0)
        acm_split_pearsons = np.median(np.split(cm_pearsons, 6)[:3], axis=0)
        acm_split_spearmans = np.median(np.split(cm_spearmans, 6)[:3], axis=0)

        # not augmented ConvMixer
        cm_split_mses = np.median(np.split(cm_mses, 6)[3:], axis=0)
        cm_split_pearsons = np.median(np.split(cm_pearsons, 6)[3:], axis=0)
        cm_split_spearmans = np.median(np.split(cm_spearmans, 6)[3:], axis=0)
        num_runs = len(acm_split_mses)

    # get the data to compare the runs to
    g_data = pd.read_csv("nononsense/{}_test_formatted.txt".format(protein_name), delimiter=",")
    g_mses = g_data["mse"].values
    g_pearsons = g_data["pearsonr"].values
    g_spearmans = g_data["spearmanr"].values

    # split them into their respective runs
    g_split_mses = np.median(np.split(g_mses, 3), axis=0)
    g_split_pearsons = np.median(np.split(g_pearsons, 3), axis=0)
    g_split_spearmans = np.median(np.split(g_spearmans, 3), axis=0)

    # calculating the relative performance simple CNN
    r_mse = 200 - (split_mses / g_split_mses) * 100
    r_pearson = (split_pearsons / g_split_pearsons) * 100
    r_spearman = (split_spearmans / g_split_spearmans) * 100

    if cm_available:
        # calculating the relative performance ConvMixer
        acm_r_mse = 200 - (acm_split_mses / g_split_mses[:num_runs]) * 100
        acm_r_pearson = (acm_split_pearsons / g_split_pearsons[:num_runs]) * 100
        acm_r_spearman = (acm_split_spearmans / g_split_spearmans[:num_runs]) * 100
        cm_r_mse = 200 - (cm_split_mses / g_split_mses[:num_runs]) * 100
        cm_r_pearson = (cm_split_pearsons / g_split_pearsons[:num_runs]) * 100
        cm_r_spearman = (cm_split_spearmans / g_split_spearmans[:num_runs]) * 100

    # creating the plot
    fig, axs = plt.subplots(2, 3)
    # plots for MeanSquaredError as well as PearsonR and SpearmanR of the simple CNN training runs
    for i in range(6):
        axs[0, 0].plot(set_sizes, split_mses[i], label=names[i], marker="x")
        axs[0, 1].plot(set_sizes, split_pearsons[i], label=names[i], marker="x")
        axs[0, 2].plot(set_sizes, split_spearmans[i], label=names[i], marker="x")
        axs[1, 0].plot(set_sizes, r_mse[i], label=names[i], marker="x")
        axs[1, 1].plot(set_sizes, r_pearson[i], label=names[i], marker="x")
        axs[1, 2].plot(set_sizes, r_spearman[i], label=names[i], marker="x")

    if cm_available:
        # plot ConvMixer results
        axs[0, 0].plot(set_sizes[:num_runs], acm_split_mses, label="aug conv_mixer", marker="^", color="chartreuse")
        axs[0, 1].plot(set_sizes[:num_runs], acm_split_pearsons, label="aug conv_mixer", marker="^", color="chartreuse")
        axs[0, 2].plot(set_sizes[:num_runs], acm_split_spearmans, label="aug conv_mixer", marker="^", color="chartreuse")
        axs[1, 0].plot(set_sizes[:num_runs], acm_r_mse, label="aug conv_mixer", marker="^", color="chartreuse")
        axs[1, 1].plot(set_sizes[:num_runs], acm_r_pearson, label="aug conv_mixer", marker="^", color="chartreuse")
        axs[1, 2].plot(set_sizes[:num_runs], acm_r_spearman, label="aug conv_mixer", marker="^", color="chartreuse")

        axs[0, 0].plot(set_sizes[:num_runs], cm_split_mses, label="conv_mixer", marker="^", color="magenta")
        axs[0, 1].plot(set_sizes[:num_runs], cm_split_pearsons, label="conv_mixer", marker="^", color="magenta")
        axs[0, 2].plot(set_sizes[:num_runs], cm_split_spearmans, label="conv_mixer", marker="^", color="magenta")
        axs[1, 0].plot(set_sizes[:num_runs], cm_r_mse, label="conv_mixer", marker="^", color="magenta")
        axs[1, 1].plot(set_sizes[:num_runs], cm_r_pearson, label="conv_mixer", marker="^", color="magenta")
        axs[1, 2].plot(set_sizes[:num_runs], cm_r_spearman, label="conv_mixer", marker="^", color="magenta")

    if protein_name == "avgfp":
        d = pd.read_csv("nononsense/logs_results_convmixer/avgfp_results.csv", delimiter=",")
        num_runs = len(d)
        cm_split_mses = d["mse"]
        cm_split_pearsons = d["pearson_r"]
        cm_split_spearmans = d["spearman_r"]
        axs[0, 0].plot(set_sizes[:num_runs], cm_split_mses, label="conv_mixer", marker="^", color="magenta")
        axs[0, 1].plot(set_sizes[:num_runs], cm_split_pearsons, label="conv_mixer", marker="^", color="magenta")
        axs[0, 2].plot(set_sizes[:num_runs], cm_split_spearmans, label="conv_mixer", marker="^", color="magenta")

    # plot the data that the trainings should be compared to
    axs[0, 0].plot(set_sizes, g_split_mses, label="sequence convolution", marker="o", color="black")
    axs[0, 1].plot(set_sizes, g_split_pearsons, label="sequence convolution", marker="o", color="black")
    axs[0, 2].plot(set_sizes, g_split_spearmans, label="sequence convolution", marker="o", color="black")

    # plot a dashed line where 100% would be in the relative performance plots
    axs[1, 0].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="black", label="break_even")
    axs[1, 1].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="black", label="break_even")
    axs[1, 2].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="black", label="break_even")

    # setting one legend for all plots on the right side
    box = axs[0, 2].get_position()
    axs[0, 2].set_position([box.x0, box.y0, box.width, box.height])
    axs[0, 2].legend(loc='center left', bbox_to_anchor=(1, -0.1))

    # define the appearance of the plots
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    axs[0, 2].set_xscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 2].set_xscale("log")

    axs[0, 0].set_yticks(np.arange(0, 5.5, 0.5))
    axs[0, 1].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[0, 2].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[1, 0].set_yticks(np.arange(-160, 170, 20))
    axs[1, 1].set_yticks(np.arange(-100, 170, 20))
    axs[1, 2].set_yticks(np.arange(-100, 170, 20))

    axs[0, 0].set_ylabel("MSE")
    axs[0, 1].set_ylabel("Correlation Coefficient")
    axs[0, 2].set_ylabel("Correlation Coefficient")
    axs[1, 0].set_ylabel("Relative performance in %")
    axs[1, 1].set_ylabel("Relative performance in %")
    axs[1, 2].set_ylabel("Relative performance in %")

    axs[0, 0].set_xlabel("training set size")
    axs[0, 1].set_xlabel("training set size")
    axs[0, 2].set_xlabel("training set size")
    axs[1, 0].set_xlabel("training set size")
    axs[1, 1].set_xlabel("training set size")
    axs[1, 2].set_xlabel("training set size")

    axs[0, 0].title.set_text("Median MSE")
    axs[0, 1].title.set_text("Median PearsonR")
    axs[0, 2].title.set_text("Median SpearmanR")
    axs[1, 0].title.set_text("Relative Performance MSE")
    axs[1, 1].title.set_text("Relative Performance  PearsonR")
    axs[1, 2].title.set_text("Relative Performance  SpearmanR")

    plt.show()


def easy_compare(file_path, data_plot="sp"):
    data = pd.read_csv(file_path)
    mse = data["mse"].values
    spearman = data["spearman_r"].values
    pearson = data["pearson_r"].values
    x_ = np.arange(len(mse))
    if data_plot == "mse":
        plt.scatter(x_, mse, label="mse", color="forestgreen")
    elif data_plot == "sp":
        plt.scatter(x_, spearman, label="spearman", color="forestgreen")
    elif data_plot == "pe":
        plt.scatter(x_, pearson, label="pearson", color="forestgreen")
    else:
        print("wrong data_plot input")
    plt.show()


if __name__ == "__main__":
    # easy_compare("nononsense/cm_tests/results.csv")
    read_res_files(protein_name="avgfp")
