import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def read_res_files(names=None, set_sizes=None, res_paths=None, protein_name=None):
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance compared to Gelman et al.\n
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
        res_paths = ["nononsense/first_split_run/logs_results/{}_results.csv".format(protein_name),
                     "nononsense/second_split_run/logs_results/{}_results.csv".format(protein_name),
                     "nononsense/third_split_run/logs_results/{}_results.csv".format(protein_name)]

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

    # get the data to compare the runs to
    g_data = pd.read_csv("nononsense/{}_test_formatted.txt".format(protein_name), delimiter=",")
    g_mses = g_data["mse"].values
    g_pearsons = g_data["pearsonr"].values
    g_spearmans = g_data["spearmanr"].values

    # split them into their respective runs
    g_split_mses = np.median(np.split(g_mses, 3), axis=0)
    g_split_pearsons = np.median(np.split(g_pearsons, 3), axis=0)
    g_split_spearmans = np.median(np.split(g_spearmans, 3), axis=0)

    # calculating the relative performance
    r_mse = (g_split_mses / split_mses) * 100
    r_pearson = (split_pearsons / g_split_pearsons) * 100
    r_spearman = (split_spearmans / g_split_spearmans) * 100

    # creating the plot
    fig, axs = plt.subplots(2, 3)
    # plots for MeanSquaredError as well as PearsonR and SpearmanR of the training runs
    for i in range(6):
        axs[0, 0].plot(set_sizes, split_mses[i], label=names[i], marker="x")
        axs[0, 1].plot(set_sizes, split_pearsons[i], label=names[i], marker="x")
        axs[0, 2].plot(set_sizes, split_spearmans[i], label=names[i], marker="x")
        axs[1, 0].plot(set_sizes, r_mse[i], label=names[i], marker="x")
        axs[1, 1].plot(set_sizes, r_pearson[i], label=names[i], marker="x")
        axs[1, 2].plot(set_sizes, r_spearman[i], label=names[i], marker="x")

    # plot the data that the trainings should be compared to
    axs[0, 0].plot(set_sizes, g_split_mses, label="gitter", marker="o", color="black")
    axs[0, 1].plot(set_sizes, g_split_pearsons, label="gitter", marker="o", color="black")
    axs[0, 2].plot(set_sizes, g_split_spearmans, label="gitter", marker="o", color="black")

    # plot a dashed line where 100% would be in the relative performance plots
    axs[1, 0].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="firebrick", label="break_even")
    axs[1, 1].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="firebrick", label="break_even")
    axs[1, 2].plot(set_sizes, np.ones(len(set_sizes)) * 100, linestyle="dashdot", color="firebrick", label="break_even")

    # setting one legend for all plots on the right side
    box = axs[0, 2].get_position()
    axs[0, 2].set_position([box.x0, box.y0, box.width, box.height])
    axs[0, 2].legend(loc='center left', bbox_to_anchor=(1, -0.1))

    # appearance of the plots
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    axs[0, 2].set_xscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 2].set_xscale("log")

    axs[0, 0].set_yticks(np.arange(0, 5.5, 0.5))
    axs[0, 1].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[0, 2].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[1, 0].set_yticks(np.arange(50, 160, 20))
    axs[1, 1].set_yticks(np.arange(-30, 160, 20))
    axs[1, 2].set_yticks(np.arange(-30, 160, 20))

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
    easy_compare("result_files/results.csv")
    # read_res_files(protein_name="gb1")
