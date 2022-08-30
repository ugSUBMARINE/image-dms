import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_res_files(names=None, set_sizes=None, res_paths=None, protein_name=None):
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance compared to Gelman et al.
    'Neural networks to learn protein sequence–function relationships from deep mutational scanning data'\n
    :parameter
        - names: list of strings or None, (optional - default None)\n
          how the different trainings are names\n
        - set_sizes: list of ints or None, (optional - default None)\n
          training set sizes\n
        - res_paths list of string or None, (optional - default None)\n
          list of file paths to the different training runs\n
        - protein_name: str (default None)\n
          how the protein is named in the result and test files\n
    :return
        None"""
    # name of the different training settings for the legend
    if names is None:
        names = [
            "simple",
            "simple transfer no train conv",
            "simple transfer train conv",
            "aug simple",
            "aug simple transfer no train conv",
            "aug simple transfer train conv",
        ]

    # train set sizes
    if set_sizes is None:
        set_sizes = [50, 100, 250, 500, 1000, 2000, 6000]

    # paths to the result files if protein name is given and no other path is specified
    if res_paths is None and protein_name is not None:
        res_paths = [
            "nononsense/first_split_run/logs_results_cnn/{}_results.csv".format(
                protein_name
            ),
            "nononsense/second_split_run/logs_results_cnn/{}_results.csv".format(
                protein_name
            ),
            "nononsense/third_split_run/logs_results_cnn/{}_results.csv".format(
                protein_name
            ),
        ]
    protein_name = protein_name.lower()

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

    # get conv mixer data
    cm_data = pd.read_csv(
        "nononsense/logs_results_convmixer/{}_results.csv".format(protein_name),
        delimiter=",",
    )[4:]

    # number of training runs (different settings / data sets)
    num_runs = 6
    split_ind = 3
    if protein_name == "avgfp":
        num_runs = 3
        split_ind = 0

    cm_mses = cm_data["mse"].values
    cm_pearsons = cm_data["pearson_r"].values
    cm_spearmans = cm_data["spearman_r"].values

    # split them into their respective runs
    # augmented ConvMixer
    if protein_name != "avgfp":
        acm_split_mses = np.median(np.split(cm_mses, num_runs)[:split_ind], axis=0)
        acm_split_pearsons = np.median(
            np.split(cm_pearsons, num_runs)[:split_ind], axis=0
        )
        acm_split_spearmans = np.median(
            np.split(cm_spearmans, num_runs)[:split_ind], axis=0
        )

    # not augmented ConvMixer
    cm_split_mses = np.median(np.split(cm_mses, num_runs)[split_ind:], axis=0)
    cm_split_pearsons = np.median(np.split(cm_pearsons, num_runs)[split_ind:], axis=0)
    cm_split_spearmans = np.median(np.split(cm_spearmans, num_runs)[split_ind:], axis=0)
    num_runs = len(cm_split_mses)

    # get the data to compare the runs to
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

    # calculating the relative performance simple CNN
    r_mse = 200 - (split_mses / g_split_mses) * 100
    r_pearson = (split_pearsons / g_split_pearsons) * 100
    r_spearman = (split_spearmans / g_split_spearmans) * 100

    # calculating the relative performance ConvMixer
    if protein_name != "avgfp":
        acm_r_mse = 200 - (acm_split_mses / g_split_mses[:num_runs]) * 100
        acm_r_pearson = (acm_split_pearsons / g_split_pearsons[:num_runs]) * 100
        acm_r_spearman = (acm_split_spearmans / g_split_spearmans[:num_runs]) * 100
    cm_r_mse = 200 - (cm_split_mses / g_split_mses[:num_runs]) * 100
    cm_r_pearson = (cm_split_pearsons / g_split_pearsons[:num_runs]) * 100
    cm_r_spearman = (cm_split_spearmans / g_split_spearmans[:num_runs]) * 100
    # print(acm_r_mse, acm_r_pearson, acm_r_spearman)
    # print(cm_r_mse, cm_r_pearson, cm_r_spearman)
    # print(r_mse)
    # print(np.max(r_mse), np.argmax(r_mse), np.max(r_pearson), np.argmax(r_pearson), np.max(r_spearman), np.argmax(r_spearman))

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

    if protein_name != "avgfp":
        # plot augmented ConvMixer results
        axs[0, 0].plot(
            set_sizes[:num_runs],
            acm_split_mses,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )
        axs[0, 1].plot(
            set_sizes[:num_runs],
            acm_split_pearsons,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )
        axs[0, 2].plot(
            set_sizes[:num_runs],
            acm_split_spearmans,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )
        axs[1, 0].plot(
            set_sizes[:num_runs],
            acm_r_mse,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )
        axs[1, 1].plot(
            set_sizes[:num_runs],
            acm_r_pearson,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )
        axs[1, 2].plot(
            set_sizes[:num_runs],
            acm_r_spearman,
            label="aug ConvMixer",
            marker="^",
            color="chartreuse",
        )

    # plot ConvMixer results
    axs[0, 0].plot(
        set_sizes[:num_runs],
        cm_split_mses,
        label="ConvMixer",
        marker="^",
        color="magenta",
    )
    axs[0, 1].plot(
        set_sizes[:num_runs],
        cm_split_pearsons,
        label="ConvMixer",
        marker="^",
        color="magenta",
    )
    axs[0, 2].plot(
        set_sizes[:num_runs],
        cm_split_spearmans,
        label="ConvMixer",
        marker="^",
        color="magenta",
    )
    axs[1, 0].plot(
        set_sizes[:num_runs], cm_r_mse, label="ConvMixer", marker="^", color="magenta"
    )
    axs[1, 1].plot(
        set_sizes[:num_runs],
        cm_r_pearson,
        label="ConvMixer",
        marker="^",
        color="magenta",
    )
    axs[1, 2].plot(
        set_sizes[:num_runs],
        cm_r_spearman,
        label="ConvMixer",
        marker="^",
        color="magenta",
    )

    # ------
    """
    d = pd.read_csv("result_files/results.csv", delimiter=",")
    mse = np.asarray(d["mse"], dtype=float)
    sp = np.asarray(d["spearman_r"], dtype=float)
    p = np.asarray(d["pearson_r"], dtype=float)
    p_s = np.split(np.asarray(np.split(p, 9)), 3)
    sp_s = np.split(np.asarray(np.split(sp, 9)), 3)
    m_s = np.split(np.asarray(np.split(mse, 9)), 3)

    to_plot = m_s
    size = [50, 100, 250, 500, 1000]
    for i in range(3):
        inter_s = []
        inter_p = []
        inter_m = []
        for j in range(3):
            inter_p += [p_s[j][i].tolist()]
            inter_s += [sp_s[j][i].tolist()]
            inter_m += [m_s[j][i].tolist()]
        axs[0, 0].plot(size, np.median(np.asarray(inter_m), axis=0), label="Simple new aug", marker="o")
        axs[0, 1].plot(size, np.median(np.asarray(inter_p), axis=0), label="Simple new aug tw", marker="o")
        axs[0, 2].plot(size, np.median(np.asarray(inter_s), axis=0), label="Simple new aug tw tl", marker="o")
        """
    # ---
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
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    axs[0, 2].set_xscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 2].set_xscale("log")

    axs[0, 0].set_yticks(np.arange(0, 8.5, 0.5))
    axs[0, 1].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[0, 2].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[1, 0].set_yticks(np.arange(-500, 170, 20))
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


def plot_reruns(protein_name, result_path=None):
    """plots MeanSquaredError, Pearsons' R, Spearman R and the relative performance compared to Gelman et al.
    'Neural networks to learn protein sequence–function relationships from deep mutational scanning data'\n
    :parameter
        - protein_name: str\n
          how the protein is named in the result and test files\n
        - result_path: list of string or None, (optional - default None)\n
          list of file paths to the different training runs\n
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
            "simple",
            "simple transfer no train conv",
            "simple transfer train conv",
        ]
        for j in range(len(s_mse_medians)):
            axs[0, 0].plot(
                set_sizes,
                s_mse_medians[j],
                label="{} {}".format(add[c], settings[j]),
                marker="x",
            )
            axs[0, 1].plot(
                set_sizes,
                s_pearson_medians[j],
                label="{} {}".format(add[c], settings[j]),
                marker="x",
            )
            axs[0, 2].plot(
                set_sizes,
                s_spearman_medians[j],
                label="{} {}".format(add[c], settings[j]),
                marker="x",
            )
            axs[1, 0].plot(
                set_sizes,
                r_mse[j],
                label="{} {}".format(add[c], settings[j]),
                marker="x",
            )
            axs[1, 1].plot(
                set_sizes,
                r_pearson[j],
                label="{} {}".format(add[c], settings[j]),
                marker="x",
            )
            axs[1, 2].plot(
                set_sizes,
                r_spearman[j],
                label="{} {}".format(add[c], settings[j]),
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
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    axs[0, 2].set_xscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 2].set_xscale("log")

    axs[0, 0].set_yticks(np.arange(0, 6.0, 0.5))
    axs[0, 1].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[0, 2].set_yticks(np.arange(-0.3, 1.1, 0.2))
    axs[1, 0].set_yticks(np.arange(-220, 190, 20))
    axs[1, 1].set_yticks(np.arange(-130, 190, 20))
    axs[1, 2].set_yticks(np.arange(-130, 190, 20))

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
    # plt.savefig("~//Downloads/pab1_dense2.png")

    plt.show()


if __name__ == "__main__":
    # easy_compare("nononsense/cm_tests/results.csv")
    # read_res_files(protein_name="pab1")
    prot = "pab1"
    """
    plot_reruns(prot,
            result_path="./result_files/"\
                    "DenseNet_results/{}_results.csv".format(prot))
    """
    plot_reruns(
        prot, result_path="./result_files/" "results.csv"
    )
