import os

import scipy.stats
import numpy as np
from matplotlib import pyplot as plt


def validate(generator_v, model_v, history_v, name_v, save_fig_v=None, plot_fig=False):
    # get loss, accuracy and history of the previous trained model and plotting it
    test_loss, test_acc = model_v.evaluate(generator_v, verbose=0)
    train_val = history_v.history["mae"]
    val_val = history_v.history["val_mae"]
    plt.plot(train_val, label="mae", color="forestgreen")
    plt.plot(val_val, label="val_mae", color="firebrick")
    plt.xlabel("Epoch")
    plt.ylabel("mae")
    all_vals = np.concatenate((train_val, val_val))
    ymin = np.min(all_vals)
    ymax = np.max(all_vals)
    plt.ylim([ymin - ymin * 0.1, ymax + ymax * 0.1])
    plt.legend()
    if save_fig_v is not None:
        plt.savefig(os.path.join(save_fig_v, "history_" + name_v))
    if plot_fig:
        plt.show()


def pearson_spearman(model, generator, labels):
    """calculating the pearson r and spearman r for predicted values
    :parameter
        generator: DataGenerator object
        Data generator to create data used to predict values (not shuffled)
        labels: ndarray
        the corresponding labels for the generator
    :return
        mae: float
        mean absolute error
        mse: float
        mean squared error
        pearson_r: float
        Pearson’s correlation coefficient
        pearson_r_p: float
        Two-tailed p-value
        spearman_r: float
        Spearman correlation coefficient
        spearman_r_p: float
        p-value for a hypothesis test whose null hypothesis is that two sets of
        data are uncorrelated
    """
    # predicted values
    pred = model.predict(generator).flatten()
    # real values
    ground_truth = labels
    pearson_r, pearson_r_p = scipy.stats.pearsonr(
        ground_truth.astype(float), pred.astype(float)
    )
    spearman_r, spearman_r_p = scipy.stats.spearmanr(
        ground_truth.astype(float), pred.astype(float), nan_policy="raise"
    )
    diff = pred - ground_truth
    # mean absolute error
    mae = np.mean(np.abs(diff))
    # mean squared error
    mse = np.mean(diff**2)
    return mae, mse, pearson_r, pearson_r_p, spearman_r, spearman_r_p


def validation(
    model,
    generator,
    labels,
    v_mutations,
    p_name,
    test_num,
    save_fig=None,
    plot_fig=False,
    silent=True,
):
    """plot validations
    :parameter
        generator: DataGenerator object
        data generator for predicting values
        labels: ndarray
        the corresponding real labels to the generator
        v_mutations: ndarray
        number of mutations per data sample in the generator
        p_name: str
        protein name
        test_num: int
        number of samples used for the test
        save_fig: str or None, (optional - default None)
        - None to not save figures
        - str specifying the file path where the figures should be stored
        plot_fig: bool, (optional - default False)
        if True shows figures
        silent: bool, (optional - default True)
        if True doesn't write mean error in the terminal
    :return
        None
    """
    # predicted values, errors between prediction and label,
    # number of mutations per label
    pred = model.predict(generator).flatten()
    all_errors = np.abs(pred - labels)
    mutations = v_mutations

    # sort the errors according to the number of mutations
    mut_sort = np.argsort(mutations)
    mutations = np.asarray(mutations)[mut_sort]
    all_errors = all_errors[mut_sort]

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

    # histogram of number of mutations present in the features used in the test
    ax1.hist(
        x=mutations, bins=np.arange(1, np.max(mutations) + 1, 1), color="forestgreen"
    )
    ax1.set_ylabel("occurrence")
    ax1.set_xlabel("mutations")
    ax1.xaxis.set_label_position("top")
    ax1.set_xlim([np.min(mutations), np.max(mutations)])
    ax1.tick_params(top=True, bottom=False, labelbottom=False, labeltop=True)
    ax1.set_xticks(np.arange(1, np.max(mutations) + 1, 1))

    c = ["firebrick", "white"]
    cc = []
    for i in range(len(mutations)):
        if i % 2 == 0:
            cc.append(c[0])
        else:
            cc.append(c[1])
    # amount of values for each number of mutation
    _, w = np.unique(mutations, return_counts=True)

    wx = []
    prev_ind = 0
    mean_error_per_mutations = []
    for i in range(len(w)):
        if i == 0:
            wx.append(0)
        else:
            wx.append(wx[-1] + w[i - 1])
        # mean of all errors when i number of mutations are present
        # value is as often in mean_error_per_mutations as often i 
        # number of mutations are present
        mean_error_per_mutations += [
            np.mean(all_errors[prev_ind : prev_ind + int(w[i])])
        ] * int(w[i])
        prev_ind += int(w[i])
    # which errors origin from how many mutations
    ax2.bar(
        x=wx,
        width=w,
        height=[np.max(all_errors)] * len(w),
        color=cc,
        align="edge",
        alpha=0.25,
    )
    # errors of each prediction illustrated as point
    ax2.scatter(
        np.arange(len(all_errors)), all_errors, color="yellowgreen", label="error", s=3
    )
    # ax2.plot(np.arange(len(mutations)), np.asarray(mutations) / 10, color="firebrick")
    # mean error of all errors originating from certain number of mutations
    ax2.plot(
        np.arange(len(mutations)),
        np.asarray(mean_error_per_mutations),
        color="firebrick",
        label="mean error per mutation",
    )
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("absolute error")
    ax2.legend(loc="upper right")

    # histogram of how often an error of magnitude "y" occurred
    ax3.hist(
        all_errors,
        bins=np.arange(0, np.max(all_errors), 0.01),
        orientation="horizontal",
        color="forestgreen",
    )
    ax3.set_xlabel("occurrence")
    ax3.tick_params(left=False, labelleft=False)
    plt.tight_layout()

    test_pearson_r, test_pearson_rp = scipy.stats.pearsonr(
        labels.astype(float), pred.astype(float)
    )
    test_spearman_r, test_spearman_rp = scipy.stats.spearmanr(
        labels.astype(float), pred.astype(float)
    )

    test_text = (
        p_name
        + "\nsample number: "
        + str(test_num)
        + "\nmean absolute error: "
        + str(np.round(np.mean(all_errors), decimals=4))
        + "\npearson r: "
        + str(np.around(test_pearson_r, 4))
        + "\nspearman r: "
        + str(np.around(test_spearman_r, 4))
    )
    plt.gcf().text(0.7, 0.8, test_text, fontsize=14)
    if save_fig is not None:
        plt.savefig(os.path.join(save_fig, "test_" + p_name))
    if plot_fig:
        plt.show()

    # boxplot of errors per number of mutations
    fig = plt.figure(figsize=(10, 10))
    boxes = []
    for i in range(1, np.max(mutations) + 1):
        i_bool = mutations == i
        boxes.append(all_errors[i_bool].tolist())
    plt.boxplot(boxes)
    plt.xticks(range(1, np.max(mutations) + 1))
    if save_fig is not None:
        plt.savefig(os.path.join(save_fig, "boxplot_" + p_name))
    plt.ylabel("error")
    plt.xlabel("number of mutations")
    if plot_fig:
        plt.show()
    if not silent:
        print("mean error:", np.mean(all_errors))

    # correlation scatter plot
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(labels, pred, color="forestgreen", s=3)
    tr = max(np.max(pred), np.max(labels))
    bl = min(np.min(pred), np.min(labels))
    plt.xlabel("true score")
    plt.ylabel("predicted score")
    plt.plot([tr, bl], [tr, bl], color="firebrick")
    if save_fig is not None:
        plt.savefig(os.path.join(save_fig, "correlation_" + p_name))
    if plot_fig:
        plt.show()


if __name__ == "__main__":
    pass
