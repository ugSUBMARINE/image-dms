import scipy.stats
import numpy as np
from matplotlib import pyplot as plt


def validate(generator_v, model_v, history_v, name_v, max_train_mutations_v, save_fig_v=None, plot_fig=False):
    # get loss, accuracy and history of the previous trained model and plotting it
    test_loss, test_acc = model_v.evaluate(generator_v, verbose=2)
    train_val = history_v.history['mae']
    val_val = history_v.history['val_mae']
    plt.plot(train_val, label='mae', color="forestgreen")
    plt.plot(val_val, label='val_mae', color="firebrick")
    plt.xlabel('Epoch')
    plt.ylabel('mae')
    all_vals = np.concatenate((train_val, val_val))
    ymin = np.min(all_vals)
    ymax = np.max(all_vals)
    plt.ylim([ymin - ymin * 0.1, ymax + ymax * 0.1])
    plt.legend()
    # epoch with the best weights
    epochs_bw = np.argmin(train_val)
    # stats
    val_text = name_v + "\nmax_mut_train: " + str(max_train_mutations_v) + "\nepochs for best weights: " + str(
        epochs_bw) + "\nmae: " + str(np.around(test_loss, decimals=4))
    plt.gcf().text(0.5, 0.9, val_text, fontsize=14)
    plt.subplots_adjust(left=0.5)
    if save_fig_v is not None:
        plt.savefig(save_fig_v + "/" + "history_" + name_v)
    if plot_fig:
        plt.show()
    return val_val, epochs_bw, test_loss


def pearson_spearman(model, generator, labels):
    """calculating the pearson r and spearman r for generator\n
        :parameter
            generator: DataGenerator object\n
            Data generator to predict values (not shuffled)\n
            labels: ndarray\n
            the corresponding labels for the generator\n
        :return
            pearson_r: float\n
            Pearson’s correlation coefficient\n
            pearson_r_p: float \n
            Two-tailed p-value\n
            spearman_r: float\n
            Spearman correlation coefficient\n
            spearman_r_p: float\n
            p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated\n
            """
    pred = model.predict(generator).flatten()
    ground_truth = labels
    pearson_r, pearson_r_p = scipy.stats.pearsonr(ground_truth.astype(float), pred.astype(float))
    spearman_r, spearman_r_p = scipy.stats.spearmanr(ground_truth.astype(float), pred.astype(float))
    return pearson_r, pearson_r_p, spearman_r, spearman_r_p


def validation(model, generator, labels, v_mutations, p_name, test_num,
               save_fig=None, plot_fig=False, silent=True):
    """plot validations\n
        :parameter
            generator: DataGenerator object\n
            data generator for predicting values\n
            labels: ndarray\n
            the corresponding real labels to the generator\n
            v_mutations: ndarray\n
            number of mutations per data sample in the generator\n
            p_name: str\n
            protein name\n
            test_num: int\n
            number of samples used for the test\n
            save_fig: None, optional\n
            anything beside None to save figures or None to not save figures\n
            plot_fig: bool, optional\n
            if True shows figures\n
            silent: bool, optional\n
            if True doesn't write mean error in the terminal
        :return
            None
            """
    # predicted values, errors between prediction and label, number of mutations per label
    pred = model.predict(generator).flatten()
    all_errors = np.abs(pred - labels)
    mutations = v_mutations

    # sort the errors acording to the number of mutations
    mut_sort = np.argsort(mutations)
    mutations = np.asarray(mutations)[mut_sort]
    all_errors = all_errors[mut_sort]

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

    # histogram of number of each mutations present in the features used in the test
    ax1.hist(x=mutations, bins=np.arange(1, np.max(mutations) + 1, 1), color="forestgreen")
    ax1.set_ylabel("occurrence")
    ax1.set_xlabel("mutations")
    ax1.xaxis.set_label_position("top")
    ax1.set_xlim([np.min(mutations), np.max(mutations)])
    ax1.tick_params(top=True, bottom=False, labelbottom=False, labeltop=True)
    ax1.set_xticks(np.arange(1, np.max(mutations) + 1, 1))

    c = ["firebrick", "black"]
    cc = []
    for i in range(len(mutations)):
        if i % 2 == 0:
            cc += [c[0]]
        else:
            cc += [c[1]]
    # amount of values for each number of mutation
    _, w = np.unique(mutations, return_counts=True)

    wx = []
    prev_ind = 0
    mean_error_per_mutations = []
    for i in range(len(w)):
        if i == 0:
            wx += [0]
        else:
            wx += [wx[-1] + w[i - 1]]
        # mean of all errors when i number of mutations are present
        # value is as often in mean_error_per_mutations as often i number of mutations are present
        mean_error_per_mutations += [np.mean(all_errors[prev_ind:prev_ind + int(w[i])])] * int(w[i])
        prev_ind += int(w[i])
    # which errors origin from how many mutations
    ax2.bar(x=wx, width=w, height=[np.max(all_errors)] * len(w), color=cc, align="edge", alpha=0.25)
    # errors of each prediction illustrated as point
    ax2.scatter(np.arange(len(all_errors)), all_errors, color="yellowgreen", label="error", s=3)
    # ax2.plot(np.arange(len(mutations)), np.asarray(mutations) / 10, color="firebrick")
    # mean error of all errors originating from certain number of mutations
    ax2.plot(np.arange(len(mutations)), np.asarray(mean_error_per_mutations), color="firebrick",
             label="mean error per mutation")
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("absolute error")
    ax2.legend(loc="upper right")

    # histogram of how often an error of magnitude "y" occurred
    ax3.hist(all_errors, bins=np.arange(0, np.max(all_errors), 0.01), orientation="horizontal", color="forestgreen")
    ax3.set_xlabel("occurrence")
    ax3.tick_params(left=False, labelleft=False)
    plt.tight_layout()

    test_pearson_r, test_pearson_rp = scipy.stats.pearsonr(labels.astype(float), pred.astype(float))
    test_spearman_r, test_spearman_rp = scipy.stats.spearmanr(labels.astype(float), pred.astype(float))

    test_text = p_name + "\nsample number: " + str(test_num) + "\nmean_error: " + str(
        np.round(np.mean(all_errors), decimals=4)) + "\npearson r: " + str(
        np.around(test_pearson_r, 4)) + "\nspearman r: " + str(np.around(test_spearman_r, 4))
    plt.gcf().text(0.7, 0.8, test_text, fontsize=14)
    if save_fig is not None:
        plt.savefig(save_fig + "/" + "test_" + p_name)
    if plot_fig:
        plt.show()

    # boxplot of errors per number of mutations
    fig = plt.figure(figsize=(10, 10))
    boxes = []
    for i in range(1, np.max(mutations) + 1):
        i_bool = mutations == i
        boxes += [all_errors[i_bool].tolist()]
    plt.boxplot(boxes)
    plt.xticks(range(1, np.max(mutations) + 1))
    if save_fig is not None:
        plt.savefig(save_fig + "/" + "boxplot_" + p_name)
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
        plt.savefig(save_fig + "/" + "correlation_" + p_name)
    if plot_fig:
        plt.show()


if __name__ == "__main__":
    pass
