from timeit import default_timer as timer
from datetime import datetime
import time
import os
import sys
import warnings
import logging
import gc
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from d4_utils import (
    create_folder,
    log_file,
    hydrophobicity,
    h_bonding,
    charge,
    sasa,
    side_chain_length,
    aa_dict_pos,
    clear_log,
)
from d4_stats import validate, validation, pearson_spearman
from d4_split import split_inds, create_split_file
from d4_interactions import (
    atom_interaction_matrix_d,
    check_structure,
    model_interactions,
)
from d4_alignments import alignment_table
import d4_models as d4m

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEES"] = str(0)
np.set_printoptions(threshold=sys.maxsize)


def augment(data, labels, mutations, runs=3, un=False):
    """creates pseudo data from original data by adding it randomly\n
    :parameter
        - data: ndarray of strings\n
          array of variants like ['S1A', 'D35T,V20R', ...]\n
        - labels: ndarray of floats or ints\n
          array with the corresponding scores of the provided data\n
        - mutations: ndarray of ints\n
          array with number of mutations of each variant\n
        - runs: int (optional - default 3)\n
          how often the augmentation should be performed\n
        - un: bool (optional - default False)\n
          whether duplicated "new" variants should be removed\n
    :return
        - nd: ndarray of strings\n
          augmented version of data\n
        - nl: ndarray of floats or ints\n
          augmented version of labels\n
        - nm: ndarray of ints\n
          augmented version of mutations\n
    """

    # all possible indices of the data
    pos_inds = np.arange(len(labels))

    nd = []
    nl = []
    nm = []
    # do augmentation for #runs
    for i in range(runs):
        # random shuffle the inds that should be added
        np.random.shuffle(pos_inds)
        # add original labels and mutations with the original shuffled
        new_labels = labels + labels[pos_inds]
        new_mutations = mutations + mutations[pos_inds]

        new_data = []
        to_del = []
        # extract the mutations that are added and check if one contains the
        # same mutation and add this index to to_del
        # to later remove this augmentations
        for cj, (j, k) in enumerate(zip(data, data[pos_inds])):
            pos_new_data = np.sort(j.split(",") + k.split(","))
            # check the new data if it has the same mutation more than once
            # - if so add its index to the to_del(ete) ids
            if len(np.unique(pos_new_data)) != new_mutations[cj]:
                to_del += [cj]
            new_data += [",".join(pos_new_data)]
        # remove the "wrong" augmentations
        new_labels = np.delete(new_labels, to_del)
        new_mutations = np.delete(new_mutations, to_del)
        new_data = np.asarray(new_data)
        new_data = np.delete(new_data, to_del)
        nd += new_data.tolist()
        nl += new_labels.tolist()
        nm += new_mutations.tolist()

    # remove duplicated entries
    if un:
        _, uni = np.unique(nd, return_index=True)
        nd = np.asarray(nd)[uni]
        nl = np.asarray(nl)[uni]
        nm = np.asarray(nm)[uni]

    return np.asarray(nd), np.asarray(nl), np.asarray(nm)


def data_generator_vals(wt_seq, alignment_path, alignment_base):
    """returns values/ numpy arrays based on the wt_seq for the DataGenerator\n
    :parameter
        - wt_seq: str\n
          wild type sequence as str eg 'AVLI'\n
    :returns
        - hm_pos_vals: ndarray of int\n
          values for interactions with valid hydrogen bonding partners\n
        - hp_norm: float\n
          max value possible for hydrophobicity interactions\n
        - ia_norm: float\n
          max value possible for interaction ares interactions\n
        - hm_converted: ndarray of float\n
          wt_seq converted into hydrogen bonding values\n
        - hp_converted: ndarray of float\n
          wt_seq converted into hydrophobicity values\n
        - cm_converted: ndarray of float\n
          wt_seq converted into charge values\n
        - ia_converted: ndarray of float\n
          wt_seq converted into SASA values\n
        - mat_index: 2D ndarray of float\n
          symmetrical index matrix\n
        - cl_converted: ndarray of float\n
          wt_seq converted into side chain length values\n
        - cl_norm: float\n
          max value possible for interaction ares interactions\n
        - co_converted: ndarray of int\n
          wt_seq converted to amino acid positions in the alignment table\n
        -co_table: nx20 ndarray of floats\n
          each row specifies which amino acids are conserved at that
          sequence position and how conserved they are\n
        -co_rows: 1D ndarray of ints\n
          inde help with indices of each sequence position\n"""

    hm_pos_vals = np.asarray([2, 3, 6, 9])

    h_vals = list(hydrophobicity.values())
    hp_norm = np.abs(max(h_vals) - min(h_vals))
    ia_norm = max(list(sasa.values())) * 2
    cl_norm = 2 * max(side_chain_length.values())

    hm_converted = np.asarray(list(map(h_bonding.get, wt_seq)))
    hp_converted = np.asarray(list(map(hydrophobicity.get, wt_seq)))
    cm_converted = np.asarray(list(map(charge.get, wt_seq)))
    ia_converted = np.asarray(list(map(sasa.get, wt_seq)))
    cl_converted = np.asarray(list(map(side_chain_length.get, wt_seq)))
    co_converted = np.asarray(list(map(aa_dict_pos.get, wt_seq)))

    co_table, co_rows = alignment_table(alignment_path, alignment_base)

    wt_len = len(wt_seq)
    mat_size = wt_len * wt_len
    pre_mat_index = np.arange(mat_size).reshape(wt_len, wt_len) / (mat_size - 1)
    pre_mat_index = np.triu(pre_mat_index)
    mat_index = pre_mat_index + pre_mat_index.T - np.diag(np.diag(pre_mat_index))
    np.fill_diagonal(mat_index, 0)

    return (
        hm_pos_vals,
        hp_norm,
        ia_norm,
        hm_converted,
        hp_converted,
        cm_converted,
        ia_converted,
        mat_index,
        cl_converted,
        cl_norm,
        co_converted,
        co_table,
        co_rows,
    )


def progress_bar(num_batches, bar_len, batch):
    """prints progress bar with percentage that can be overwritten with a "\
            "subsequent print statement - should be implemented with "\
            "on_train_batch_end\n
        :parameter
            - num_batches: int\n
              number of batches per epoch
            - bar_len: int\n
              length of the progress bar\b
            - batch: int\n
              number of the current batch
        :return
            None\n"""
    try:
        # current bar length - how many '=' the bar needs to have at current batch
        cur_bar = int(bar_len * (bar_len * (batch / bar_len) / num_batches))
        # cur_bar = int(bar_len * (batch / num_batches))

        # to get a complete bar at the end
        if batch == num_batches - 1:
            cur_bar = bar_len
        # printing the progress bar
        bar_string = "\r[{}>{}] {:0.0f}%".format(
            "=" * cur_bar, " " * (bar_len - cur_bar), (batch + 1) / num_batches * 100
        )
        print(bar_string, end="")

        # set cursor to start of the line to overwrite progress bar when epoch
        # is done
        if num_batches - batch == 1:
            print("\r[{}>] 100%".format("=" * bar_len), end="")
            print("\r\r", end="")
    except Exception as e:
        print(e, "led to an error in the progress bar display")


class DataGenerator(keras.utils.Sequence):
    """
    Generates n_channel x n x n matrices to feed them as batches to a network"\
            "where n denotes len(wild type sequence)\n
    modified after 'https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'\n
    ...\n
    Attributes:\n
    - features: list of str\n
      features that should be encoded eg ['A2S,E3R' 'T6W']\n
    - labels: list of int / float\n
      the corresponding labels to the features\n
    - interaction_matrix: 2D ndarray of bool\n
      boolean matrix whether residues interact or not\n
    - dim: tuple\n
      dimensions of the matrices (len(wt_seq) x len(wt_seq))\n
    - n_channels: int\n
      number of matrices used\n
    - batch_size: int\n
      Batch size (if 1 gradient gets updated after every sample in training)\n
    - first_ind: int\n
      index of the start of the protein sequence\n
    - hm_converted: ndarray of floats\n
      wt sequence h-bonding encoded\n
    - hm_pos_vals: ndarray of ints\n
      valid values for h-bonding residues\n
    - factor: 2D ndarray of floats\n
      1 - norm(distance) for all residues in the interaction matrix\n
    - hp_converted: ndarray of floats\n
      wt sequence hydrophobicity encoded\n
    - hp_norm: int or float\n
      max possible value for hydrophobicity change\n
    - cm_converted: ndarray of floats\n
      wt sequence charge encoded\n
    - ia_converted: ndarray of floats\n
      wt sequence interaction area encoded\n
    - ia_norm: float\n
      max value for interaction area change\n
    - mat_index: 2D ndarray of ints\n
      symmetrical index matrix (for adjacency matrix) that represents the "\
              "position of each interaction in the matrices
    - cl_converted: ndarray of floats\n
      wild type sequence clash encoded
    - cl_norm: float\n
      normalization value for the clash matrix
    - dist_mat 2D ndarray of floats\n
      ture distances between all residues
    - dist_th
      maximum distance for residues to be counted as interaction
    - co_converted ndarray of int or floats:\n
      wild type sequence position in alignment_table encoded\n
    - co_table: ndarray or floats\n
      nx20 array- which amino acids are how conserved at which sequence "\
                 "position\n
    - co_rows: ndarray of ints\n
      indexing help for alignment_table\n      
    - shuffle: bool, (optional - default True)\n
      if True data gets shuffled after every epoch\n
    - train: bool, (optional - default True)\n
      if True Generator returns features and labels (use turing training) "\
             "else only features\n
    """

    def __init__(
        self,
        features,
        labels,
        interaction_matrix,
        dim,
        n_channels,
        batch_size,
        first_ind,
        hm_converted,
        hm_pos_vals,
        factor,
        hp_converted,
        hp_norm,
        cm_converted,
        ia_converted,
        ia_norm,
        mat_index,
        cl_converted,
        cl_norm,
        dist_mat,
        dist_th,
        co_converted,
        co_table,
        co_rows,
        shuffle=True,
        train=True,
    ):
        self.features, self.labels = features, labels
        self.interaction_matrix = interaction_matrix
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.first_ind = first_ind
        self.hm_converted = hm_converted
        self.hm_pos_vals = hm_pos_vals
        self.factor = factor
        self.hp_converted = hp_converted
        self.hp_norm = hp_norm
        self.cm_converted = cm_converted
        self.ia_converted = ia_converted
        self.ia_norm = ia_norm
        self.mat_index = mat_index
        self.cl_converted = cl_converted
        self.cl_norm = cl_norm
        self.dist_mat = dist_mat
        self.dist_th = dist_th
        self.co_converted = co_converted
        self.co_table = co_table
        self.co_rows = co_rows
        self.shuffle = shuffle
        self.train = train

    def __len__(self):
        """number of batches per epoch"""
        return int(np.ceil(len(self.features) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        features_batch = self.features[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        label_batch = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        f, l = self.__batch_variants(features_batch, label_batch)
        if self.train:
            return f, l
        else:
            return f

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idx = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __batch_variants(self, features_to_encode, corresponding_labels):
        """creates interaction matrices of variants for a batch"""
        first_dim = corresponding_labels.shape[0]
        batch_features = np.empty((first_dim, *self.dim, self.n_channels))
        batch_labels = np.empty(first_dim, dtype=float)

        for ci, i in enumerate(features_to_encode):
            # variant i encoded as matrices
            final_matrix = model_interactions(
                feature_to_encode=i,
                interaction_matrix=self.interaction_matrix,
                index_matrix=self.mat_index,
                factor_matrix=self.factor,
                distance_matrix=self.dist_mat,
                dist_thrh=self.dist_th,
                first_ind=self.first_ind,
                hmc=self.hm_converted,
                hb=h_bonding,
                hm_pv=self.hm_pos_vals,
                hpc=self.hp_converted,
                hp=hydrophobicity,
                hpn=self.hp_norm,
                cmc=self.cm_converted,
                c=charge,
                iac=self.ia_converted,
                sa=sasa,
                ian=self.ia_norm,
                clc=self.cl_converted,
                scl=side_chain_length,
                cln=self.cl_norm,
                coc=self.co_converted,
                cp=aa_dict_pos,
                cot=self.co_table,
                cor=self.co_rows,
            )

            batch_features[ci] = final_matrix
            batch_labels[ci] = corresponding_labels[ci]
        return batch_features, batch_labels


class SaveToFile(keras.callbacks.Callback):
    """writes training stats in a temp file
     ...\n
    Attributes:\n
    - features: str\n
      path where the temp.csv file should be saved
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.start_time_epoch = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        log_string = "{},{:0.4f},{:0.4f},{:0.4f},{}".format(
            str(epoch),
            logs["loss"],
            logs["val_loss"],
            time.time() - self.start_time_epoch,
            time.strftime("%H:%M:%S", time.localtime(self.start_time_epoch)),
        )
        log_file(
            self.filepath,
            write_str=log_string,
            optional_header="epoch,loss,mae,val_loss,val_mae,sec_per_epoch,epoch_start_time",
        )

    def on_train_end(self, logs=None):
        log_file(self.filepath, write_str="Finished training")


class CustomPrint(keras.callbacks.Callback):
    """prints custom stats during training\n
    ...\n
    Attributes:\n
    - num_batches: int\n
      number of batches per epoch\n
    - epoch_print: int, (optional - default 1)\n
      interval at which loss and the change in loss should be printed\n
    - epoch_stat_print: int, (optional - default 10)\n
      interval at which best train epoch, the best validation epoch and the "\
              "difference in the loss between them
      should be printed\n
    - pb_len: int, (optional - default 60)\n
      length of the progress bar\n
    - model_d: str, (optional - default '')\n
      filepath where the models should be saved\n
    - model_save_interval: int, (optional - default 5)\n
      minimum nuber of epochs to pass to save the model - only gets saved "\
              "when the validation loss has improved 
      since the last time the model was saved\n
    """

    def __init__(
        self,
        num_batches,
        epoch_print=1,
        epoch_stat_print=10,
        pb_len=60,
        model_d="",
        model_save_interval=5,
        save=False,
    ):
        self.epoch_print = epoch_print
        self.best_loss = np.Inf
        self.bl_epoch = 0
        self.best_val_loss = np.Inf
        self.bvl_epoch = 0
        self.latest_loss = 0.0
        self.latest_val_loss = 0.0
        self.epoch_stat_print = epoch_stat_print
        self.start_time_epoch = 0.0
        self.start_time_training = 0.0
        self.num_batches = num_batches
        self.pb_len = pb_len
        self.model_d = model_d
        self.epoch_since_model_save = 0
        self.model_save_interval = model_save_interval
        self.save = save

    def on_train_begin(self, logs=None):
        self.start_time_training = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time_epoch = time.time()
        if epoch == 0:
            print("*** training started ***")

    def on_train_batch_end(self, batch, logs=None):
        progress_bar(num_batches=self.num_batches, bar_len=self.pb_len, batch=batch)

    def on_epoch_end(self, epoch, logs=None):
        # loss and validation loss of this epoch
        cur_loss = logs["loss"]
        cur_val_loss = logs["val_loss"]

        if epoch % self.epoch_print == 0:
            print(
                "E {} - loss: {:0.4f}  val_loss: {:0.4f} - loss change: {:0.4f}  val_loss change: {:0.4f} - "
                "seconds per epoch: {:0.4f}\n".format(
                    str(epoch),
                    cur_loss,
                    cur_val_loss,
                    cur_loss - self.latest_loss,
                    cur_val_loss - self.latest_val_loss,
                    time.time() - self.start_time_epoch,
                )
            )
        # update the latest loss and latest validation loss to loss of this epoch
        self.latest_loss = cur_loss
        self.latest_val_loss = cur_val_loss
        # update the best loss if loss of current epoch was better
        if cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.bl_epoch = epoch
        # update the best validation loss if current epoch was better
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.bvl_epoch = epoch
            # save model if the validation loss improved since the last time i
            # it was saved and min model_save_interval epochs have passed
            if self.save:
                if epoch - self.epoch_since_model_save >= self.model_save_interval:
                    self.model.save(self.model_d, overwrite=True)
                    self.epoch_since_model_save = epoch
        # print stats of the epoch after the given epoch_stat_print interval
        if epoch % self.epoch_stat_print == 0 and epoch > 0:
            d = np.abs(self.best_loss - self.best_val_loss)
            if d != 0.0 and self.best_val_loss != 0.0:
                dp = (d / self.best_val_loss) * 100
            else:
                dp = np.nan
            d_cl = cur_loss - self.best_loss
            d_cvl = cur_val_loss - self.best_val_loss
            print(
                "Best train epoch: {}\nBest validation epoch: {}\ndelta: {:0.4f} (equals {:0.2f}% of val_loss)\n"
                "difference to best loss: {:0.4f}\ndifference to best val_loss: {:0.4f}\n".format(
                    str(self.bl_epoch), str(self.bvl_epoch), d, dp, d_cl, d_cvl
                )
            )

    def on_train_end(self, logs=None):
        # save model in the end and print overall training stats
        if self.save:
            self.model.save(self.model_d + "_end")
        print("Overall best epoch stats")
        print(
            "Best training epoch: {} with a loss of {:0.4f}".format(
                str(self.bl_epoch), self.best_loss
            )
        )
        print(
            "Best validation epoch: {} with a loss of {:0.4f}".format(
                str(self.bvl_epoch), self.best_val_loss
            )
        )
        print(
            "Total training time in minutes: {:0.1f}\n".format(
                (time.time() - self.start_time_training) / 60
            )
        )


class ClearMemory(keras.callbacks.Callback):
    """clears garbage collection and clears session after each epoch\n
    ...\n
    Attributes:\n
    None\n
    """

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def run_all(
    model_to_use,
    optimizer,
    tsv_file,
    pdb_file,
    wt_seq,
    number_mutations,
    variants,
    score,
    dist_thr,
    channel_num,
    max_train_mutations,
    training_epochs,
    test_num,
    first_ind,
    algn_path,
    algn_bl,
    r_seed=None,
    deploy_early_stop=True,
    es_monitor="val_loss",
    es_min_d=0.01,
    es_patience=20,
    es_mode="auto",
    es_restore_bw=True,
    load_trained_model=None,
    batch_size=64,
    save_fig=None,
    show_fig=False,
    write_to_log=True,
    silent=False,
    extensive_test=False,
    save_model=False,
    load_trained_weights=None,
    no_nan=True,
    settings_test=False,
    p_dir="",
    split_def=None,
    validate_training=False,
    lr=0.001,
    transfer_conv_weights=None,
    train_conv_layers=False,
    write_temp=False,
    split_file_creation=False,
    use_split_file=None,
    daug=False,
    clear_el=False,
):
    """runs all functions to train a neural network\n
    :parameter\n
    - model_to_use: function object\n
      function that returns the model\n
    - optimizer: Optimizer object\n
      keras optimizer to be used\n
    - tsv_file: str\n
      path to tsv file containing dms data of the protein of interest\n
    - pdb_file: str\n
      path to pdb file containing the structure\n
    - wt_seq: str\n
      wt sequence of the protein of interest eg. 'AVL...'\n
    - number_mutations: str\n
      how the number of mutations column is named\n
    - variants:str\n
      name of the variant column\n
    - score: str\n
      name of the score column\n
    - dist_thr: int or float\n
      threshold distances between any side chain atom to count as interacting\n
    - channel_num: int\n
      number of channels\n
    - max_train_mutations: int or None\n
      - int specifying maximum number of mutations per sequence to be used for training\n
      - None to use all mutations for training\n
    - training_epochs: int\n
      number of epochs used for training the model\n
    - test_num: int\n
      number of samples for the test after the model was trained\n
    - first_ind: int\n
      offset of the start of the sequence (when sequence doesn't start with residue 0)
    - algn_path: str\n
      path to the multiple sequence alignment in clustalw format\n
    - algn_bl: str\n
      name of the wild type sequence in the alignment file\n
    - r_seed: None, int, (optional - default None)\n
      numpy and tensorflow random seed\n
    - deploy_early_stop: bool, (optional - default True)\n
      whether early stop during training should be enabled (Ture) or not (False)\n
            - es_monitor: str, (optional - default 'val_loss')\n
              what to monitor to determine whether to stop the training or not\n
            - es_min_d: float, (optional - default 0.01)\n
              min_delta - min difference in es_monitor to not stop training\n
            - es_patience: int, (optional - default 20)\n
              number of epochs the model can try to decrease its es_monitor value for at least min_delta before
              stopping\n
            - es_mode: str, (optional - default 'auto')\n
              direction of quantity monitored in es_monitor\n
            - es_restore_bw: bool, (optional - default True)\n
              True stores the best weights of the training - False stores the last\n
    - batch_size: int, (optional - default 64)\n
      after how many samples the gradient gets updated\n
    - load_trained_model: str or None, (optional - default None)\n
      path to an already trained model or None to not load a model\n
    - save_fig: str or None, (optional - default None)\n
            - None to not save figures\n
            - str specifying the file path where the figures should be stored\n
    - show_fig: bool, (optional - default False)\n
      True to show figures\n
    - write_to_log: bool, (optional - default True)\n
      if True writes all parameters used in the log file - **should be always enabled**\n
    - silent: bool, (optional - default False)\n
      True to print stats in the terminal\n
    - extensive_test: bool, (optional - default False)\n
      if True more test are done and more detailed plots are created\n
    - save_model: bool, (optional - default False)\n
      True to save the model after training\n
    - load_trained_weights: str or None, (optional - default None)\n
      path to model of who's weights should be used None if it shouldn't be used\n
    - no_nan: bool, (optional - default True)\n
      True terminates training on nan\n
    - settings_test: bool, (optional - default False)\n
      Ture doesn't train the model and only executes everything of the function that is before model.fit()
    - p_dir: str, (optional - default '')\n
      path to the projects content root\n
    - split_def: list of int/float or None, (optional - default None)\n
      specifies the split for train, tune, test indices\n
            - float specifies fractions of the whole dataset
              eg [0.25, 0.25, 0.5] creates a train and tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            - int specifies the different number of samples per dataset
              eg [50,50,100] leads to a train and a tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            - None uses [0.8, 0.15, 0.05] as split\n
    - validate_training: bool, (optional - default False)\n
      if True validation of the training will be performed
    - lr: float, (optional - default 0.001)\n
      learning rate (how much the weights can change during an update)\n
    - transfer_conv_weights: str or None, (optional - default None)\n
      path to model who's weights of it's convolution layers should be used for transfer learning (needs to have the
      same architecture for the convolution part as the newly build model (model_to_use) or None to not transfer
      weights\n
    - train_conv_layers: bool, (optional - default False)\n
      if True convolution layers are trainable - only applies when transfer_conv_weights is not None\n
    - write_temp: bool, (optional - default False)\n
      if True writes mae, loss and time per epoch of each epoch to the temp.csv in result_files\n
    - split_file_creation: bool, (optional - default False)\n
      if True creates a directory containing train.txt, tune.txt and test.txt files that store the indices of the
      rows used from the tsv file during training, validating and testing\n
    - use_split_file: None or str, (optional - default None)\n
      if not None this needs the file_path to a directory containing splits specifying
      the 'train', 'tune', 'test' indices - these files need to be named 'train.txt', 'tune.txt' and 'test.txt'
      otherwise splits of the tsv file according to split_def will be used\n
    - daug: bool (optional - default True)\n
      True to use data augmentation\n
    - clear_el: bool (optional - default False)\n
      if True error log gets cleared before a run\n
    :return\n
        None\n
    """
    try:
        # dictionary with argument names as keys and the input as values
        arg_dict = locals()

        # convert inputs to their respective fuction
        model_to_use = getattr(d4m, model_to_use)
        architecture_name = model_to_use.__code__.co_name
        optimizer = getattr(tf.keras.optimizers, optimizer)

        # getting the proteins name
        p_name = os.path.split(tsv_file)[1].split(".")[0]

        # creating a "unique" name for protein
        time_ = str(datetime.now().strftime("%d_%m_%Y_%H%M%S")).split(" ")[0]
        name = "{}_{}".format(p_name, time_)
        print(name)

        # path of the directory where results are stored
        result_dir = os.path.join(p_dir, "result_files")
        # path where the temp_file is located
        temp_path = os.path.join(result_dir, "temp.csv")
        # path where the log_file is located
        log_file_path = os.path.join(result_dir, "log_file.csv")
        # error log file path
        error_log_path = os.path.join(result_dir, "error.log")
        # dir where models are stored
        model_base_dir = os.path.join(result_dir, "saved_models")
        recent_model_dir = os.path.join(result_dir, "saved_models", name)

        # create result dir, base model dir and recent model dir if they don't exist
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        if save_model:
            if not os.path.isdir(model_base_dir):
                os.mkdir(model_base_dir)
            if not os.path.isdir(recent_model_dir):
                os.mkdir(recent_model_dir)

        # clear temp file from previous content or create it if it doesn't exist
        clear_log(
            temp_path,
            name + "\n" + "epoch,loss,val_loss,time_in_sec,epoch_start_time\n",
        )

        # clear error.log from previous run or create it if it doesn't exist
        if not os.path.exists(error_log_path) or clear_el:
            clear_log(error_log_path)

        # resets all state generated by keras
        tf.keras.backend.clear_session()

        # check for write_to_log
        if not write_to_log:
            warnings.warn(
                "Write to log file disabled - not recommend behavior", UserWarning
            )

        # set random seed
        if r_seed is not None:
            np.random.seed(r_seed)
            tf.random.set_seed(r_seed)
            random.seed(r_seed)

        # creates a directory where plots will be saved
        if save_fig is not None:
            save_fig = os.path.join(save_fig, "plots_" + name)
            if not os.path.isdir(save_fig):
                os.mkdir(save_fig)

        if not settings_test:
            # writes used arguments to log file
            if write_to_log:
                header = (
                    "name," + ",".join(list(arg_dict.keys())) + ",training_time_in_min"
                )
                prep_values = []
                for i in list(arg_dict.values()):
                    if type(i) == list:
                        try:
                            prep_values += ["".join(i)]
                        except TypeError:
                            prep_values += [
                                "".join(str(i)).replace(",", "_").replace(" ", "")
                            ]
                    else:
                        prep_values += [str(i)]
                values = name + "," + ",".join(prep_values) + ",nan"
                log_file(log_file_path, values, header)

        starting_time = timer()

        # creating a list of the wt sequence string e.g. 'AVL...'  -> ['A', 'V', 'L',...]
        wt_seq = list(wt_seq)

        # split dataset
        ind_dict, data_dict = split_inds(
            file_path=tsv_file,
            variants=variants,
            score=score,
            number_mutations=number_mutations,
            split=split_def,
            split_file_path=use_split_file,
            test_name="stest",
        )

        # Create files with the corresponding indices of the train, tune and test splits
        if split_file_creation:
            create_split_file(
                p_dir=result_dir,
                name=name,
                train_split=ind_dict["train"],
                tune_split=ind_dict["tune"],
                test_split=ind_dict["test"],
            )

        # data to train the model on
        # variants
        train_data = data_dict["train_data"]
        # their respective scores
        train_labels = data_dict["train_labels"]
        # number of mutations per variant
        train_mutations = data_dict["train_mutations"]

        if daug:
            # original data
            otd = data_dict["train_data"]
            otl = data_dict["train_labels"]
            otm = data_dict["train_mutations"]
            ot_len = len(otl)

            # data augmentation
            decay = 0.2
            cap = 20000
            for i in range(3):
                aug_data, aug_labels, aug_mutations = augment(
                    train_data, train_labels, train_mutations, runs=4
                )
                # concatenation of original and augmented train data
                train_data = np.concatenate((train_data, aug_data))
                train_labels = np.concatenate(
                    (train_labels, aug_labels * (1 - i * decay))
                )
                train_mutations = np.concatenate((train_mutations, aug_mutations))
            nt_len = len(train_labels)

            # shuffle augmented data
            s_inds = np.arange(nt_len)
            # np.random.shuffle(s_inds)
            train_data = train_data[s_inds]
            train_labels = train_labels[s_inds]
            train_mutations = train_mutations[s_inds]
            # only use as much fake data as needed to get cap# of training data or all if not enough could be created
            if nt_len + ot_len > cap:
                # number of augmented data needed to get cap# of training data points
                need = cap - ot_len
                print("{} augmented data points created".format(str(len(train_data))))
                if need < 0:
                    need = 0
                print(
                    "{} of them and {} original data points used in training".format(
                        str(need), str(ot_len)
                    )
                )
                if need > 0:
                    train_data = np.concatenate((train_data[:need], otd))
                    train_labels = np.concatenate((train_labels[:need], otl))
                    train_mutations = np.concatenate((train_mutations[:need], otm))
                # if enough original data is available
                else:
                    train_data = otd
                    train_labels = otl
                    train_mutations = otm
            # use all the augmented data if it + original data is less than cap#
            else:
                train_data = np.concatenate((train_data, otd))
                train_labels = np.concatenate((train_labels, otl))
                train_mutations = np.concatenate((train_mutations, otm))

            print("new train split size:", len(train_data))

        # ---
        "test data restriction"
        tdr = int(len(data_dict["train_data"]) * 0.2)
        # !!! REMOVE the slicing for test_data !!!
        # ---

        # data to validate during training
        test_data = data_dict["tune_data"]  # [:tdr]
        test_labels = data_dict["tune_labels"]  # [:tdr]
        test_mutations = data_dict["tune_mutations"]  # [:tdr]

        # data the model has never seen
        unseen_data = data_dict["test_data"]
        unseen_labels = data_dict["test_labels"]
        unseen_mutations = data_dict["test_mutations"]

        # create test data for the test_generator
        if len(unseen_mutations) > 0:
            if test_num > len(unseen_data):
                test_num = len(unseen_data)
            pos_test_inds = np.arange(len(unseen_data))
            test_inds = np.random.choice(pos_test_inds, size=test_num, replace=False)
            t_data = unseen_data[test_inds]
            t_labels = unseen_labels[test_inds]
            t_mutations = unseen_mutations[test_inds]
            print(
                "\n--- will be using unseen data for final model performance evaluation ---\n"
            )
        else:
            if test_num > len(test_data):
                test_num = len(test_data)
            pos_test_inds = np.arange(len(test_data))
            test_inds = np.random.choice(pos_test_inds, size=test_num, replace=False)
            t_data = test_data[test_inds]
            t_labels = test_labels[test_inds]
            t_mutations = test_mutations[test_inds]
            print(
                "\n--- will be using validation data for evaluating the models performance ---\n"
            )

        # possible values and encoded wt_seq (based on different properties) for the DataGenerator
        (
            hm_pos_vals,
            hp_norm,
            ia_norm,
            hm_converted,
            hp_converted,
            cm_converted,
            ia_converted,
            mat_index,
            cl_converted,
            cl_norm,
            co_converted,
            co_table,
            co_rows,
        ) = data_generator_vals(wt_seq, algn_path, algn_bl)

        # distance-, factor- and interaction matrix
        dist_m, factor, comb_bool = atom_interaction_matrix_d(
            pdb_file, dist_th=dist_thr, plot_matrices=show_fig
        )

        # checks whether sequence in the pdb and the wt_seq match
        check_structure(pdb_file, comb_bool, wt_seq)

        # neural network model function
        model = model_to_use(wt_seq, channel_num)

        # load weights to model
        if load_trained_weights is not None:
            old_model = keras.models.load_model(load_trained_weights)
            model.set_weights(old_model.get_weights())

        # loads a model defined in load_trained_model and ignores the model built above
        if load_trained_model is not None:
            model = keras.models.load_model(load_trained_model)

        # load weights of a models convolutional part to a model that has a convolution part with the same architecture
        # but maybe a different/ not trained classifier
        if transfer_conv_weights is not None:
            # loads model and its weights
            trained_model = keras.models.load_model(transfer_conv_weights)
            temp_weights = [layer.get_weights() for layer in trained_model.layers]

            # which layers are conv layers (or not dense or flatten since these are sensitive to different input size)
            transfer_layers = []
            for i in range(len(trained_model.layers)):
                if i > 0:
                    l_name = trained_model.layers[i].name
                    #
                    """
                    layer_i = trained_model.layers[i]
                    if not any([isinstance(layer_i, keras.layers.Dense), isinstance(layer_i, keras.layers.Flatten)])
                        transfer_layers += [i]
                    """
                    #
                    if not any(["dense" in l_name, "flatten" in l_name]):
                        transfer_layers += [i]

            # Transfer weights to new model
            # fraction of layers that should be transferred (1. all conv layer weighs get transferred)
            fraction_to_train = 1.0  # 0.6
            for i in transfer_layers[: int(len(transfer_layers) * fraction_to_train)]:
                model.layers[i].set_weights(temp_weights[i])
                if train_conv_layers is False:
                    model.layers[i].trainable = False

            # summary of the new model
            model.summary()

        model.compile(
            optimizer(learning_rate=lr), loss="mean_absolute_error", metrics=["mae"]
        )

        all_callbacks = []
        # deploying early stop parameters
        if deploy_early_stop:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor=es_monitor,
                min_delta=es_min_d,
                patience=es_patience,
                mode=es_mode,
                restore_best_weights=es_restore_bw,
            )
            all_callbacks += [es_callback]

        # stops training on nan
        if no_nan:
            all_callbacks += [tf.keras.callbacks.TerminateOnNaN()]

        # save mae and loss to temp file
        if write_temp:
            all_callbacks += [SaveToFile(temp_path)]

        # clear Session after each epoch
        # all_callbacks += [ClearMemory()]

        # custom stats print
        # number of batches needed for status bar increments
        n_batches = int(np.ceil(len(train_data) / batch_size))
        all_callbacks += [
            CustomPrint(
                num_batches=n_batches,
                epoch_print=1,
                epoch_stat_print=10,
                model_d=recent_model_dir,
                save=save_model,
            )
        ]

        # parameters for the DataGenerator
        params = {
            "interaction_matrix": comb_bool,
            "dim": comb_bool.shape,
            "n_channels": channel_num,
            "batch_size": batch_size,
            "first_ind": first_ind,
            "hm_converted": hm_converted,
            "hm_pos_vals": hm_pos_vals,
            "factor": factor,
            "hp_converted": hp_converted,
            "hp_norm": hp_norm,
            "cm_converted": cm_converted,
            "ia_converted": ia_converted,
            "ia_norm": ia_norm,
            "mat_index": mat_index,
            "cl_converted": cl_converted,
            "cl_norm": cl_norm,
            "dist_mat": dist_m,
            "dist_th": dist_thr,
            "co_converted": co_converted,
            "co_table": co_table,
            "co_rows": co_rows,
            "shuffle": True,
            "train": True,
        }

        test_params = {
            "interaction_matrix": comb_bool,
            "dim": comb_bool.shape,
            "n_channels": channel_num,
            "batch_size": batch_size,
            "first_ind": first_ind,
            "hm_converted": hm_converted,
            "hm_pos_vals": hm_pos_vals,
            "factor": factor,
            "hp_converted": hp_converted,
            "hp_norm": hp_norm,
            "cm_converted": cm_converted,
            "ia_converted": ia_converted,
            "ia_norm": ia_norm,
            "mat_index": mat_index,
            "cl_converted": cl_converted,
            "cl_norm": cl_norm,
            "dist_mat": dist_m,
            "dist_th": dist_thr,
            "co_converted": co_converted,
            "co_table": co_table,
            "co_rows": co_rows,
            "shuffle": False,
            "train": False,
        }

        # DataGenerator for training and the validation during training
        training_generator = DataGenerator(train_data, train_labels, **params)
        validation_generator = DataGenerator(test_data, test_labels, **params)
        test_generator = DataGenerator(t_data, np.zeros(len(t_labels)), **test_params)

        # ---
        """ 
        import keras_tuner
        def build_model(hp):
            filter_num = hp.Int(
                                "filter_num", 
                                min_value=0, 
                                max_value=256, 
                                step=32, 
                                default=12
                                )
            block_num = hp.Int(
                              "block_num", 
                              min_value=1, 
                              max_value=6, 
                              step=1, 
                              default=4
                              )
            block_depth = hp.Int(
                                 "block_depth", 
                                 min_value=2, 
                                 max_value=6, 
                                 step=1, 
                                 default=4
                                 ) 
            filter_size = hp.Choice(
                                    "filter_size", 
                                    [3, 5, 7, 9]
                                    )
            e_pool = hp.Choice(
                               "e_pool", 
                               ["avg", "max"]
                               )
            l_pool = hp.Choice(
                               "l_pool", 
                               ["avg", "max"]
                               )
            classif_l = hp.Int(
                               "classif_l", 
                               min_value=0, 
                               max_value=3, 
                               step=1
                               )
            model = model_to_use(
                                 wt_seq, 
                                 channel_num, 
                                 filter_num=filter_num, 
                                 block_num=block_num, 
                                 block_depth=block_depth, 
                                 classif_l=classif_l, 
                                 e_pool=e_pool, 
                                 l_pool=l_pool,
                                 filter_size=filter_size, 
                                 bn=False
                                 )
            model.compile(
                          optimizer(learning_rate=lr), 
                          loss="mean_absolute_error", 
                          metrics="mae"
                          )
            return model
        
        tuner_dir = os.path.join(p_dir, "tuner")
        print("Tuning results will be saved in", tuner_dir)
        tuner = keras_tuner.BayesianOptimization(
                                                 hypermodel=build_model,
                                                 objective="val_loss",
                                                 max_trials=45,
                                                 executions_per_trial=2,
                                                 overwrite=True,
                                                 directory=tuner_dir,
                                                 project_name=\
                                                         "dense_net2_tune_{}"\
                                                         .format(
                                                                 len(train_data)
                                                                 ))
        tuner.search_space_summary()

        tuner.search(
                     training_generator, 
                     validation_data=validation_generator,
                     epochs=60, 
                     callbacks=[all_callbacks]
                     ) 
        tuner.results_summary()
        # best_hps = tuner.get_best_hyperparameters(5)
        # model = build_model(best_hps[0])
        """
        """
        import keras_tuner
        def build_model(hp):
            times = hp.Int(
                              "times", 
                              min_value=1, 
                              max_value=6, 
                              step=1, 
                              default=2
                              )
            num_blocks = hp.Int(
                                 "num_blocks", 
                                 min_value=1, 
                                 max_value=4, 
                                 step=1, 
                                 default=3
                                 ) 
            filter_s = hp.Choice(
                                    "filter_s", 
                                    [3, 5, 7, 9]
                                    )
            l_pool = hp.Choice(
                               "l_pool", 
                               ["avg", "max"]
                               )
            layer_base_size = hp.Int(
                                "layer_base_size", 
                                min_value=16, 
                                max_value=256, 
                                step=16, 
                                default=16
                                )
            num_dense = hp.Int(
                               "num_dense", 
                               min_value=0, 
                               max_value=5, 
                               step=1
                               )
            dense_size = hp.Int(
                                "dense_size",
                                min_value=32,
                                max_value=512,
                                step= 32
                                )
            f_b = hp.Choice(
                               "f_b", 
                               ["flat", "global"]
                               )

            model = model_to_use(
                                 wt_seq, 
                                 channel_num, 
                                 times=times,
                                 num_blocks=num_blocks,
                                 filter_s=filter_s,
                                 l_pool=l_pool,
                                 layer_base_size=layer_base_size,
                                 num_dense=num_dense,
                                 dense_size=dense_size,
                                 f_b=f_b
                                 )
            model.compile(
                          optimizer(learning_rate=lr), 
                          loss="mean_absolute_error", 
                          metrics="mae"
                          )
            return model
        
        tuner_dir = os.path.join(p_dir, "tuner")
        print("Tuning results will be saved in", tuner_dir)
        tuner = keras_tuner.BayesianOptimization(
                                                 hypermodel=build_model,
                                                 objective="val_loss",
                                                 max_trials=40,
                                                 executions_per_trial=1,
                                                 overwrite=True,
                                                 directory=tuner_dir,
                                                 project_name=\
                                                         "simple_tune_{}"\
                                                         .format(
                                                                 len(train_data)
                                                                 ))
        tuner.search_space_summary()

        tuner.search(
                     training_generator, 
                     validation_data=validation_generator,
                     epochs=70, 
                     callbacks=[all_callbacks]
                     ) 
        tuner.results_summary()
        best_hps = tuner.get_best_hyperparameters(5)
        model = build_model(best_hps[0])
        """
        # ---

        if not settings_test:
            # training
            history = model.fit(
                training_generator,
                validation_data=validation_generator,
                epochs=training_epochs,
                use_multiprocessing=True,
                workers=12,
                callbacks=[all_callbacks],
                verbose=0,
            )

            end_time = timer()

            # adds training time to result_files and replaces the nan time
            log_f = open(log_file_path, "r")
            prev_log = log_f.readlines()
            log_f.close()
            log_cont_len = len(prev_log)
            w_log = open(log_file_path, "w+")
            for ci, i in enumerate(prev_log):
                if len(prev_log) > 1:
                    if log_cont_len - ci == 1:
                        loi = i.strip().split(",")
                        loi[-1] = str(np.round((end_time - starting_time) / 60, 0))
                        w_log.write(",".join(loi) + "\n")
                    else:
                        w_log.write(i)
            w_log.close()

            # training and validation plot of the training
            if validate_training:
                try:
                    validate(
                        validation_generator,
                        model,
                        history,
                        name,
                        save_fig_v=save_fig,
                        plot_fig=show_fig,
                    )
                except ValueError:
                    print("Plotting validation failed due to nan in training")

            # calculating pearsons' r and spearman r for the test dataset
            try:
                mae, mse, pearsonr, pp, spearmanr, sp = pearson_spearman(
                    model, test_generator, t_labels
                )
                print(
                    "{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n".format(
                        "MAE",
                        mae,
                        "MSE",
                        mse,
                        "PearsonR",
                        pearsonr,
                        "PearsonP",
                        pp,
                        "SpearmanR",
                        spearmanr,
                        "SpearmanP",
                        sp,
                    )
                )
            except ValueError:
                print("MAE:", mae)
                print(
                    "Value Error while calculating statistics - most probably Nan during training."
                )

            # creating more detailed plots
            if extensive_test:
                validation(
                    model=model,
                    generator=test_generator,
                    labels=t_labels,
                    v_mutations=t_mutations,
                    p_name=p_name,
                    test_num=test_num,
                    save_fig=save_fig,
                    plot_fig=show_fig,
                    silent=silent,
                )

            # data for the result file
            result_string = ",".join(
                [
                    name,
                    architecture_name,
                    str(len(train_data)),
                    str(len(test_data)),
                    str(np.round(mae, 4)),
                    str(np.round(mse, 4)),
                    str(np.round(pearsonr, 4)),
                    str(np.round(pp, 4)),
                    str(np.round(spearmanr, 4)),
                    str(np.round(sp, 4)),
                ]
            )
            if write_to_log:
                # writing results to the result file
                log_file(
                    os.path.join(result_dir, "results.csv"),
                    result_string,
                    "name,architecture,train_data_size,test_data_size,mae,mse,pearson_r,pearson_p,spearman_r,"
                    "spearman_p",
                )

        gc.collect()
        del model

    except Exception as e:
        # writing exception to error.log
        result_dir = os.path.join(p_dir, "result_files")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            handlers=[
                logging.FileHandler(os.path.join(result_dir, "error.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.exception(e)


if __name__ == "__main__":
    pass
