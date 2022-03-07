import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import tensorflow as tf
from tensorflow import keras
from timeit import default_timer as timer
import os
import sys
import warnings
from datetime import datetime

from d4_utils import create_folder, log_file, hydrophobicity, h_bonding, charge, sasa, side_chain_length
from d4_stats import validate, validation, pearson_spearman
from d4_data import check_structure, data_coord_extraction, split_inds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=sys.maxsize)


def dist_calc(arr1, arr2):
    """calculates distance between arr1 and arr2 and returns a 2D array with all distances of all arr1 points
        against all arr2 points\n
        :parameter
            arr1, arr2: ndarray\n
            2D arrays of 1D lists with 3D coordinates eg [[x1, y1, z1],...]\n
        :return
            dist: 2D ndarray\n
            len(arr1) x len(arr2) distance matrix between arr1 and arr2\n"""
    # get only the x,y,z coordinates from the input arrays and reshape them so they can be subtracted from each other
    arr1_coords_rs = arr1.reshape(arr1.shape[0], 1, 3)
    arr2_coord_rs = arr2.reshape(1, arr2.shape[0], 3)
    # calculating the distance between each point and returning a 2D array with all distances
    dist = np.sqrt(((arr1_coords_rs - arr2_coord_rs) ** 2).sum(axis=2))
    return dist


def atom_interaction_matrix_d(path_to_pdb_file, dist_th=10, plot_matrices=False):
    """computes the adjacency matrix for a given pdb file based on the closest side chain atoms\n
            :parameter
                path_to_pdb_file: str\n
                path to pdb file of the protein of interest\n
                dist_th: int or float, optional\n
                maximum distance in \u212B of atoms of two residues to be seen as interacting\n
                plot_matrices: bool,optional\n
                if True plots matrices for (from left to right)
                    - distance to the closest side chain atom per residue\n
                    - distance between all side chain atoms\n
                    - inverse normalized 1st plot\n
                    - distance between CA atoms\n
                    - all interacting residues\n
            :returns
                adjacency is given per residue (the closest atom to any side chain atom of any other residue)\n
                red2: adjacency matrix of the given protein as 2d numpy array\n
                red2_nome: inverse of the normalized red2: (1 - (red2 / np.max(red2))\n
                interacting_residues: boolean matrix - which residues interact\n"""
    # data [[ATOM, RES, CHAIN, ResNR],..]
    data, coords = data_coord_extraction(path_to_pdb_file)
    if plot_matrices:
        cab = data[:, 0] == "CA"
        dca = dist_calc(coords[cab], coords[cab])

    # to get only data and coords that belong to side chain atoms
    main_chain_label = np.invert(np.isin(data[:, 0], np.asarray(["C", "CA", "N", "O"])))
    data = data[main_chain_label]
    coords = coords[main_chain_label]

    # distance between all atoms
    d = dist_calc(coords, coords)

    # getting the start and end of each residues' entry in data
    udata, uind, ucount = np.unique(data[:, 1:], axis=0, return_index=True, return_counts=True)
    # sort it again by chain and sequence
    u_sort = np.lexsort((udata[:, 2].astype(int), udata[:, 1]))
    # udata = udata[u_sort]
    uind = uind[u_sort]
    ucount = ucount[u_sort]

    # reduce all distances to the closest distance of one side chain atom to another per residue
    red1 = []
    for i, j in zip(uind, ucount):
        red = np.min(d[:, i:i + j], axis=1)
        red1 += [red.tolist()]
    red1 = np.asarray(red1)

    red2 = []
    for i, j in zip(uind, ucount):
        red_ = np.min(red1[:, i:i + j], axis=1)
        red2 += [red_.tolist()]
    red2 = np.asarray(red2)

    # excluding the diagonal, distances > dist_th and normalization for red2_norm
    np.fill_diagonal(red2, dist_th + 1)
    r2_bool = red2 > dist_th
    red2[r2_bool] = 0
    red2_norm = 1 - (red2 / np.max(red2))
    red2_norm[r2_bool] = 0
    interacting_residues = np.invert(r2_bool)

    if plot_matrices:
        fig = plt.figure()
        ax1 = fig.add_subplot(151)
        f = ax1.imshow(red2)
        plt.colorbar(f, ax=ax1)
        ax2 = fig.add_subplot(152)
        f2 = ax2.imshow(d)
        plt.colorbar(f2, ax=ax2)
        ax3 = fig.add_subplot(153)
        f3 = ax3.imshow(red2_norm)
        plt.colorbar(f3, ax=ax3)
        ax4 = fig.add_subplot(154)
        f4 = ax4.imshow(dca)
        plt.colorbar(f4, ax=ax4)
        ax5 = fig.add_subplot(155)
        f5 = ax5.imshow(interacting_residues.astype(int))
        plt.colorbar(f5, ax=ax5)
        ax1.title.set_text("red2")
        ax2.title.set_text("all side chain atoms")
        ax3.title.set_text("inverse normalized red2")
        ax4.title.set_text("CA distances")
        ax5.title.set_text("interacting residues")
        plt.show()

    return red2, red2_norm, interacting_residues


def hydrophobicity_matrix(res_bool_matrix, converted, norm):
    """matrix that represents how similar its pairs are in terms of hydrophobicity only for pairs that are true in
        res_bool_matrix\n
        :parameter
            res_bool_matrix: 2D ndarray of float or int\n
            matrix (len(wt_seq) x len(wt_seq)) where pairs that obey distance and angle criteria are True\n
            converted: ndarray of int or floats\n
            the sequence converted to the values of the corresponding dict\n
            norm: float or int\n
            max value possible for interactions between two residues\n
        :return
            hp_matrix: 2d ndarray of floats\n
            len(wt_seq) x len(wt_seq) matrix with the corresponding normalized similarity in terms of hydrophobicity
            of each pair\n"""
    interactions = np.abs(converted - converted.reshape(len(converted), -1))
    hp_matrix = 1 - (interactions / norm)
    hp_matrix[np.invert(res_bool_matrix)] = 0
    return hp_matrix


def hbond_matrix(res_bool_matrix, converted, valid_vals):
    """matrix that represents whether pairs can form h bonds (True) or not (False) only for pairs that are true in
       res_bool_matrix\n
        :parameter
            res_bool_matrix: 2D ndarray of float or int\n
            matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True\n
            valid_vals: ndarray of int or float\n
            which values of the matrix are True after multiplying the encoded sequence against itself\n
            converted: ndarray of int or floats\n
            the sequence converted to the values of the corresponding dict\n
        :return
            hb_mat: 2d ndarray of floats\n
            len(wt_seq) x len(wt_seq) matrix where pairs that can form h bonds are True\n"""
    interactions = converted * converted.reshape(len(converted), -1)
    hb_matrix = np.isin(interactions, valid_vals)
    hb_mat = np.all(np.stack((hb_matrix, res_bool_matrix)), axis=0)
    return hb_mat


def charge_matrix(res_bool_matrix, converted, good, mid, bad):
    """matrix that represents whether pairs of amino acids are of the same charge (0), of opposite charge /
       both uncharged (1), or one charged one neutral (0.5) only for pairs that are true in res_bool_matrix\n
        :parameter
            res_bool_matrix: 2D ndarray of float\n
            matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True\n
            good, mid, bad: ndarrays of int or float\n
            each holds the possible values for the different 'quality' of interaction\n
            converted: ndarray of int or floats\n
            the sequence converted to the values of the corresponding dict\n
        :return
            c_mat: 2d ndarray of floats\n
            len(wt_seq) x len(wt_seq) matrix containing the 'interaction quality value' for all interacting residues\n
            """
    interactions = converted * converted.reshape(len(converted), -1)
    interactions[np.invert(res_bool_matrix)] = 0
    interactions[np.isin(interactions, bad)] = 0
    interactions[np.isin(interactions, mid)] = 0.5
    interactions[np.isin(interactions, good)] = 1
    return interactions


def interaction_area(res_bool_matrix, wt_converted, mut_converted, norm):
    """matrix that represents the change in solvent accessible surface area (SASA) due to a mutation
       only for pairs that are true in res_bool_matrix\n
        :parameter
            res_bool_matrix: 2D ndarray of float\n
            matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True\n
            wt_converted: ndarray of float or int\n
            wild type sequence converted with the corresponding dict\n
            mut_converted: ndarray of float or int\n
            mutated sequence converted with the corresponding dict\n
            norm: int or float\n
            max value possible for interactions between two residues\n
        :return
            ia_matrix: len(wt_seq) x len(wt_seq) matrix with values corresponding to the
            absolute magnitude of change in the SASA of a residue pair\n"""
    d = wt_converted - mut_converted
    dd = np.abs(d + d.reshape(len(d), -1)) / norm
    dd[np.invert(res_bool_matrix)] = 0
    dd = 1 - dd
    return dd


def clashes(res_bool_matrix, wt_converted, mut_converted, norm, dist_mat, dist_thr):
    """matrix that represents whether clashes ore holes are occurring due to the given mutations
        :parameter
            res_bool_matrix: 2D ndarray of float\n
            matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True\n
            wt_converted: ndarray of float or int\n
            wild type sequence converted with the corresponding dict\n
            mut_converted: ndarray of float or int\n
            mutated sequence converted with the corresponding dict\n
            norm: int or float\n
            max value possible for interactions between two residues\n
            dist_mat: 2D ndarray of float\n
            matrix with distances between all residues\n
            dist_thr: int or float\n
            threshold for how close residues need to be to count as interacting\n
        :return
            sub_norm: 2D ndarray of float\n
            len(wt_seq) x len(wt_seq) matrix with values corresponding to whether new mutations lead to potential
            clashes or holes between interacting residues"""
    diff = wt_converted - mut_converted
    new_mat = (diff + diff.reshape(len(diff), -1)) * res_bool_matrix
    sub = dist_mat - new_mat
    sub_ = sub * (sub != dist_mat)
    sub_norm = sub_ / (norm + dist_thr)
    return sub_norm


def mutate_sequences(wt_sequence, mutations, f_dict, first_ind):
    """mutates the wild type sequence at positions defined in mutations and returns the mutated sequences\n
        :parameter
            wt_sequence: ndarray of float or int\n
            the encoded wild type sequence as ndarray e.g. [0.3, 0.8, 0.1, 1.]\n
            mutations: list of str\n
            list of strings where the mutations take place e.g. ['F1K,K2G', 'R45S']\n
            f_dict: dict\n
            dictionary with values for encoding\n
            first_ind: int\n
            int that denotes the number of the first residue (e.g. if protein sequence starts with RES #3 first_ind=3)
        return:
            mutated_sequences: list of float or int\n
            mutated sequences as list\n"""
    a_to_mut = wt_sequence.copy()
    muts = mutations.strip().split(",")
    for j in muts:
        j = j.strip()
        a_to_mut[int(j[1:-1]) - first_ind] = f_dict[j[-1]]
    return a_to_mut


class DataGenerator(keras.utils.Sequence):
    """
    Generates n_channel x n x n matrices to feed them as batches to a network where n denotes len(wild type sequence)\n
    ...
    Attributes:\n
    - features: list of str\n
      features that should be encoded ec ['A2S,E3R' 'T6W']\n
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
    - ch_good_vals: int\n
      values for +-, nn interactions\n
    - ch_mid_vals: int\n
      values for n+, n- interactions\n
    - ch_bad_vals: int\n
      values for --, ++ interactions\n
    - ia_converted: ndarray of floats\n
      wt sequence interaction area encoded\n
    - ia_norm: float\n
      max value for interaction area change\n
    - mat_index: 2D ndarray of ints\n
      symmetrical index matrix (for adjacency matrix) that represents the position of each interaction in the matrices
    - cl_converted: ndarray of floats\n
      wild type sequence clash encoded
    - cl_norm: float\n
      normalization value for the clash matrix
    - dist_mat 2D ndarray of floats\n
      ture distances between all residues
    - dist_th
      maximum distance to be counted as interaction
    - shuffle: bool, optional\n
      if True data gets shuffled after every epoch\n
    - train: bool, optional\n
      if True Generator returns features and labels (use turing training) else only features\n
    """

    def __init__(self, features, labels, interaction_matrix, dim, n_channels, batch_size, first_ind,
                 hm_converted, hm_pos_vals, factor, hp_converted, hp_norm, cm_converted, ch_good_vals, ch_mid_vals,
                 ch_bad_vals, ia_converted, ia_norm, mat_index, cl_converted, cl_norm, dist_mat, dist_th, shuffle=True,
                 train=True):
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
        self.ch_good_vals = ch_good_vals
        self.ch_mid_vals = ch_mid_vals
        self.ch_bad_vals = ch_bad_vals
        self.ia_converted = ia_converted
        self.ia_norm = ia_norm
        self.mat_index = mat_index
        self.shuffle = shuffle
        self.train = train
        self.cl_converted = cl_converted
        self.cl_norm = cl_norm
        self.dist_mat = dist_mat
        self.dist_th = dist_th

    def __len__(self):
        return int(np.floor(len(self.features) / self.batch_size))  # floor

    def __getitem__(self, idx):
        features_batch = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

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
        batch_features = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_labels = np.empty(self.batch_size, dtype=float)

        for ci, i in enumerate(features_to_encode):
            # hydrogen bonging
            cur_hb = mutate_sequences(self.hm_converted, i, h_bonding, self.first_ind)
            part_hb = hbond_matrix(self.interaction_matrix, cur_hb, self.hm_pos_vals) * self.factor

            # hydrophobicity
            cur_hp = mutate_sequences(self.hp_converted, i, hydrophobicity, self.first_ind)
            part_hp = hydrophobicity_matrix(self.interaction_matrix, cur_hp, self.hp_norm) * self.factor

            # charge
            cur_cm = mutate_sequences(self.cm_converted, i, charge, self.first_ind)
            part_cm = charge_matrix(self.interaction_matrix, cur_cm, self.ch_good_vals, self.ch_mid_vals,
                                    self.ch_bad_vals) * self.factor
            # interaction area
            cur_ia = mutate_sequences(self.ia_converted, i, sasa, self.first_ind)
            part_ia = interaction_area(self.interaction_matrix, self.ia_converted, cur_ia,
                                       self.ia_norm) * self.factor
            # clashes
            cur_cl = mutate_sequences(self.cl_converted, i, side_chain_length, self.first_ind)
            part_cl = clashes(self.interaction_matrix, self.cl_converted, cur_cl, self.cl_norm, self.dist_mat,
                              dist_thr=self.dist_th) * self.factor

            batch_features[ci] = np.stack((part_hb, part_hp, part_cm, part_ia, part_cl, self.mat_index * self.factor),
                                          axis=2)
            batch_labels[ci] = corresponding_labels[ci]
        return batch_features, batch_labels


def data_generator_vals(wt_seq):
    """returns values/ numpy arrays based on the wt_seq for the DataGenerator
        :parameter
            wt_seq: wild type sequence as list eg ['A', 'V', 'L']\n
        :returns
            - hm_pos_vals: ndarray of int\n
              values for interactions with valid hydrogen bonding partners\n
            - ch_good_vals: ndarray of float\n
              values representing +/- charge pairs\n
            - ch_mid_vals: ndarray of float\n
              values representing +/n or -/n charge pairs\n
            - ch_bad_vals: ndarray of float\n
              values representing +/+ or -/- charge pairs\n
            - hp_norm: float\n
              max value possible for hydrophobicity interactions\n
            - ia_norm: float\n
              max value possible for interaction ares interactions\n
            - hm_converted: ndarray of float\n
              wt_seq converted into hydrogen bonding values
            - hp_converted: ndarray of float\n
              wt_seq converted into hydrophobicity values
            - cm_converted: ndarray of float\n
              wt_seq converted into charge values
            - ia_converted: ndarray of float\n
              wt_seq converted into SASA values
            - mat_index: 2D ndarray of float\n
              symmetrical index matrix\n
            - cl_converted: ndarray of float\n
              wt_seq converted into side chain length values
            - cl_norm: float\n
              max value possible for interaction ares interactions\n"""
    hm_pos_vals = np.asarray([2, 3, 6, 9])

    ch_good_vals = np.asarray([-1., 4.])
    ch_mid_vals = np.asarray([-2., 2.])
    ch_bad_vals = np.asarray([1.])

    h_vals = list(hydrophobicity.values())
    hp_norm = np.abs(max(h_vals) - min(h_vals))
    ia_norm = max(list(sasa.values())) * 2
    cl_norm = 2 * max(side_chain_length.values())  # + dist_thr

    hm_converted = np.asarray(list(map(h_bonding.get, wt_seq)))
    hp_converted = np.asarray(list(map(hydrophobicity.get, wt_seq)))
    cm_converted = np.asarray(list(map(charge.get, wt_seq)))
    ia_converted = np.asarray(list(map(sasa.get, wt_seq)))
    cl_converted = np.asarray(list(map(side_chain_length.get, wt_seq)))

    wt_len = len(wt_seq)
    mat_size = wt_len * wt_len
    pre_mat_index = np.arange(mat_size).reshape(wt_len, wt_len) / (mat_size - 1)
    pre_mat_index = np.triu(pre_mat_index)
    mat_index = pre_mat_index + pre_mat_index.T - np.diag(np.diag(pre_mat_index))
    np.fill_diagonal(mat_index, 0)

    return hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
           cm_converted, ia_converted, mat_index, cl_converted, cl_norm


class SaveToFile(keras.callbacks.Callback):
    """writes training stats in a temp file
     ...
    Attributes:\n
    - features: str\n
      path where the temp.csv file should be saved
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        log_string = "{},{:0.4f},{:0.4f},{:0.4f},{:0.4f}".format(str(epoch), logs["loss"], logs["mae"],
                                                                 logs["val_loss"], logs["val_mae"])
        log_file(self.filepath, write_str=log_string, optional_header="epoch,loss,mae,val_loss,val_mae")


def run_all(architecture_name, model_to_use, optimizer, tsv_file, pdb_file, wt_seq, number_mutations, variants, score,
            dist_thr, channel_num, max_train_mutations, training_epochs, test_num, first_ind, r_seed=None,
            deploy_early_stop=True, es_monitor="val_loss", es_min_d=0.01, es_patience=20, es_mode="auto",
            es_restore_bw=True, load_trained_model=None, batch_size=64, save_fig=None, show_fig=False,
            write_to_log=True, silent=False, extensive_test=True, save_model=False, load_trained_weights=None,
            no_nan=True, settings_test=False, p_dir="", split_def=None, validate_training=False, lr=0.001,
            transfer_conv_weights=None, train_conv_layers=False, write_temp=False):
    """
    - architecture_name: str\n
      name of the architecture\n
    - model_to_use: function object\n
      function that returns the model\n
    - optimizer: Optimizer object
      optimizer to be used\n
    - tsv_file: str\n
      path to tsv file containing dms data of the protein of interest\n
    - pdb_file: str\n
      path to pdb file containing the structure\n
    - wt_seq: str\n
      wt sequence of the protein of interest\n
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
      int specifying maximum number of mutations per sequence to be used for training\n
      None to use all mutations for training\n
    - training_epochs: int\n
      number of epochs used for training the model\n
    - test_num: int\n
      number of samples for the test after the model was trained\n
    - first_ind: int\n
      offset of the start of the sequence (when sequence doesn't start with residue 0)
    - r_seed: None, int, optional\n
      numpy random seed\n
    - deploy_early_stop: bool, optional\n
      whether early stop during training should be enabled (Ture) or not (False)
            - es_monitor: str, optional\n
              what to monitor to determine whether to stop the training or not\n
            - es_min_d: float, optional,
              min_delta min difference in es_monitor to not stop training\n
            - es_patience: int, optional\n
              number of epochs the model can try to get a es_monitor > es_min_d before stopping\n
            - es_mode: str, optional\n
              direction of quantity monitored in es_monitor\n
            - es_restore_bw: bool, optional\n
              True stores the best weights of the training - False stores the last\n
    - batch_size: int, optional\n
      after how many samples the gradient gets updated\n
    - load_trained_model: str or None, optional\n
      path to an already trained model\n
    - save_fig: bool, optional\n
      True to save figures in result_path\n
    - show_fig: bool, optional\n
      True to show figures\n
    - write_to_log: bool, optional\n
      if True writes all parameters used in the log file - !should be always enabled!\n
    - silent: bool, optional\n
      True to print stats in the terminal\n
    - save_model: bool, optional\n
      Ture to saves the model\n
    - load_trained_weights: str or None, optional\n
      path to model of who's weights should be used None if it shouldn't be used\n
    - no_nan: bool, optional\n
      True terminates training on nan\n
    - settings_test: bool, optional\n
      Ture doesn't train the model and only executes everything of the function that is before fit
    - p_dir: str, optional\n
      path to where the results, figures and log_file should be saved
    - split_def: list of int/float or None, optional\n
      specifies the split for train, tune, test indices
            - float specifies fractions of the whole dataset
              eg [0.25, 0.25, 0.5] creates a train and tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            - int specifies the different number of samples per dataset
              eg [50,50,100] leads to a train and a tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
    - validate_training: bool, optional\n
      if True validation of the training will be performed
    - lr: float, optional\n
      learning rate (how much the weights can change during an update)
    - transfer_conv_weights: str or None, optional\n
      path to model who's weights of it's convolution layers should be used for transfer learning (needs to have the
      same architecture for the convolution part as the newly build model (model_to_use)\n
    - train_conv_layers: bool, optional\n
      if True convolution layers are trainable - only applies whe transfer_conv_weights is not None\n
    - write_temp bool, optional\n
      if True writes mae and loss of each epoch to the temp.csv in result_files
    """
    tf.keras.backend.clear_session()
    if not write_to_log:
        warnings.warn("Write to log file disabled - not recommend behavior", UserWarning)
    # dictionary with argument names as keys and the input as values
    arg_dict = locals()

    if r_seed is not None:
        np.random.seed(r_seed)

    # clear temp file from previous content and create result_files folder if it doesn't exist already
    temp_path = p_dir + "/temp.csv"
    try:
        temp_stats = open(temp_path, "w+")
    except FileNotFoundError:
        create_folder(parent_dir="/".join(p_dir.split("/")[:-1]), dir_name="result_files")
        temp_stats = open(temp_path, "w+")
    temp_stats.close()

    starting_time = timer()
    wt_seq = list(wt_seq)
    # getting the proteins name
    if "/" in tsv_file:
        p_name = tsv_file.strip().split("/")[-1].split(".")[0]
    else:
        p_name = tsv_file.strip().split(".")[0]

    # creating a "unique" name for protein
    time_ = str(datetime.now().strftime("%d_%m_%Y_%H%M%S")).split(" ")[0]
    name = "{}_{}".format(p_name, time_)
    print(name)

    # creates a directory where plots will be saved
    log_file_path = p_dir + "/log_file.csv"
    if save_fig is not None:
        try:
            result_path = create_folder(p_dir, name)
        except FileExistsError:
            result_path = create_folder(p_dir, name, "_1")
        if save_fig is not None:
            save_fig = result_path

    if not settings_test:
        # writes used arguments to log file
        if write_to_log:
            header = "name," + ",".join(list(arg_dict.keys())) + ",training_time_in_min"
            prep_values = []
            for i in list(arg_dict.values()):
                if type(i) == list:
                    try:
                        prep_values += ["".join(i)]
                    except TypeError:
                        prep_values += ["".join(str(i)).replace(",", "_").replace(" ", "")]
                else:
                    prep_values += [str(i)]
            values = name + "," + ",".join(prep_values) + ",nan"
            log_file(log_file_path, values, header)

    # split dataset
    ind_dict, data_dict = split_inds(file_path=tsv_file, variants=variants, score=score,
                                     number_mutations=number_mutations, split=split_def)

    train_data = data_dict["train_data"]
    train_labels = data_dict["train_labels"]
    train_mutations = data_dict["train_mutations"]

    test_data = data_dict["tune_data"]
    test_labels = data_dict["tune_labels"]
    test_mutations = data_dict["tune_mutations"]

    unseen_data = data_dict["test_data"]
    unseen_labels = data_dict["test_labels"]
    unseen_mutations = data_dict["test_mutations"]

    # possible values and encoded wt_seq (based on different properties) for the DataGenerator
    hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
    cm_converted, ia_converted, mat_index, cl_converted, cl_norm = data_generator_vals(wt_seq)

    # distance-, factor- and interaction matrix
    dist_m, factor, comb_bool = atom_interaction_matrix_d(pdb_file, dist_th=dist_thr, plot_matrices=show_fig)

    # checks whether sequence in the pdb and the wt_seq match
    check_structure(pdb_file, comb_bool, wt_seq)

    # neural network model function
    model = model_to_use(wt_seq, channel_num)

    # load weights to model
    if load_trained_weights is not None:
        old_model = keras.models.load_model(load_trained_weights)
        model.set_weights(old_model.get_weights())

    # loads a model defined in load_trained_model
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
                if any(["dense" in l_name, "flatten" in l_name]):
                    pass
                else:
                    transfer_layers += [i]

        # Transfer weights to new model
        for i in transfer_layers:
            model.layers[i].set_weights(temp_weights[i])
            if train_conv_layers is False:
                model.layers[i].trainable = False

        model.summary()

    model.compile(optimizer(learning_rate=lr), loss="mean_absolute_error", metrics=["mae"])

    all_callbacks = []
    # deploying early stop parameters
    if deploy_early_stop:
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            min_delta=es_min_d,
            patience=es_patience,
            mode=es_mode,
            restore_best_weights=es_restore_bw)
        all_callbacks += [es_callback]

    # stops training on nan
    if no_nan:
        all_callbacks += [tf.keras.callbacks.TerminateOnNaN()]

    # save mae and loss to temp file
    if write_temp:
        all_callbacks += [SaveToFile(temp_path)]

    # parameters for the DatGenerator
    params = {'interaction_matrix': comb_bool,
              'dim': comb_bool.shape,
              'n_channels': channel_num,
              'batch_size': batch_size,
              'first_ind': first_ind,
              'hm_converted': hm_converted,
              'hm_pos_vals': hm_pos_vals,
              'factor': factor,
              'hp_converted': hp_converted,
              'hp_norm': hp_norm,
              'cm_converted': cm_converted,
              'ch_good_vals': ch_good_vals,
              'ch_mid_vals': ch_mid_vals,
              'ch_bad_vals': ch_bad_vals,
              'ia_converted': ia_converted,
              'ia_norm': ia_norm,
              'mat_index': mat_index,
              'cl_converted': cl_converted,
              'cl_norm': cl_norm,
              'dist_mat': dist_m,
              'dist_th': dist_thr,
              'shuffle': True,
              'train': True}

    # DatGenerator for training and the validation during training
    training_generator = DataGenerator(train_data, train_labels, **params)
    validation_generator = DataGenerator(test_data, test_labels, **params)

    if len(unseen_mutations) > 0:
        if test_num > len(unseen_data):
            test_num = len(unseen_data)
        pos_test_inds = np.arange(len(unseen_data))
        test_inds = np.random.choice(pos_test_inds, size=test_num, replace=False)
        t_data = unseen_data[test_inds]
        t_labels = unseen_labels[test_inds]
        t_mutations = unseen_mutations[test_inds]
    else:
        if test_num > len(test_data):
            test_num = len(test_data)
        pos_test_inds = np.arange(len(test_data))
        test_inds = np.random.choice(pos_test_inds, size=test_num, replace=False)
        t_data = test_data[test_inds]
        t_labels = test_labels[test_inds]
        t_mutations = test_mutations[test_inds]

    test_params = {'interaction_matrix': comb_bool,
                   'dim': comb_bool.shape,
                   'n_channels': channel_num,
                   'batch_size': 1,
                   'first_ind': first_ind,
                   'hm_converted': hm_converted,
                   'hm_pos_vals': hm_pos_vals,
                   'factor': factor,
                   'hp_converted': hp_converted,
                   'hp_norm': hp_norm,
                   'cm_converted': cm_converted,
                   'ch_good_vals': ch_good_vals,
                   'ch_mid_vals': ch_mid_vals,
                   'ch_bad_vals': ch_bad_vals,
                   'ia_converted': ia_converted,
                   'ia_norm': ia_norm,
                   'mat_index': mat_index,
                   'cl_converted': cl_converted,
                   'cl_norm': cl_norm,
                   'dist_mat': dist_m,
                   'dist_th': dist_thr,
                   'shuffle': False,
                   'train': False}

    test_generator = DataGenerator(t_data, np.zeros(len(t_labels)), **test_params)

    if not settings_test:
        # training
        history = model.fit(training_generator, validation_data=validation_generator, epochs=training_epochs,
                            use_multiprocessing=True, workers=12, callbacks=[all_callbacks])

        end_time = timer()

        # adds training time to result_files
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

        # --------------------------------------------------------------------------------------------
        """
        val_data = pd.read_csv("avgfp_augmentation_1/validate_avgfp.tsv", delimiter="\t")
        t_data = np.asarray(val_data[variants])
        t_labels = np.asarray(val_data[score])
        t_mutations = np.asarray(val_data[number_mutations])
        val_bool = []
        for i in t_data:
            val_bool += ["*" not in i]
        t_data = t_data[val_bool]
        t_labels = t_labels[val_bool]
        test_generator = DataGenerator(t_data, np.zeros(len(t_labels)), **test_params)
        predicted_labels = model.predict(test_generator).flatten()
        error = np.abs(predicted_labels - t_labels)

        
        order = np.lexsort((t_labels.astype(float), t_mutations[val_bool].astype(int)))
        plt.scatter(np.arange(len(t_labels)), predicted_labels[order],
                    color="yellowgreen", label="error", s=3)
        plt.plot(t_labels[order], color="firebrick")
        plt.show()
        

        try:
            pearson_r, pearson_r_p = scipy.stats.pearsonr(t_labels.astype(float), predicted_labels.astype(float))
            spearman_r, spearman_r_p = scipy.stats.spearmanr(t_labels.astype(float), predicted_labels.astype(float))
            print("MAE: {}\nSTD: {}\nPearson's r: {}\nPearson's r p-value:{}\nSpearman r: {}\nSpearman r p-value: {}\n".
                  format(str(np.mean(error)), str(error.std()), str(pearson_r), str(pearson_r_p), str(spearman_r),
                         str(spearman_r_p)))
        except ValueError:
            print("Invalid loss")
        """
        # --------------------------------------------------------------------------------------------

        # saves model in result path
        if save_model:
            try:
                result_path = create_folder(p_dir, name)
                model.save(result_path + "/" + name)
            except FileExistsError:
                model.save(result_path + "/" + name)

        # training and validation plot of the training
        if validate_training:
            try:
                val_val, _, _ = validate(validation_generator, model, history, name,
                                         max_train_mutations, save_fig_v=save_fig, plot_fig=show_fig)
            except ValueError:
                val_val = "nan"
                log_file(p_dir + "/log_file.csv", "nan in training history")
        else:
            val_val = "nan"
        # calculating pearsons' r and spearman r for the test dataset
        try:
            mae, pearsonr, pp, spearmanr, sp = pearson_spearman(model, test_generator, t_labels)
        except ValueError:
            mae, pearsonr, pp, spearmanr, sp = "nan", "nan", "nan", "nan", "nan"

        # creating more detailed plots
        if extensive_test:
            validation(model, test_generator, t_labels, t_mutations, val_val, p_name, max_train_mutations, test_num,
                       save_fig=save_fig, plot_fig=show_fig, silent=silent)

        # writing results to the result file
        result_string = ",".join([name, architecture_name, str(len(train_data)), str(len(test_data)), str(mae),
                                  str(pearsonr), str(pp), str(spearmanr), str(sp)])

        log_file(p_dir + "/results.csv", result_string,
                 "name,architecture,train_data_size,test_data_size,mae,pearson_r,pearson_p,spearman_r,spearman_p")


if __name__ == "__main__":
    pass