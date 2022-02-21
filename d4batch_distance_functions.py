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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=sys.maxsize)


def read_and_process(path_to_file, variants, silent=True, remove_astrix=True):
    """reads in the deep mutational scanning file and returns its data\n
        input:
            path_to_file: path to the tsv file\n
            variants: how the variants column in the file ins named\n
            silent: if True doesn't print any stats/ information\n
            remove_astrix: if True excludes nonsense mutations from the file\n
        :returns
            p_name: name of the protein\n
            raw_data: data as pandas df\n"""
    raw_data = pd.read_csv(path_to_file, delimiter="\t")
    # getting the proteins name
    if "/" in path_to_file:
        p_name = path_to_file.strip().split("/")[-1].split(".")[0]
    else:
        p_name = path_to_file.strip().split(".")[0]
    # some information about the file
    if not silent:
        print(" - ".join(raw_data.columns.tolist()))
        print()
        sc = 0
        for i in list(raw_data):
            if "score" in i:
                sc += 1
        if sc > 1:
            print("*** multiple scores are given - choose appropriate one ***")
        print("raw data length:", len(raw_data))
        print("protein name:", p_name)
    # checking each variant whether there is a mutation with an "*" and removing it
    if remove_astrix:
        rd_bool = []
        for i in raw_data[variants]:
            rd_bool += ["*" not in i]
        raw_data = raw_data[rd_bool]
        if not silent:
            print("*** {} rows had to be excluded due to incompatibility ***".format(np.sum(np.invert(rd_bool))))
    return p_name, raw_data


def check_seq(raw_data_cs, name_cs, wt_seq_cs, variants_cs, save_fig=None, plot_fig=False, silent=True):
    """checks whether the wild sequence and the sequence reconstructed from the raw data match and plot a histogram
        which shows how often a certain residue was part of a mutation\n
        input:
            raw_data_cs: dms data as pd dataframe\n
            name_cs: protein name\n
            wt_seq_cs: wild type sequence as list eg ['A', 'V', 'L']\n
            variants_cs: name of the variants column in the raw data file\n
            save_fig: None doesn't safe the histogram anything else does\n
            plot_fig: if True shows the histogram\n
            silent: if True doesn't show stats during execution\n
        :return
            first_ind: starting int of the sequence\n
            """
    # all variants in a list
    v = raw_data_cs[variants_cs].tolist()
    # list of lists with original amino acid and its residue index in the sequence
    # from all listed variations
    pre_seq = []
    for i in v:
        vi = i.strip().split(",")
        for j in vi:
            pre_seq += [[j[0], j[1:-1]]]

    pro_seq = np.unique(pre_seq, axis=0)
    # list of lists with original amino acid and its residue index in the sequence
    # only unique entries = reconstructed sequence
    pro_seq_sorted = pro_seq[np.argsort(pro_seq[:, 1].astype(int))]

    # checking the indexing of the sequence
    first_ind = int(pro_seq_sorted[0][1])

    if first_ind != 1:
        if not silent:
            print("*** {} used as start of the sequence indexing in the mutation file ***".format(str(first_ind)))
            print()

    # checking whether the reconstructed sequence is the same as the wt sequence
    pro_seq_inds = pro_seq_sorted[:, 1].astype(int)
    gap_count = 0
    gaps = []
    for i in range(first_ind, len(pro_seq_inds) + first_ind):
        if i < len(pro_seq_inds) - 1:
            if pro_seq_inds[i + 1] - pro_seq_inds[i] > 1:
                gap_count += pro_seq_inds[i + 1] - pro_seq_inds[i] - 1
                gaps += [np.arange(pro_seq_inds[i] - first_ind + 1, pro_seq_inds[i + 1] - first_ind)]
                if not silent:
                    print("*** residues between {} and {} not mutated***".format(str(pro_seq_inds[i]),
                                                                                 str(pro_seq_inds[i + 1])))

    if gap_count != len(wt_seq_cs) - len(pro_seq_sorted):
        if not silent:
            print("    Sequence constructed from mutations:\n   ", "".join(pro_seq_sorted[:, 0]))
        raise ValueError("Wild type sequence doesn't match the sequence reconstructed from the mutation file")
    elif gap_count > 0:
        fill = pro_seq_inds.copy().astype(object)
        offset = 0
        for i in gaps:
            for j in i:
                fill = np.insert(fill, j - offset, "_")
                offset += 1
        print("*** sequence check passed ***")

        under_fill = []
        rec_seq = []
        for i in fill:
            if i == "_":
                rec_seq += ["-"]
                under_fill += ["*"]
            else:
                rec_seq += [wt_seq_cs[int(i) - 1]]
                under_fill += [" "]

        r_seq_str = "".join(rec_seq)
        w_seq_str = "".join(wt_seq_cs)
        uf = "".join(under_fill)
        if not silent:
            print("reconstructed sequence\nwild type sequence\ngap indicator\n")
            for i in range(0, len(wt_seq_cs), 80):
                print(r_seq_str[i:i + 80])
                print(w_seq_str[i:i + 80])
                print(uf[i:i + 80])
                print()
    else:
        print("*** sequence check passed ***")

    if save_fig is not None:
        # histogram for how often a residue site was part of a mutation
        fig = plt.figure(figsize=(10, 6))
        plt.hist(x=np.asarray(pre_seq)[:, 1].astype(int), bins=np.arange(first_ind, len(pro_seq) + 1 + first_ind),
                 color="forestgreen")
        plt.xlabel("residue index")
        plt.ylabel("number of mutations")
        plt.title(name_cs)
        plt.savefig(save_fig + "/" + "mutation_histogram_" + name_cs)
        if plot_fig:
            plt.show()
    return first_ind


def split_data(raw_data_sd, variants_sd, score_sd, number_mutations_sd, max_train_mutations, train_split, r_seed,
               silent=False):
    """splits data from raw_data_sd into training and test dataset\n
        input:
            raw_data_sd: dms data as pandas dataframe\n
            variants_sd: name of the variants column in raw_data_sd\n
            score_sd: name of the score column in raw_data_sd\n
            number_mutations_sd: name of the column that stats the number of mutations per variant in raw_data_sd \n
            max_train_mutations: maximum number of mutations per sequence to be used for training\n
                (None to use all mutations for training) variants with mutations > max_train_mutations get stored in
                unseen_data\n
            train_split: how much of the dataset should be used as training data (int to specify a number of data for the
                training dataset or a float (<=1) to specify the fraction of the dataset used as training data\n
            r_seed: random seed for pandas random_state\n
            silent: if True doesn't show stats during execution\n
        :returns
            all numpy arrays\n
            data: variants\n
            labels: score\n
            mutations: number of mutations\n
           train_data, train_labels, train_mutations\n
           test_data, test_labels, test_mutations\n
           if max_train_mutations is used (variants with more mutations than max_train_mutations):
            unseen_data, unseen_labels, unseen_mutations\n """
    vas = raw_data_sd[[variants_sd, score_sd, number_mutations_sd]]

    if max_train_mutations is None:
        if isinstance(train_split, float):
            vas_train = vas.sample(frac=train_split, random_state=r_seed)
            vas_test = vas.drop(vas_train.index)
        else:
            vas_train = vas.sample(n=train_split, random_state=r_seed)
            vas_test = vas.drop(vas_train.index)
        vas_train = np.asarray(vas_train)
        vas_test = np.asarray(vas_test)

        train_data = vas_train[:, 0]
        train_labels = vas_train[:, 1]
        train_mutations = vas_train[:, 2]

        test_data = vas_test[:, 0]
        test_labels = vas_test[:, 1]
        test_mutations = vas_test[:, 2]
        if not silent:
            print("train data size:", len(train_data))
            print("test data size:", len(test_data))

        unseen_mutations = None
        unseen_labels = None
        unseen_data = None
    else:
        vs_un = vas[vas[number_mutations_sd] <= max_train_mutations]
        vs_up = vas[vas[number_mutations_sd] > max_train_mutations]
        if isinstance(train_split, float):
            vs_un_train = vs_un.sample(frac=train_split, random_state=r_seed)
            vs_un_test = vs_un.drop(vs_un_train.index)
        else:
            vs_un_train = vs_un.sample(n=train_split, random_state=r_seed)
            vs_un_test = vs_un.drop(vs_un_train.index)

        vs_un_train = np.asarray(vs_un_train)
        vs_un_test = np.asarray(vs_un_test)
        vs_up = np.asarray(vs_up)

        train_data = vs_un_train[:, 0]
        train_labels = vs_un_train[:, 1]
        train_mutations = vs_un_train[:, 2]

        test_data = vs_un_test[:, 0]
        test_labels = vs_un_test[:, 1]
        test_mutations = vs_un_test[:, 2]

        unseen_data = vs_up[:, 0]
        unseen_labels = vs_up[:, 1]
        unseen_mutations = vs_up[:, 2]

        if not silent:
            print("train data size:", len(train_data))
            print("test data size:", len(test_data))
            print("unseen data size:", len(unseen_data))
    return train_data, train_labels, train_mutations, test_data, test_labels, test_mutations, unseen_data, unseen_labels, unseen_mutations


def data_coord_extraction(target_pdb_file):
    """calculates distance between residues and builds artificial CB for GLY based on the
        side chains of amino acids (!= GLY) before if there is an or after it if Gly is the start amino acid\n
        No duplicated side chain entries allowed
            input:
                target_pdb_file: pdb file with data of protein of interest\n
            :returns:
                new_data: 2d list [[Atom type, Residue 3letter, ChainID, ResidueID],...]\n
                new_coords: 2d list of corresponding coordinates to the new_data entries\n
                split_tuples: list of tuples which indicate the start and end of a residue in the new_data and
                new_coords lists\n"""
    # list of all data of the entries like [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data += [[line[12:16].replace(" ", "").strip(), line[17:20].replace(" ", "").strip(),
                          line[21].replace(" ", "").strip(), line[22:26].replace(" ", "").strip()]]
            res_coords += [[line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)

    def art_cb(inc_bool):
        """gets the CA and CB coordinates of the residue at inc_bool=True,computed the difference in the CA atom
            coordinates of this residue and the Gly and uses this difference to compute the 'artificial CB' for the Gly
            if entry for CB is duplicated for an amino acid the mean of its coordinates are used
            input:
                increase: residue id of the closest amino acid with a CB
            :return
                art_cbc: CA, CB coordinates as [[xa, ya, za]], [[xb, yb, zb]]"""
        # data and coords of the next amino acid != GLY
        increased_data = res_data[inc_bool]
        increased_coords = res_coords[inc_bool]
        # CB and CA coordinates of the next amino acid != GLY
        next_cb = increased_coords[increased_data[:, 0] == "CB"]
        next_ca = increased_coords[increased_data[:, 0] == "CA"]
        # CA coords of the GLY
        true_cac = i_coords[i_data[:, 0] == "CA"]
        # difference in CA coordinates to compute the artificial CB for the GLY
        delta = next_ca - true_cac
        art_cbc = next_cb - delta
        # print("pseudoatom tmpPoint2, resi=40, chain=ZZ, b=40, color=red, pos=", art_cbc[0].tolist())
        return art_cbc

    split_tuples = []
    last_ind = 0
    new_coords = []
    new_data = []
    # RES, CHAIN, ResNR sorted
    residues = np.unique(res_data[:, 1:], axis=0)
    residues = residues[np.lexsort((residues[:, 2].astype(int), residues[:, 1]))]
    for ci, i in enumerate(residues):
        i_bool = np.all(res_data[:, 1:] == i, axis=1)
        i_data = res_data[i_bool]
        i_coords = res_coords[i_bool]
        if i_data[0][1] == "GLY":
            # if GLY is the first amino acid or all residues so far were GLY
            if len(new_data) == 0 or (np.all(residues[:ci, 0] == "GLY") and len(residues[:ci]) > 0):
                # to look at next amino acid(s)
                i_increase = 1
                sign = 1
            else:
                # to look at previous amino acid(s)
                i_increase = -1
                sign = -1
            # get the index of the next amino acid that is no Gly
            while residues[ci + i_increase][0] == "GLY":
                i_increase += 1 * sign
            # where this data is located as boolean list
            increase_bool = np.all(res_data[:, 1:] == residues[ci + i_increase], axis=1)
            # artificial CB coordinates of Gly
            cb_coords = art_cb(increase_bool)
            new_i_coords = np.append(i_coords, cb_coords, axis=0)

            data_inter = i_data[0].copy()
            # artificial CB entry
            data_inter[0] = "CB"
            new_i_data = np.append(i_data, np.asarray([data_inter]), axis=0)
            # new_i_data[:, 1] = "ALA"
            new_coords += new_i_coords.tolist()
            new_data += new_i_data.tolist()
            # length of the
            length = len(new_i_data)
            # start and end of the residue entries
            split_tuples += [(last_ind, last_ind + length)]
            last_ind += length
        else:
            new_coords += i_coords.tolist()
            new_data += i_data.tolist()
            length = len(i_data)
            split_tuples += [(last_ind, last_ind + length)]
            last_ind += length
    return np.asarray(new_data), np.asarray(new_coords, dtype=float), split_tuples


def dist_calc(arr1, arr2):
    """calculates distance between arr1 and arr2 and returns a 2D array with all distances of all arr1 points
        against all arr2 points\n
        input:
            arr1, arr2: 2D arrays of 1D lists with 3D coordinates eg [[x1, y1, z1],...]\n
        :return
            dist: distance matrix between arr1 and arr2\n"""
    # get only the x,y,z coordinates from the input arrays and reshape them so they can be subtracted from each other
    arr1_coords_rs = arr1.reshape(arr1.shape[0], 1, 3)
    arr2_coord_rs = arr2.reshape(1, arr2.shape[0], 3)
    # calculating the distance between each point and returning a 2D array with all distances
    dist = np.sqrt(((arr1_coords_rs - arr2_coord_rs) ** 2).sum(axis=2))
    return dist


def atom_interaction_matrix_d(path_to_pdb_file, dist_th=10, plot_matrices=False):
    """computes the adjacency matrix for a given pdb file based on the closest side chain atoms\n
            input:
                path_to_pdb_file: pdb file of the protein of interest\n
                dist_th: maximum distance in A of atoms of two residues to be seen as interacting\n
                plot_matrices: if True plots matrices for\n
                    from left to right:\n
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
    data, coords, _ = data_coord_extraction(path_to_pdb_file)
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


def check_structure(path_to_pdb_file, comb_bool_cs, wt_seq_cs):
    """checks whether the given wild type sequence matches the sequence in the pdb file\n
        input:
            path_to_pdb_file: path to used pdb file\n
            comb_bool_cs: interacting_residues of atom_interaction_matrix_d\n
            wt_seq_cs: wild type sequence as list eg ['A', 'V', 'L']\n
        :return
            None
        """
    if len(comb_bool_cs) != len(wt_seq_cs):
        raise ValueError("Wild type sequence doesn't match the sequence in the pdb file (check for multimers)\n")
    else:
        # could be read only one time (additional input for atom_interaction_matrix(_d))
        pdb_seq = np.unique(data_coord_extraction(path_to_pdb_file)[0][:, 1:], axis=0)
        pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
        # sequence derived from pdb file
        pdb_seq_ol_list = np.asarray(list(map(aa_dict.get, pdb_seq_sorted[:, 0])))
        if not np.all(np.asarray(wt_seq_cs) == pdb_seq_ol_list):
            raise ValueError("Wild type sequence doesn't match the sequence derived from the pdb file\n")
        else:
            print("*** structure check passed ***")


class DataGenerator(keras.utils.Sequence):
    """
    Generates n_channel x n x n matrices to feed them as batches to a network\n
    ...
    Attributes:\n
    - features: features that should be encoded\n
    - labels: the corresponding labels to the features\n
    - interaction_matrix: boolean matrix whether residues interact or not\n
    - wild_type_seq: wild type sequence as str\n
    - dim: dimensions of the matrices (len(wt_seq) x len(wt_seq))\n
    - n_channels: number of matrices used\n
    - batch_size: Batch size (if 1 gradient gets updated after every sample in training)\n
    - first_ind: index of the start of the sequence\n
    - hm_converted: wt sequence h-bonding encoded\n
    - hm_pos_vals: valid values for h-bonding residues\n
    - factor: 1 - norm(distance) for all residues in the interaction matrix\n
    - hp_converted: wt sequence hydrophobicity encoded\n
    - hp_norm: max possible value for hydrophobicity change\n
    - cm_converted: wt sequence charge encoded\n
    - ch_good_vals: values for +-, nn interactions\n
    - ch_mid_vals: values for n+, n- interactions\n
    - ch_bad_vals: values for --, ++ interactions\n
    - ia_converted: wt sequence interaction area encoded\n
    - ia_norm: max value for interaction area change\n
    - mat_index: index matrix for adjacency matrix\n
    - shuffle: if True data gets shuffled after every epoch\n
    - train: set True if used for training\n
    """

    def __init__(self, features, labels, interaction_matrix, wild_type_seq, dim, n_channels, batch_size, first_ind,
                 hm_converted, hm_pos_vals, factor, hp_converted, hp_norm, cm_converted, ch_good_vals, ch_mid_vals,
                 ch_bad_vals, ia_converted, ia_norm, mat_index, shuffle=True, train=True):
        self.features, self.labels = features, labels
        self.interaction_matrix = interaction_matrix
        self.wild_type_seq = wild_type_seq
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

    def __len__(self):
        return int(np.ceil(len(self.features) / self.batch_size))

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

    def __mutate_sequences(self, wt_sequence, mutations, f_dict):
        """mutates the wild type sequence at positions defined in mutations and
            returns the mutated sequences
            input:
                wt_sequence: the encoded wild type sequence as list e.g.
                    [0.3, 0.8, 0.1, 1.]
                mutations: str where the mutations take place e.g. 'F1K,K2G'
                f_dict: dictionary with values for encoding
            return:
                mutated_sequences: mutated sequences as list"""
        a_to_mut = wt_sequence.copy()
        muts = mutations.strip().split(",")
        for j in muts:
            j = j.strip()
            a_to_mut[int(j[1:-1]) - self.first_ind] = f_dict[j[-1]]
        return a_to_mut

    def __hydrophobicity_matrix(self, res_bool_matrix, converted, norm):
        """matrix that represents how similar its pairs are in terms of hydrophobicity
            only for pairs that are true in res_bool_matrix
            input:
                res_bool_matrix: matrix (len(wt_seq) x len(wt_seq)) where pairs that obey distance and angle criteria
                 are True
                converted: the sequence converted to the values of the corresponding dict as 1D numpy array
                norm: max value possible for interactions between two residues
            :return
                hp_matrix: len(wt_seq) x len(wt_seq) matrix with the corresponding normalized similarity in terms of
                hydrophobicity of each pair"""
        interactions = np.abs(converted - converted.reshape(len(converted), -1))
        hp_matrix = 1 - (interactions / norm)
        hp_matrix[np.invert(res_bool_matrix)] = 0
        return hp_matrix

    def __hbond_matrix(self, res_bool_matrix, converted, valid_vals):
        """matrix that represents whether pairs can form h bonds (True) or not (False)
            only for pairs that are true in res_bool_matrix
            input:
                res_bool_matrix: matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True
                valid_vals: which values of the matrix are True after multiplying the encoded sequence against itself
                converted: the sequence converted to the values of the corresponding dict as 1D numpy array
            :return
                hb_mat: len(wt_seq) x len(wt_seq) matrix where pairs that can form h bonds are True"""
        interactions = converted * converted.reshape(len(converted), -1)
        hb_matrix = np.isin(interactions, valid_vals)
        hb_mat = np.all(np.stack((hb_matrix, res_bool_matrix)), axis=0)
        return hb_mat

    def __charge_matrix(self, res_bool_matrix, converted, good, mid, bad):
        """matrix that represents whether pairs of amino acids are of the same charge (0), of opposite charge /
            both uncharged (1), or one charged one neutral (0.5) only for pairs that are true in res_bool_matrix
            input:
                res_bool_matrix: matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True
                invalid_val: value that marks wrong charge pairs (1 if neg -1,neu 0,pos 1 encoding)
                converted: the sequence converted to the values of the corresponding dict as 1D numpy array
            :return
                c_mat: len(wt_seq) x len(wt_seq) matrix where amino acid pairs which are of the opposite charge internal
                or both uncharged are True"""
        interactions = converted * converted.reshape(len(converted), -1)
        interactions[np.invert(res_bool_matrix)] = 0
        interactions[np.isin(interactions, bad)] = 0
        interactions[np.isin(interactions, mid)] = 0.5
        interactions[np.isin(interactions, good)] = 1
        return interactions

    def __interaction_area(self, res_bool_matrix, wt_converted, mut_converted, norm):
        """matrix that represents the change in solvent accessible surface area (SASA) due to a mutation
            only for pairs that are true in res_bool_matrix
            input:
                res_bool_matrix: matrix (len(wt_seq) x len(wt_seq)) with interacting pairs as True
                wt_converted: wild type sequence converted with the corresponding dict
                mut_converted: mutated sequence converted with the corresponding dict
                norm: max value possible for interactions between two residues
            :return
                ia_matrix: len(wt_seq) x len(wt_seq) matrix with values corresponding to the
                absolute magnitude of change in the SASA of a residue pair """
        d = wt_converted - mut_converted
        dd = np.abs(d + d.reshape(len(d), -1)) / norm
        dd[np.invert(res_bool_matrix)] = 0
        dd = 1 - dd
        return dd

    def __batch_variants(self, features_to_encode, corresponding_labels):
        """creates encoded variants for a batch"""
        batch_features = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_labels = np.empty(self.batch_size, dtype=float)

        for ci, i in enumerate(features_to_encode):
            # hydrogen donging
            cur_hb = self.__mutate_sequences(self.hm_converted, i, h_bonding)
            part_hb = self.__hbond_matrix(self.interaction_matrix, cur_hb, self.hm_pos_vals) * self.factor

            # hydrophobicity
            cur_hp = self.__mutate_sequences(self.hp_converted, i, hydrophobicity)
            part_hp = self.__hydrophobicity_matrix(self.interaction_matrix, cur_hp, self.hp_norm) * self.factor

            # charge
            cur_cm = self.__mutate_sequences(self.cm_converted, i, charge)
            part_cm = self.__charge_matrix(self.interaction_matrix, cur_cm, self.ch_good_vals, self.ch_mid_vals,
                                           self.ch_bad_vals) * self.factor
            # interaction area
            cur_ia = self.__mutate_sequences(self.ia_converted, i, sasa)
            part_ia = self.__interaction_area(self.interaction_matrix, self.ia_converted, cur_ia,
                                              self.ia_norm) * self.factor

            batch_features[ci] = np.stack((part_hb, part_hp, part_cm, part_ia, self.mat_index * self.factor), axis=2)
            batch_labels[ci] = corresponding_labels[ci]
        return batch_features, batch_labels


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
        input:
            generator: Data generator to predict values (not shuffled)\n
            labels: the corresponding labels for the generator\n
        return:
            pearson_r: Pearson’s correlation coefficient\n
            pearson_r_p: Two-tailed p-value\n
            spearman_r: Spearman correlation coefficient\n
            spearman_r_p: p-value for a hypothesis test whose null hypothesis is that two sets of data are
            uncorrelated\n
            """
    pred = model.predict(generator).flatten()
    ground_truth = labels
    pearson_r, pearson_r_p = scipy.stats.pearsonr(ground_truth.astype(float), pred.astype(float))
    spearman_r, spearman_r_p = scipy.stats.spearmanr(ground_truth.astype(float), pred.astype(float))
    return pearson_r, pearson_r_p, spearman_r, spearman_r_p


def validation(model, generator, labels, v_mutations, p_name, test_num, name,
               save_fig=None, plot_fig=False, silent=True):
    """plot validations\n
        input:
            generator: data generator for predicting values\n
            labels: the corresponding real labels to the generator\n
            v_mutations: number of mutations per data sample in the generator\n
            p_name: protein name
            test_num: number of samples used for the test, name,
               save_fig=None, plot_fig=False, silent=True
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
        plt.savefig(save_fig + "/" + "test_" + name)
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
        plt.savefig(save_fig + "/" + "boxplot_" + name)
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
        plt.savefig(save_fig + "/" + "correlation_" + name)
    if plot_fig:
        plt.show()


def log_file(file_path, write_str, optional_header=""):
    """reads and writes log info's to log file\n
        input:
            file_path: path to log life\n
            write_str: string that should be written to log file\n
            optional_header: optional header to indicate the column names\n"""
    try:
        log_file_read = open(file_path, "r")
        prev_log = log_file_read.readlines()
        log_file_read.close()
        if len(list(prev_log)) == 0:
            prev_log = optional_header + "\n"
    except FileNotFoundError:
        if len(optional_header) > 0:
            prev_log = optional_header + "\n"
        else:
            prev_log = optional_header
    log_file_write = open(file_path, "w+")
    for i in prev_log:
        log_file_write.write(i)
    log_file_write.write(write_str + "\n")
    log_file_write.close()


def create_folder(parent_dir, dir_name, add=""):
    """creates directory for current experiment\n
        input:
            parent_dir: path where the new directory should be created\n
            dir_name: name of the new directory\n
            add: add to the name of the new directory\n"""
    if "/" in dir_name:
        dir_name = dir_name.replace("/", "_")
    directory = dir_name + add
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    return path


def data_generator_vals(wt_seq):
    """returns values based on the wt_seq for the DataGenerator:
            - hm_pos_vals
            - ch_good_vals
            - ch_mid_vals
            - ch_bad_vals
            - hp_norm
            - ia_norm
            - hm_converted
            - hp_converted,
            cm_converted, ia_converted, mat_index\n
        input:
            wt_seq: wild type sequence as list eg ['A', 'V', 'L']\n"""
    hm_pos_vals = np.asarray([2, 3, 6, 9])

    ch_good_vals = np.asarray([-1., 4.])
    ch_mid_vals = np.asarray([-2., 2.])
    ch_bad_vals = np.asarray([1.])

    h_vals = list(hydrophobicity.values())
    hp_norm = np.abs(max(h_vals) - min(h_vals))
    ia_norm = max(list(sasa.values())) * 2

    hm_converted = np.asarray(list(map(h_bonding.get, wt_seq)))
    hp_converted = np.asarray(list(map(hydrophobicity.get, wt_seq)))
    cm_converted = np.asarray(list(map(charge.get, wt_seq)))
    ia_converted = np.asarray(list(map(sasa.get, wt_seq)))

    wt_len = len(wt_seq)
    mat_size = wt_len * wt_len
    pre_mat_index = np.arange(mat_size).reshape(wt_len, wt_len) / (mat_size - 1)
    pre_mat_index = np.triu(pre_mat_index)
    mat_index = pre_mat_index + pre_mat_index.T - np.diag(np.diag(pre_mat_index))
    np.fill_diagonal(mat_index, 0)

    return hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
           cm_converted, ia_converted, mat_index


def protein_settings(protein_name):
    """gets different setting for the protein of interest from the protein_settings file\n
        input:
            protein_name: name of the protein in the protein_settings file"""
    settings = pd.read_csv("datasets/protein_settings.txt", delimiter=",")
    protein_name = protein_name.lower()
    content = np.asarray(settings[settings["name"] == protein_name][["attribute", "value"]])
    protein_settings_dict = dict(zip(content[:, 0], content[:, 1]))
    return protein_settings_dict


def run_all(architecture_name, model_to_use, optimizer, tsv_file, pdb_file, wt_seq, number_mutations, variants, score,
            dist_thr, channel_num, max_train_mutations, train_split, training_epochs, test_num, r_seed=None,
            deploy_early_stop=True, es_monitor="val_loss", es_min_d=0.01, es_patience=20, es_mode="auto",
            es_restore_bw=True, load_model=None, batch_size=64, save_fig=None, show_fig=False,
            write_to_log=True, silent=False, extensive_test=True, save_model=False, load_weights=None, no_nan=True,
            settings_test=False):
    """
    - architecture_name: name of the architecture\n
    - model_to_use: function that returns the model\n
    - optimizer: Optimizer object to be used\n
    - tsv_file: tsv file containing mutations of the protein of interest\n
    - pdb_file: pdb file containing the structure\n
    - wt_seq: wt sequence as string of the protein of interest\n
    - number_mutations: how the number of mutations column is named\n
    - variants: name of the variant column\n
    - score: name of the score column\n
    - dist_thr: threshold distances between any side chain atom to count as interacting\n
    - channel_num: number of channels\n
    - max_train_mutations: maximum number of mutations per sequence to be used for training (None to use all mutations for
                        training)\n
    - train_split: how much of the data should be used for training (if float it's used as fraction of the data otherwise
                it's the number of samples to use as training data)\n
    - training_epochs: number of epochs used for training the model\n
    - test_num: number of samples for the test after the model was trained\n
    - r_seed: numpy random seed\n
    - deploy_early_stop: whether early stop during training should be enabled or not\n
        - es_monitor: what to monitor to determine whether to stop the training or not\n
        - es_min_d: min_delta min difference in es_monitor to not stop training\n
        - es_patience: number of epochs the model can try to get a es_monitor > es_min_d before stopping\n
        - es_mode: direction of quantity monitored in es_monitor\n
    - es_restore_bw: True stores the best weights of the training - False stores the last\n
    - batch_size: after how many samples the gradient gets updated\n
    - load_model: path to an already trained model\n
    - save_fig: whether the figures should be saved\n
    - show_fig: whether the figures should be shown\n
    - write_to_log: if True writes all parameters used in the log file - !should be always enabled!\n
    - silent: whether to print stats in the terminal\n
    - save_model: if Ture saves the model as h5 format\n
    - load_weights: path to model of who's weights should be used None if it shouldn't be used\n
    - no_nan: if True terminates training on nan\n
    - settings_test: if Ture doesn't train the model and only executes everything of the function that is before fit
    """

    if not write_to_log:
        warnings.warn("Write to log file disabled - not recommend behavior", UserWarning)
    # dictionary with argument names as keys and the input as values
    arg_dict = locals()
    starting_time = timer()
    wt_seq = list(wt_seq)
    # getting the raw data as well as the protein name from the tsv file
    p_name, raw_data = read_and_process(tsv_file, variants, silent=silent)
    # creating a "unique" name for protein
    time_ = str(datetime.now().strftime("%d_%m_%Y_%H%M")).split(" ")[0]
    name = "{}_{}".format(p_name, time_)

    # creates a directory where plots will be saved
    p_dir = "./result_files"
    # log_file_path = "result_files/log_file.csv"
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
                    prep_values += ["".join(i)]
                else:
                    prep_values += [str(i)]
            values = name + "," + ",".join(prep_values) + ",nan"
            log_file(log_file_path, values, header)

    # starting index of the protein sequence
    first_ind = check_seq(raw_data, p_name, wt_seq, variants, save_fig=save_fig, plot_fig=show_fig, silent=silent)
    print(first_ind)
    # split dataset
    train_data, train_labels, train_mutations, test_data, test_labels, test_mutations, unseen_data, unseen_labels, \
    unseen_mutations = split_data(raw_data, variants, score, number_mutations, max_train_mutations, train_split,
                                  r_seed, silent=silent)

    # possible values and encoded wt_seq (based on different properties) for the DataGenerator
    hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
    cm_converted, ia_converted, mat_index = data_generator_vals(wt_seq)

    # factor and interaction matrix
    _, factor, comb_bool = atom_interaction_matrix_d(pdb_file, dist_th=dist_thr, plot_matrices=show_fig)

    # checks whether sequence in the pdb and the wt_seq match
    check_structure(pdb_file, comb_bool, wt_seq)

    # neural network model function
    model = model_to_use(wt_seq, channel_num)

    # loads weight to model
    if load_weights is not None:
        old_model = keras.models.load_model(load_weights)
        model.set_weights(old_model.get_weights())
    model.compile(optimizer, loss="mean_absolute_error", metrics=["mae"])

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

    # parameters for the DatGenerator
    params = {'wild_type_seq': wt_seq,
              'interaction_matrix': comb_bool,
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
              'shuffle': True,
              'train': True}

    # DatGenerator for training and the validation during training
    training_generator = DataGenerator(train_data, train_labels, **params)
    validation_generator = DataGenerator(test_data, test_labels, **params)

    # if r_seed is not None:
    #     np.random.seed = r_seed

    if unseen_mutations is not None:
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

    test_params = {'wild_type_seq': wt_seq,
                   'interaction_matrix': comb_bool,
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
                   'shuffle': False,
                   'train': False}

    # test_generator = DataGenerator(t_data, t_labels, **test_params)

    if load_model is not None:
        model = keras.models.load_model(load_model)

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
                if log_cont_len - ci == 2:
                    loi = i.strip().split(",")
                    loi[-1] = str(np.round((end_time - starting_time) / 60, 0))
                    w_log.write(",".join(loi) + "\n")
                else:
                    w_log.write(i)
        w_log.close()

        # --------------------------------------------------------------------------------------------

        val_data = pd.read_csv("avgfp_augmentation_1/validate_avgfp.tsv", delimiter="\t")
        t_data = np.asarray(val_data[variants])
        t_labels = np.asarray(val_data[score])
        val_bool = []
        for i in t_data:
            val_bool += ["*" not in i]
        t_data = t_data[val_bool]
        t_labels = t_labels[val_bool]
        test_generator = DataGenerator(t_data, np.zeros(len(t_labels)), **test_params)
        predicted_labels = model.predict(test_generator).flatten()
        error = np.abs(predicted_labels - t_labels)
        try:
            pearson_r, pearson_r_p = scipy.stats.pearsonr(t_labels.astype(float), predicted_labels.astype(float))
            spearman_r, spearman_r_p = scipy.stats.spearmanr(t_labels.astype(float), predicted_labels.astype(float))
            print("MAE: {}\nSTD: {}\nPearson's r: {}\nPearson's r p-value:{}\nSpearman r: {}\nSpearman r p-value: {}\n".
                  format(str(np.mean(error)), str(error.std()), str(pearson_r), str(pearson_r_p), str(spearman_r),
                         str(spearman_r_p)))
        except ValueError:
            print("Invalid loss")

        # --------------------------------------------------------------------------------------------
        # saves model in result path
        if save_model:
            try:
                result_path = create_folder(p_dir, name)
                model.save(result_path + "/" + name)
            except FileExistsError:
                model.save(result_path + "/" + name)

        # training and validation plot of the training
        try:
            val_val, epochs_bw, test_loss = validate(validation_generator, model, history, name, max_train_mutations,
                                                     save_fig_v=save_fig, plot_fig=show_fig)
        except ValueError:
            val_val = "nan"
            epochs_bw = 0
            test_loss = "nan"
            log_file("result_files/log_file.csv", "nan in training history")

        # calculating pearsons' r and spearman r for the test dataset
        try:
            pearsonr, pp, spearmanr, sp = pearson_spearman(model, test_generator, t_labels)
        except ValueError:
            pearsonr, pp, spearmanr, sp = "nan", "nan", "nan", "nan"

        # creating more detailed plots
        if extensive_test:
            validation(model, test_generator, t_labels, t_mutations, val_val, p_name, max_train_mutations, test_num,
                       name, save_fig=save_fig, plot_fig=show_fig, silent=silent)

        # writing results to the result file
        result_string = ",".join([name, architecture_name, str(len(train_data)), str(len(test_data)), str(test_loss),
                                  str(epochs_bw), str(pearsonr), str(pp), str(spearmanr), str(sp)])

        log_file("result_files/results.csv", result_string, "name,train_data_size,test_data_size,mae,"
                                                            "epochs_best_weight,pearson_r,pearson_p,"
                                                            "spearman_r,spearman_p")


aa_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
           "VAL": "V", "LEU": "L", "ILE": "I", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "TRP": "W",
           "TYR": "Y", "THR": "T"}

hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9,
                  'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                  'W': -0.9, 'Y': -1.3}

# neutral 0, negatively charged -1, positively charged 1
charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
          'Q': 0, 'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}

# hydrogen bonding capability 0 no hydrogen bonding, 1 acceptor, 2 donor, 3 donor and acceptor
h_bonding = {'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 3, 'I': 0, 'K': 2, 'L': 0, 'M': 0, 'N': 3, 'P': 0,
             'Q': 3, 'R': 2, 'S': 3, 'T': 3, 'V': 0, 'W': 2, 'Y': 3}

# surface accessible side chain area
sasa = {'A': 75, 'C': 115, 'D': 130, 'E': 161, 'F': 209, 'G': 0, 'H': 180, 'I': 172, 'K': 205, 'L': 172, 'M': 184,
        'N': 142, 'P': 134, 'Q': 173, 'R': 236, 'S': 95, 'T': 130, 'V': 143, 'W': 254, 'Y': 222}

if __name__ == "__main__":
    pass
