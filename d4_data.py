import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from d4_utils import aa_dict


def read_and_process(path_to_file, variants, silent=True, remove_astrix=True):
    """reads in the deep mutational scanning file and returns its data\n
        :parameter
            path_to_file: str\n
            path to the tsv file\n
            variants: str\n
            how the variants' column in the file ins named\n
            silent: bool, optional\n
            if True doesn't print any stats/ information\n
            remove_astrix: bool, optional\n
            if True excludes nonsense mutations from the file\n
        :returns
            p_name: str\n
            name of the protein\n
            raw_data: pd.Dataframe\n
            data as pandas df\n"""
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
        :parameter
            raw_data_cs: pd dataframe\n
            dms data\n
            name_cs: str\n
            protein name\n
            wt_seq_cs: list\n
            wild type sequence as list eg ['A', 'V', 'L']\n
            variants_cs: str\n
            name of the variants' column in the raw data file\n
            save_fig: any, optional\n
            None doesn't safe the histogram anything else does\n
            plot_fig: bool, optional\n
            if True shows the histogram\n
            silent:  bool, optional\n
            if True doesn't show stats during execution\n
        :return
            first_ind: int\n
            starting int of the sequence\n
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
    print(pro_seq_sorted)
    # checking the indexing of the sequence
    first_ind = int(pro_seq_sorted[0][1])
    print(first_ind)
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
        :parameter
            raw_data_sd: pd dataframe\n
            dms data\n
            variants_sd: str\n
            name of the variants column in raw_data_sd\n
            score_sd: str\n
            name of the score column in raw_data_sd\n
            number_mutations_sd: str\n
            name of the column that stats the number of mutations per variant in raw_data_sd \n
            max_train_mutations: int on None
                - maximum number of mutations per sequence to be used for training\n
                - None: to use all mutations for training \n
                - int: variants with mutations > max_train_mutations get stored in unseen_data\n
            train_split: int or float
                how much of the dataset should be used as training data (int to specify a number of data for the
                training dataset or a float (<=1) to specify the fraction of the dataset used as training data\n
            r_seed: int\n
            random seed for pandas random_state\n
            silent: bool, optional\n
            if True doesn't show stats during execution\n
        :returns
            data: ndarray\n
            variants\n
            labels: ndarray\n
            score for each variant\n
            mutations: ndarray\n
            number of mutations for each variant\n
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


# ---------------------------------------------- NOW USED --------------------------------------------------------------
def split_inds(file_path, variants, score, number_mutations, split=None, remove_nonsense=True):
    """get indices of variants that don't feature a nonsense mutation\n
        :parameter
            file_path: str\n
            path to the tsv file of interest\n
            split: None or list of int/float\n
            specifies the split for train, tune, test indices\n
            - float specifies fractions of the whole dataset
              eg [0.25, 0.25, 0.5] creates a train and tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            - int specifies the different number of samples per dataset
              eg [50,50,100] leads to a train and a tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            remove_nonsense: bool, optional\n
            True removes indices of nonsense mutations of all possible indices to choose from
        :returns
            data_dict: dict\n
            dictionary containing the arrays with indices for the three data splits\n
            :key 'train', 'tune', 'test'
            data: dict\n
            dictionary containing the arrays with variants (data), scores (labels) and number of mutations (mutations)
            for the train, tune and test splits\n
            prefix = ['train', 'tune', 'test']\n
            :key prefix_data, prefix_labels, prefix_mutations"""
    raw_data = pd.read_csv(file_path, delimiter="\t")
    # extract variants column
    variants_raw = np.asarray(raw_data["variant"])
    # check which variant doesn't feature a nonsense mutation
    if remove_nonsense:
        no_ast_bool = []
        for i in variants_raw:
            var = i.strip()
            no_ast_bool += ["*" not in var]
    else:
        no_ast_bool = np.ones(len(variants_raw)).astype(bool)

    # get all possible indices of all rows and shuffle them
    possible_inds = np.arange(len(raw_data))[no_ast_bool]
    np.random.shuffle(possible_inds)
    # number of rows
    pi_len = len(possible_inds)

    # check inputs
    if any([isinstance(split, list), split is None]):
        if split is not None and len(split) >= 2:
            pass
    else:
        raise ValueError("split needs to contain at least 2 inputs or needs to be None")
    if split is None:
        split = [int(np.ceil(pi_len * 0.8)), int(np.floor(pi_len * 0.15))]
    elif isinstance(split, list):
        if len(split) == 3:
            if all([isinstance(split[0], float), isinstance(split[1], float), isinstance(split[2], float)]):
                split = [int(np.ceil(pi_len * split[0])), int(np.floor(pi_len * split[1]))]
            elif all([isinstance(split[0], int), isinstance(split[1], int), isinstance(split[2], int)]):
                pass
        elif len(split) == 2:
            if all([isinstance(split[0], float), isinstance(split[1], float)]):
                split = [int(np.ceil(pi_len * split[0])), int(np.floor(pi_len * split[1]))]
            elif all([isinstance(split[0], int), isinstance(split[1], int)]):
                pass
        else:
            raise ValueError("split as list needs to contain either 2 or 3 items")
    else:
        raise ValueError("Incorrect split input needs to be list containing 'float' or 'int' or needs to be None")

    # split indices in separate data sets for train, tune, test
    train = possible_inds[:split[0]]
    tune = possible_inds[split[0]:split[0] + split[1]]
    test = possible_inds[split[0] + split[1]:]

    print("size train split: {}\nsize tune split: {}\nsize test split: {}".format(len(train), len(tune), len(test)))
    data_dict = {"train": train, "tune": tune, "test": test}

    train_dataset = raw_data.iloc[train]
    tune_dataset = raw_data.iloc[tune]
    test_dataset = raw_data.iloc[test]

    train_data = np.asarray(train_dataset[variants])
    train_labels = np.asarray(train_dataset[score])
    train_mutations = np.asarray(train_dataset[number_mutations])

    tune_data = np.asarray(tune_dataset[variants])
    tune_labels = np.asarray(tune_dataset[score])
    tune_mutations = np.asarray(tune_dataset[number_mutations])

    test_data = np.asarray(test_dataset[variants])
    test_labels = np.asarray(test_dataset[score])
    test_mutations = np.asarray(test_dataset[number_mutations])

    data = {"train_data": train_data,
            "train_labels": train_labels,
            "train_mutations": train_mutations,
            "tune_data": tune_data,
            "tune_labels": tune_labels,
            "tune_mutations": tune_mutations,
            "test_data": test_data,
            "test_labels": test_labels,
            "test_mutations": test_mutations}

    return data_dict, data


def data_coord_extraction(target_pdb_file):
    """calculates distance between residues and builds artificial CB for GLY based on the
       side chains of amino acids (!= GLY) before if there is an or after it if Gly is the start amino acid\n
       No duplicated side chain entries allowed
       :parameter
            target_pdb_file: str\n
            path to pdb file for protein of interest\n
       :returns:
            new_data: 2D ndarray\n
            contains information about all residues [[Atom type, Residue 3letter, ChainID, ResidueID],...] \n
            new_coords: 2d ndarray\n
            contains coordinates of corresponding residues to the new_data entries\n
            """
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
            :parameter
                inc_bool: bool,
                residue data of the closest amino acid with a CB
            :return
                art_cbc: ndarray\n
                CA, CB coordinates as [[xa, ya, za]], [[xb, yb, zb]]"""
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
        else:
            new_coords += i_coords.tolist()
            new_data += i_data.tolist()
    return np.asarray(new_data), np.asarray(new_coords, dtype=float)


def check_structure(path_to_pdb_file, comb_bool_cs, wt_seq_cs):
    """checks whether the given wild type sequence matches the sequence in the pdb file\n
        :parameter
            path_to_pdb_file: str\n
            path to used pdb file\n
            comb_bool_cs: 2D ndarray\n
            interacting_residues of atom_interaction_matrix_d\n
            wt_seq_cs: list\n
            wild type sequence as list eg ['A', 'V', 'L']\n
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


if __name__ == "__main__":
    pass
    from d4batch_distance_functions import atom_interaction_matrix_d
    from d4_utils import protein_settings
    _, d, _ = atom_interaction_matrix_d("datasets/bgl3.pdb", 10)
    f = protein_settings("bgl3")["sequence"]
    check_structure("datasets/bgl3.pdb", d, list(f))