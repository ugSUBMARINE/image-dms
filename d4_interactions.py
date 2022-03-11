import numpy as np
from matplotlib import pyplot as plt
from d4_utils import aa_dict


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


def atom_interaction_matrix_d(path_to_pdb_file, dist_th=10., plot_matrices=False):
    """computes the adjacency matrix for a given pdb file based on the closest side chain atoms\n
            :parameter
                path_to_pdb_file: str\n
                path to pdb file of the protein of interest\n
                dist_th: int or float, (optional - default 10.)\n
                maximum distance in \u212B of atoms of two residues to be seen as interacting\n
                plot_matrices: bool,(optional - default False)\n
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
            res_bool_matrix: boolean 2D ndarray\n
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
            res_bool_matrix: boolean 2D ndarray\n
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
            res_bool_matrix: boolean 2D ndarray\n
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
            res_bool_matrix: boolean 2D ndarray\n
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
            res_bool_matrix: boolean 2D ndarray\n
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
    inter = diff +  diff.reshape(len(diff), -1)
    dist_impact = (dist_mat + inter) / (norm + dist_thr)
    cl_mat = dist_impact * res_bool_matrix
    return cl_mat

    # new_mat = (diff + diff.reshape(len(diff), -1)) * res_bool_matrix
    # sub = dist_mat - new_mat
    # sub_ = sub * (sub != dist_mat)
    # sub_norm = sub_ / (norm + dist_thr)
    # return sub_norm


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
