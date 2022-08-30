import numpy as np
from matplotlib import pyplot as plt

from d4_utils import aa_dict


def data_coord_extraction(target_pdb_file):
    """calculates distance between residues and builds artificial CB for GLY based on the
    side chains of amino acids (!= GLY) before if there is one or after it if Gly is the start amino acid\n
    No duplicated side chain entries allowed\n
    :parameter
         - target_pdb_file: str\n
           path to pdb file for protein of interest\n
    :returns
         - new_data: 2D ndarray\n
           contains information about all residues like [[Atom type, Residue 3letter, ChainID, ResidueID],...] \n
         - new_coords: 2d ndarray\n
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
            res_data += [
                [
                    line[12:16].replace(" ", "").strip(),
                    line[17:20].replace(" ", "").strip(),
                    line[21].replace(" ", "").strip(),
                    line[22:26].replace(" ", "").strip(),
                ]
            ]
            res_coords += [
                [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
            ]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    # Change the CA from GLY to CB, so it won't be excluded in the atom_interaction_matrix_d
    res_data[:, 0][
        np.all(
            np.column_stack((res_data[:, 0] == "CA", res_data[:, 1] == "GLY")), axis=1
        )
    ] = "CB"
    # remove duplicated side chain entries and store only their first appearings
    rd_un, rd_uc = np.unique(res_data, axis=0, return_index=True)
    rd_uc = np.sort(rd_uc)
    res_data = res_data[rd_uc]
    res_coords = res_coords[rd_uc]
    return res_data, res_coords


def dist_calc_old(arr1, arr2):
    """calculates distance between arr1 and arr2 and returns a 2D array with all distances of all arr1 points
    against all arr2 points\n
    :parameter
        - arr1, arr2: ndarray\n
          2D arrays of 1D lists with 3D coordinates eg [[x1, y1, z1],...]\n
    :return
        - dist: 2D ndarray\n
          len(arr1) x len(arr2) distance matrix between arr1 and arr2\n"""
    # get only the x,y,z coordinates from the input arrays and reshape them so they can be subtracted from each other
    arr1_coords_rs = arr1.reshape(arr1.shape[0], 1, arr1.shape[1])
    arr2_coord_rs = arr2.reshape(1, arr2.shape[0], arr2.shape[1])
    # calculating the distance between each point and returning a 2D array with all distances
    dist = np.sqrt(((arr1_coords_rs - arr2_coord_rs) ** 2).sum(axis=2))
    return dist


def dist_calc(arr1, arr2):
    """
    calculates euclidean distances between all points in two k-dimensional arrays 'arr1' and 'arr2'
        :parameter
            - arr1: N x k array
            - arr2: M x k array
        :return
            - dist: M x N array with pairwise distances
    """
    norm_1 = np.sum(arr1 * arr1, axis=1).reshape(1, -1)
    norm_2 = np.sum(arr2 * arr2, axis=1).reshape(-1, 1)

    dist = (norm_1 + norm_2) - 2.0 * np.dot(arr2, arr1.T)
    # necessary due to limited numerical accuracy
    dist[dist < 1.0e-11] = 0.0

    return np.sqrt(dist)


def atom_interaction_matrix_d(path_to_pdb_file, dist_th=10.0, plot_matrices=False):
    """computes the adjacency matrix for a given pdb file based on the closest side chain atoms\n
    :parameter
        - path_to_pdb_file: str\n
          path to pdb file of the protein of interest\n
        - dist_th: int or float, (optional - default 10.)\n
          maximum distance in \u212B of atoms of two residues to be seen as interacting\n
        - plot_matrices: bool, (optional - default False)\n
          if True plots matrices for (from left to right)
            - distance to the closest side chain atom per residue\n
            - distance between all side chain atoms\n
            - inverse normalized 1st plot\n
            - distance between CA atoms\n
            - all interacting residues\n
    :returns
        adjacency is given per residue (the closest atom to any side chain atom of any other residue)\n
        - red2: 2D ndarray of floats\n
          adjacency (distance) matrix of the given protein with size len(protein_seq) x len(protein_seq)\n
        - red2_norm: 2D ndarray of floats\n
          inverse of the scaled red2: (1 - (red2 / np.max(red2))\n
        - interacting_residues: boolean 2D ndarray\n
          matrix where interacting residues are True\n"""
    # data [[ATOM, RES, CHAIN, ResNR],..]
    data, coords = data_coord_extraction(path_to_pdb_file)
    # ca alpha distances
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
    udata, uind, ucount = np.unique(
        data[:, 1:], axis=0, return_index=True, return_counts=True
    )
    # sort it again by chain and sequence
    u_sort = np.lexsort((udata[:, 2].astype(int), udata[:, 1]))
    # udata = udata[u_sort]
    uind = uind[u_sort]
    ucount = ucount[u_sort]

    # reduce all distances to the closest distance of one side chain atom to another per residue
    red1 = []
    for i, j in zip(uind, ucount):
        red1.append(np.min(d[:, i : i + j], axis=1))
    red1 = np.asarray(red1)

    red2 = []
    for i, j in zip(uind, ucount):
        red2.append(np.min(red1[:, i : i + j], axis=1))
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


def hydrophobicity_matrix(converted, norm):
    """matrix that represents how similar its pairs are in terms of hydrophobicity only for pairs that are true in
    res_bool_matrix\n
    :parameter
        - converted: ndarray of int or floats\n
          the sequence converted to the values of the corresponding dict\n
        - norm: float or int\n
          max value possible for interactions between two residues\n
    :return
        - hp_matrix: 2d ndarray of floats\n
          len(wt_seq) x len(wt_seq) matrix with the similarity in terms of hydrophobicity of each pair\n"""
    # creating the specific interaction matrix
    interactions = np.abs(converted - converted.reshape(len(converted), -1))
    # calculating the interaction values and resizing them to be in range [0,1]
    hp_matrix = 1 - (interactions / norm)
    # return 2 * hp_matrix -1
    return hp_matrix


def hbond_matrix(converted, valid_vals):
    """matrix that represents whether pairs can form H bonds (True) or not (False) only for pairs that are true in
    res_bool_matrix\n
     :parameter
         - valid_vals: ndarray of int or float\n
           which values of the matrix are True (can form H bonds) after multiplying the encoded sequence against
           itself\n
         - converted: ndarray of int or floats\n
           the sequence converted to the values of the corresponding dict\n
     :return
         - hb_matrix: 2d ndarray of floats\n
           len(wt_seq) x len(wt_seq) matrix where pairs that can form h bonds are True\n"""
    # creating the specific interaction matrix
    interactions = converted * converted.reshape(len(converted), -1)
    # checking which interactions are can form H bonds
    hb_matrix = np.isin(interactions, valid_vals)
    return hb_matrix


def charge_matrix(converted):
    """matrix that represents whether pairs of amino acids are of the same charge (-1), of opposite charge
    (1), or one charged one neutral/ both uncharged (0) only for pairs that are true in res_bool_matrix\n
     :parameter
         - converted: ndarray of int or floats\n
           the sequence converted to the values of the corresponding dict\n
     :return
         - interactions: 2d ndarray of floats\n
           len(wt_seq) x len(wt_seq) matrix containing the 'charge interaction quality value'
           for all interacting residues\n"""
    # creating the specific interaction matrix
    interactions = converted * converted.reshape(len(converted), -1)
    interactions = interactions * -1
    return interactions


def interaction_area(converted, norm):
    """matrix that represents the change in solvent accessible surface area (SASA) due to a mutation
    only for pairs that are true in res_bool_matrix\n
     :parameter
         - mut_converted: ndarray of float or int\n
           mutated sequence converted with the corresponding dict\n
         - norm: int or float\n
           max value possible for interactions between two residues\n
     :return
         - interactions: len(wt_seq) x len(wt_seq) matrix with values corresponding to the
           absolute magnitude of change in the SASA of a residue pair\n"""

    # creating the specific interaction matrix
    interactions = converted + converted.reshape(len(converted), -1)
    # scaling to be in range [0,1]
    interactions = interactions / norm
    return interactions


def clashes(wt_converted, mut_converted, norm, dist_mat, dist_thr):
    """matrix that represents whether clashes ore holes are occurring due to the given mutations\n
    :parameter
        - wt_converted: ndarray of float or int\n
          wild type sequence converted with the corresponding dict\n
        - mut_converted: ndarray of float or int\n
          mutated sequence converted with the corresponding dict\n
        - norm: int or float\n
          max value possible for interactions between two residues\n
        - dist_mat: 2D ndarray of float\n
          matrix with distances between all residues\n
        - dist_thr: int or float\n
          threshold for how close residues need to be to count as interacting\n
    :return
        - dist_impact: 2D ndarray of float\n
          len(wt_seq) x len(wt_seq) matrix with values corresponding to whether new mutations lead to potential
          clashes or holes between interacting residues\n"""
    # difference in side chain length between the wild type and the variant
    diff = wt_converted - mut_converted
    # creating the specific interaction matrix
    inter = diff + diff.reshape(len(diff), -1)
    # scaling to be in range [-1,1]
    dist_impact = (dist_mat + inter) / (norm + dist_thr)
    return dist_impact


def conservation_m(converted, conservation_table, row_ind, res_bool_matrix):
    """matrix that represents how conserved residues at each sequence position are\n
    :parameter
        - converted: ndarray of int or floats\n
          the sequence converted to the values of the corresponding dict\n
        - conservation_table: nx20 ndarray of floats\n
          each row specifies which amino acids are conserved at that
          sequence position and how conserved they are\n
        - row_ind: 1D ndarray of ints\n
          indexing help with indices of each sequence position\n
        - res_bool_matrix: 2D boolean ndarray\n
          specifies which residues interact\n
    :return
        - compare: 2d ndarray of floats\n
          len(wt_seq) x len(wt_seq) matrix of specifying the conservation
          of residues\n"""
    # how conserved the current amino acid at that position is
    con_vect = conservation_table[row_ind, converted]
    compare = con_vect * con_vect.reshape(len(converted), -1)
    # masking not interacting residues
    compare = compare * res_bool_matrix
    return compare


def mutate_sequences(wt_sequence, mutations, f_dict, first_ind):
    """mutates the wild type sequence at positions defined in mutations and returns the mutated sequences\n
    :parameter
        wt_sequence: ndarray of float or int\n
        the encoded wild type sequence as ndarray e.g. [0.3, 0.8, 0.1, 1.]\n
        mutations: list of str\n
        strings where the mutations take place e.g. 'F1K,K2G'\n
        f_dict: dict\n
        dictionary with values for encoding\n
        first_ind: int\n
        int that denotes the number of the first residue (e.g. if protein sequence starts with RES #3 first_ind=3)
    return:
        mutated_sequences: list of float or int\n
        mutated sequences as list\n"""
    a_to_mut = wt_sequence.copy()
    # get each mutation
    muts = mutations.strip().split(",")
    # change the wt sequence according to all the mutations
    for j in muts:
        j = j.strip()
        a_to_mut[int(j[1:-1]) - first_ind] = f_dict[j[-1]]
    return a_to_mut


def check_structure(path_to_pdb_file, comb_bool_cs, wt_seq_cs, silent=False):
    """checks whether the given wild type sequence matches the sequence in the pdb file\n
    :parameter
        path_to_pdb_file: str\n
        path to used pdb file\n
        comb_bool_cs: 2D ndarray\n
        interacting_residues of atom_interaction_matrix_d\n
        wt_seq_cs: list\n
        wild type sequence as list eg ['A', 'V', 'L']\n
        silent: bool, (optional default False)\n
        whether to print that the structure check is passed or not\n
    :return
        None
    """
    if len(comb_bool_cs) != len(wt_seq_cs):
        raise ValueError(
            "Wild type sequence doesn't match the sequence in the pdb file (check for multimers)\n"
        )
    else:
        # extract the residues from the pdb file and sort them again after their true appearance
        pdb_seq = np.unique(data_coord_extraction(path_to_pdb_file)[0][:, 1:], axis=0)
        pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
        # sequence derived from pdb file
        pdb_seq_ol_list = np.asarray(list(map(aa_dict.get, pdb_seq_sorted[:, 0])))
        if not np.all(np.asarray(wt_seq_cs) == pdb_seq_ol_list):
            raise ValueError(
                "Wild type sequence doesn't match the sequence derived from the pdb file\n"
            )
        else:
            if not silent:
                print("*** structure check passed ***")


def model_interactions(
    feature_to_encode,
    interaction_matrix,
    index_matrix,
    factor_matrix,
    distance_matrix,
    dist_thrh,
    first_ind,
    hmc,
    hb,
    hm_pv,
    hpc,
    hp,
    hpn,
    cmc,
    c,
    iac,
    sa,
    ian,
    clc,
    scl,
    cln,
    coc,
    cp,
    cot,
    cor,
):
    """creates the matrix that describes the changes of interactions between residues due to mutation\n
    :parameter
        - feature_to_encode: str\n
          the mutation that should be modeled e.g. 'S3A,K56L'\n
        - interaction_matrix: 2D boolean ndarray\n
          matrix where interacting residues are True\n
        - index_matrix: 2D ndarray of ints\n
          matrix that is symmetrical along the diagonal and describes the indices of the interactions\n
        - factor_matrix: 2D ndarray of floats\n
          describes the strength of the interaction based on the distance of interacting residues\n
        - distance_matrix: 2D ndarray of floats\n
          2D matrix with distances between all residues\n
        - dist_thrh: int or float\n
          maximum distance of residues to count as interacting\n
        - first_ind: int\n
          offset of the start of the sequence (when sequence doesn't start with residue 0)\n
        - hb: dict\n
          describing the hydrogen bonding capability of a residue\n
        - hp: dict\n
          describing the hydrophobicity of a residue\n
        - c: dict\n
          describing the charge of a residue\n
        - sa: dict\n
          describing the SASA of a residue\n
        - scl: dict\n
          describing the side chain length of a residue\n
        - cp: dict\n
          describing the aa position in look up table of alignemtn\n
        for a detailed description please refer to the docstring of data_generator_vals in d4_generation.py\n
        - hmc: hm_converted\n
        - hm_pv: hm_pos_vals\n
        - hpc: hp_converted\n
        - hpn: hp_norm\n
        - cmc: cm_converted\n
        - iac: ia_converted\n
        - ian: ia_norm\n
        - clc: cl_converted\n
        - cln: cl_norm\n
        - coc: co_converted\n
        - cot: co_table\n
        - cor: co_rows\n
    :return
        - n_channelxnxn ndarray of floats\n
          encoded interactions"""

    # for all matrices *factor masks non-interacting residues and scales the strength based on their distance
    # hydrogen bonging
    cur_hb = mutate_sequences(hmc, feature_to_encode, hb, first_ind)
    part_hb = hbond_matrix(cur_hb, hm_pv) * factor_matrix

    # hydrophobicity
    cur_hp = mutate_sequences(hpc, feature_to_encode, hp, first_ind)
    part_hp = hydrophobicity_matrix(cur_hp, hpn) * factor_matrix

    # charge
    cur_cm = mutate_sequences(cmc, feature_to_encode, c, first_ind)
    part_cm = charge_matrix(cur_cm) * factor_matrix

    # interaction area
    cur_ia = mutate_sequences(iac, feature_to_encode, sa, first_ind)
    part_ia = interaction_area(cur_ia, ian) * factor_matrix

    # clashes
    cur_cl = mutate_sequences(clc, feature_to_encode, scl, first_ind)
    part_cl = (
        clashes(clc, cur_cl, cln, distance_matrix, dist_thr=dist_thrh) * factor_matrix
    )

    # conservation matrix
    cur_con = mutate_sequences(coc, feature_to_encode, cp, first_ind)
    part_co = conservation_m(cur_con, cot, cor, interaction_matrix)  # * factor_matrix

    # interaction position
    position = index_matrix * interaction_matrix  # * factor_matrix

    return np.stack(
        (part_hb, part_hp, part_cm, part_ia, part_cl, part_co, position), axis=2
    )


if __name__ == "__main__":
    from d4_generation import data_generator_vals
    from d4_utils import protein_settings
    from d4_utils import (
        hydrophobicity,
        h_bonding,
        sasa,
        charge,
        side_chain_length,
        aa_dict_pos,
    )

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
    ) = data_generator_vals(
        protein_settings("pab1")["sequence"],
        alignment_base="pab1",
        alignment_path="./datasets/alignment_files/pab1_1000_experimental.clustal",
    )
    dist_m, factor, comb_bool = atom_interaction_matrix_d("datasets/pab1.pdb", 20)

    a = model_interactions(
        feature_to_encode="N127R,A178H,G177S,A178G,G188H,E195K,L133M,P135S",
        interaction_matrix=comb_bool,
        index_matrix=mat_index,
        factor_matrix=factor,
        distance_matrix=dist_m,
        dist_thrh=20,
        first_ind=126,
        hmc=hm_converted,
        hb=h_bonding,
        hm_pv=hm_pos_vals,
        hpc=hp_converted,
        hp=hydrophobicity,
        hpn=hp_norm,
        cmc=cm_converted,
        c=charge,
        iac=ia_converted,
        sa=sasa,
        ian=ia_norm,
        clc=cl_converted,
        scl=side_chain_length,
        cln=cl_norm,
        coc=co_converted,
        cp=aa_dict_pos,
        cot=co_table,
        cor=co_rows,
    )

    b = model_interactions(
        feature_to_encode="N127E,A178S,G177S,A178G,G188A,E195F,L133M,P135W",
        interaction_matrix=comb_bool,
        index_matrix=mat_index,
        factor_matrix=factor,
        distance_matrix=dist_m,
        dist_thrh=20,
        first_ind=126,
        hmc=hm_converted,
        hb=h_bonding,
        hm_pv=hm_pos_vals,
        hpc=hp_converted,
        hp=hydrophobicity,
        hpn=hp_norm,
        cmc=cm_converted,
        c=charge,
        iac=ia_converted,
        sa=sasa,
        ian=ia_norm,
        clc=cl_converted,
        scl=side_chain_length,
        cln=cl_norm,
        coc=co_converted,
        cp=aa_dict_pos,
        cot=co_table,
        cor=co_rows,
    )

    # size = 5
    # row = np.random.randint(0, 75 - size)
    # col = np.random.randint(0, 75 - size)

    # a[row: row + 10, :, :] = 0.
    # a[:, row: row + 10, :] = 0.
    # a[[10, 20, 50],[49, 14, 28],0] = 1000
    # size = 10
    # r = np.random.randint(0, 75-size)
    # c = np.random.randint(0, 75-size)
    # a[r:r+size, c:c+size, 0] = np.random.uniform(0,1,size*size).reshape(size, size)
    for i in range(7):
        # print(np.max(a[:,:,i]), np.min(a[:,:,i]))
        print(i)
        plt.imshow(a[:, :, i])
        plt.colorbar()
        plt.show()
        # print(np.max(a[:,:,i]), np.min(a[:,:,i]))
