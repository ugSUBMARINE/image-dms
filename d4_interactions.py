import numpy as np
from matplotlib import pyplot as plt

from d4_utils import aa_dict


def data_coord_extraction(
    target_pdb_file: str,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[any]],
    np.ndarray[tuple[int, 3], np.dtype[float]],
]:
    """reads the pdb file and stores all coordinates and the residue data - changes
    *** CA of GLY to CB ***
    :parameter
         - target_pdb_file:
           path to pdb file for protein of interest
    :returns
         - new_data:
           contains information about all residues like
           [[Atom type, Residue 3letter, ChainID, ResidueID],...]
         - new_coords:
           contains coordinates of corresponding residues to the new_data entries
    """
    # list of all data of the entries like
    # [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data.append(
                [
                    line[12:16].replace(" ", "").strip(),
                    line[17:20].replace(" ", "").strip(),
                    line[21].replace(" ", "").strip(),
                    line[22:26].replace(" ", "").strip(),
                ]
            )
            res_coords.append(
                [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
            )
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    # Change the CA from GLY to CB, so it won't be excluded in the
    # atom_interaction_matrix_d
    res_data[:, 0][
        np.all(
            np.column_stack((res_data[:, 0] == "CA", res_data[:, 1] == "GLY")), axis=1
        )
    ] = "CB"
    # remove duplicated side chain entries and store only their first appearing
    rd_un, rd_uc = np.unique(res_data, axis=0, return_index=True)
    rd_uc = np.sort(rd_uc)
    res_data = res_data[rd_uc]
    res_coords = res_coords[rd_uc]
    return res_data, res_coords


def dist_calc(
    arr1: np.ndarray[tuple[int, int], np.dtype[int | float]],
    arr2: np.ndarray[tuple[int, int], np.dtype[int | float]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    calculates euclidean distances between all points in two k-dimensional arrays
    'arr1' and 'arr2'
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


def atom_interaction_matrix_d(
    path_to_pdb_file: str, dist_th: int | str = 10.0, plot_matrices: bool = False
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[float]],
    np.ndarray[tuple[int, int], np.dtype[bool]],
]:
    """computes the adjacency matrix for a given pdb file based on the closest
    side chain atoms
    :parameter
        - path_to_pdb_file:
          path to pdb file of the protein of interest
        - dist_th:
          maximum distance in \u212B of atoms of two residues to be seen as interacting
        - plot_matrices:
          if True plots matrices for (from left to right)
            - distance to the closest side chain atom per residue
            - distance between all side chain atoms
            - inverse normalized 1st plot
            - distance between CA atoms
            - all interacting residues
    :returns
        adjacency is given per residue (the closest atom to any side chain atom of
        any other residue)
        - red2:
          adjacency (distance) matrix of the given protein with
          size len(protein_seq) x len(protein_seq)
        - red2_norm:
          inverse of the scaled red2: (1 - (red2 / np.max(red2))
        - interacting_residues:
          matrix where interacting residues are True"""
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

    # reduce all distances to the closest distance of one side chain atom to another
    # per residue
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


def hydrophobicity_matrix(
    converted: np.ndarray[tuple[int], np.dtype[float]], norm: float
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents how similar its pairs are in terms of hydrophobicity
    only for pairs that are true in res_bool_matrix
    :parameter
        - converted:
          the sequence converted to the values of the corresponding dict
        - norm:
          max value possible for interactions between two residues
    :return
        - hp_matrix:
          len(wt_seq) x len(wt_seq) matrix with the similarity in terms of
          hydrophobicity of each pair
    """
    # creating the specific interaction matrix
    interactions = np.abs(converted - converted.reshape(len(converted), -1))
    # calculating the interaction values and resizing them to be in range [0,1]
    hp_matrix = 1 - (interactions / norm)
    # return 2 * hp_matrix -1
    return hp_matrix


def hbond_matrix(
    converted: np.ndarray[tuple[int], np.dtype[int]],
    valid_vals: np.ndarray[tuple[int], np.dtype[int]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents whether pairs can form H bonds (True) or not
    (False) only for pairs that are true in res_bool_matrix
     :parameter
         - valid_vals:
           which values of the matrix are True (can form H bonds) after multiplying
           the encoded sequence against itself
         - converted:
           the sequence converted to the values of the corresponding dict
     :return
         - hb_matrix:
           len(wt_seq) x len(wt_seq) matrix where pairs that can form h bonds are True
    """
    # creating the specific interaction matrix
    interactions = converted * converted.reshape(len(converted), -1)
    # checking which interactions are can form H bonds
    hb_matrix = np.isin(interactions, valid_vals)
    return hb_matrix


def charge_matrix(
    converted: np.ndarray[tuple[int], np.dtype[int]]
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents whether pairs of amino acids are of the same charge (-1),
    of opposite charge (1), or one charged one neutral/ both uncharged (0)
    only for pairs that are true in res_bool_matrix
     :parameter
         - converted:
           the sequence converted to the values of the corresponding dict
     :return
         - interactions: 2d ndarray of floats
           len(wt_seq) x len(wt_seq) matrix containing the 'charge interaction quality
           value' for all interacting residues
    """
    # creating the specific interaction matrix
    interactions = converted * converted.reshape(len(converted), -1)
    interactions = interactions * -1
    return interactions


def interaction_area(
    converted: np.ndarray[tuple[int], np.dtype[int]], norm: int
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents the change in solvent accessible surface area (SASA)
    due to a mutation only for pairs that are true in res_bool_matrix
     :parameter
         - mut_converted: ndarray of float or int
           mutated sequence converted with the corresponding dict
         - norm: int or float
           max value possible for interactions between two residues
     :return
         - interactions: len(wt_seq) x len(wt_seq) matrix with values corresponding
           to the absolute magnitude of change in the SASA of a residue pair
    """

    # creating the specific interaction matrix
    interactions = converted + converted.reshape(len(converted), -1)
    # scaling to be in range [0,1]
    interactions = interactions / norm
    return interactions


def clashes(
    wt_converted: np.ndarray[tuple[int], np.dtype[int]],
    mut_converted: np.ndarray[tuple[int], np.dtype[int]],
    norm: float,
    dist_mat: np.ndarray[tuple[int, int], np.dtype[float]],
    dist_thr: int | float,
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents whether clashes ore holes are occurring due to the given
    mutations
    :parameter
        - wt_converted:
          wild type sequence converted with the corresponding dict
        - mut_converted:
          mutated sequence converted with the corresponding dict
        - norm:
          max value possible for interactions between two residues
        - dist_mat:
          matrix with distances between all residues
        - dist_thr:
          threshold for how close residues need to be to count as interacting
    :return
        - dist_impact: 2D ndarray of float
          len(wt_seq) x len(wt_seq) matrix with values corresponding to whether
          new mutations lead to potential clashes or holes between interacting residues
    """
    # difference in side chain length between the wild type and the variant
    diff = wt_converted - mut_converted
    # creating the specific interaction matrix
    inter = diff + diff.reshape(len(diff), -1)
    # scaling to be in range [-1,1]
    dist_impact = (dist_mat + inter) / (norm + dist_thr)
    return dist_impact


def conservation_m(
    converted: np.ndarray[tuple[int], np.dtype[float]],
    conservation_table: np.ndarray[tuple[int, 20], np.dtype[float]],
    row_ind: np.ndarray[tuple[int], np.dtype[int]],
    res_bool_matrix: np.ndarray[tuple[int, int], np.dtype[bool]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """matrix that represents how conserved residues at each sequence position are
    :parameter
        - converted:
          the sequence converted to the values of the corresponding dict
        - conservation_table:
          each row specifies which amino acids are conserved at that
          sequence position and how conserved they are
        - row_ind:
          indexing help with indices of each sequence position
        - res_bool_matrix:
          specifies which residues interact
    :return
        - compare:
          len(wt_seq) x len(wt_seq) matrix of specifying the conservation of residues
    """
    # how conserved the current amino acid at that position is
    con_vect = conservation_table[row_ind, converted]
    compare = con_vect * con_vect.reshape(len(converted), -1)
    # masking not interacting residues
    compare = compare * res_bool_matrix
    return compare


def mutate_sequences(
    wt_sequence: np.ndarray[tuple[int], np.dtype[int | float]],
    mutations: list[str],
    f_dict: dict,
    first_ind: int,
) -> np.ndarray[tuple[int], np.dtype[int | float]]:
    """mutates the wild type sequence at positions defined in mutations and returns
       the mutated sequences
    :parameter
        wt_sequence:
        the encoded wild type sequence as ndarray e.g. [0.3, 0.8, 0.1, 1.]
        mutations:
        strings where the mutations take place e.g. 'F1K,K2G'
        f_dict:
        dictionary with values for encoding
        first_ind:
        int that denotes the number of the first residue (e.g. if protein sequence
        starts with RES #3 first_ind=3)
    return:
        mutated_sequences:
        mutated sequences as list
    """
    a_to_mut = wt_sequence.copy()
    # get each mutation
    muts = mutations.strip().split(",")
    # change the wt sequence according to all the mutations
    for j in muts:
        j = j.strip()
        a_to_mut[int(j[1:-1]) - first_ind] = f_dict[j[-1]]
    return a_to_mut


def check_structure(
    path_to_pdb_file: str,
    comb_bool_cs: np.ndarray[tuple[int, int], np.dtype[bool]],
    wt_seq_cs: list[str],
    silent: bool = False,
) -> None:
    """checks whether the given wild type sequence matches the sequence in the pdb file
    :parameter
        path_to_pdb_file:
        path to used pdb file
        comb_bool_cs:
        interacting_residues of atom_interaction_matrix_d
        wt_seq_cs:
        wild type sequence as list eg ['A', 'V', 'L']
        silent:
        whether to print that the structure check is passed or not
    :return
        None
    """
    if len(comb_bool_cs) != len(wt_seq_cs):
        raise ValueError(
            "Wild type sequence doesn't match the sequence in the pdb file (check for"
            "multimers)\n"
        )
    else:
        # extract the residues from the pdb file and sort them again after their True
        # appearance
        pdb_seq = np.unique(data_coord_extraction(path_to_pdb_file)[0][:, 1:], axis=0)
        pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
        # sequence derived from pdb file
        pdb_seq_ol_list = np.asarray(list(map(aa_dict.get, pdb_seq_sorted[:, 0])))
        if not np.all(np.asarray(wt_seq_cs) == pdb_seq_ol_list):
            raise ValueError(
                "Wild type sequence doesn't match the sequence derived from the pdb "
                "file\n"
            )
        else:
            if not silent:
                print("*** structure check passed ***")


def model_interactions(
    feature_to_encode: str,
    interaction_matrix: np.ndarray[tuple[int, int], np.dtype[bool]],
    index_matrix: np.ndarray[tuple[int, int], np.dtype[int]],
    factor_matrix: np.ndarray[tuple[int, int], np.dtype[float]],
    distance_matrix: np.ndarray[tuple[int, int], np.dtype[float]],
    dist_thrh: int | float,
    first_ind: int,
    hmc: np.ndarray[tuple[int], np.dtype[int]],
    hb: dict,
    hm_pv: np.ndarray[tuple[int], np.dtype[int]],
    hpc: np.ndarray[tuple[int], np.dtype[float]],
    hp: dict,
    hpn: float,
    cmc: np.ndarray[tuple[int], np.dtype[int]],
    c: dict,
    iac: np.ndarray[tuple[int], np.dtype[int]],
    sa: dict,
    ian: int,
    clc: np.ndarray[tuple[int], np.dtype[float]],
    scl: dict,
    cln: float,
    coc: np.ndarray[tuple[int], np.dtype[int]],
    cp: dict,
    cot: np.ndarray[tuple[int, 20], np.dtype[float]],
    cor: np.ndarray[tuple[int], np.dtype[int]],
) -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
    """creates the matrix that describes the changes of interactions between residues
    due to mutation
    :parameter
        - feature_to_encode:
          the mutation that should be modeled e.g. 'S3A,K56L'
        - interaction_matrix:
          matrix where interacting residues are True
        - index_matrix:
          matrix that is symmetrical along the diagonal and describes the indices
          of the interactions
        - factor_matrix:
          describes the strength of the interaction based on the distance
          of interacting residues
        - distance_matrix:
          2D matrix with distances between all residues
        - dist_thrh:
          maximum distance of residues to count as interacting
        - first_ind:
          offset of the start of the sequence
          (when sequence doesn't start with residue 0)
        - hb:
          describing the hydrogen bonding capability of a residue
        - hp:
          describing the hydrophobicity of a residue
        - c:
          describing the charge of a residue
        - sa:
          describing the SASA of a residue
        - scl:
          describing the side chain length of a residue
        - cp:
          describing the aa position in look up table of alignment
         for a detailed description please refer to the docstring of data_generator_vals
         in d4_generation.py
        - hmc: hm_converted
        - hm_pv: hm_pos_vals
        - hpc: hp_converted
        - hpn: hp_norm
        - cmc: cm_converted
        - iac: ia_converted
        - ian: ia_norm
        - clc: cl_converted
        - cln: cl_norm
        - coc: co_converted
        - cot: co_table
        - cor: co_rows
    :return
        - n_channelxnxn ndarray encoded interactions
    """

    # for all matrices *factor masks non-interacting residues and scales the strength
    # based on their distance hydrogen bonging
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

    # interaction position
    position = index_matrix * interaction_matrix

    # when alignment file is given
    if cot is not None:
        # conservation matrix
        cur_con = mutate_sequences(coc, feature_to_encode, cp, first_ind)
        part_co = conservation_m(cur_con, cot, cor, interaction_matrix)

        return np.stack(
            (part_hb, part_hp, part_cm, part_ia, part_cl, part_co, position), axis=2
        )
    else:
        return np.stack((part_hb, part_hp, part_cm, part_ia, part_cl, position), axis=2)


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
        alignment_path="datasets/alignment_files/" "pab1_1000_experimental.clustal",
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
