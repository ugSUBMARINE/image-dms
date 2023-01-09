from itertools import product, combinations
import sys
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

from d4_interactions import model_interactions, atom_interaction_matrix_d
from d4_generation import data_generator_vals
from d4_utils import (
    aa_dict,
    aa_dict_pos,
    protein_settings,
    log_file,
    hydrophobicity,
    h_bonding,
    sasa,
    charge,
    side_chain_length,
    aa_dict_pos,
)

np.set_printoptions(threshold=sys.maxsize)


def create_mutants(
    first_ind: int,
    seq: list[str],
    tsv_path: str | None = None,
    tsv_inds_path: str | None = None,
) -> list[str]:
    """creates every single and double mutant of a given protein
    :parameter
        - first_ind:
          index of the first residue in the protein
        - seq:
          protein sequence as ['A', 'V', 'L', 'I']
        - tsv_path:
          path to tsv file of which the mutations should be excluded
        - tsv_inds_path:
          path to file containing the indices of the used mutants in the
          tsv_path file
    :return
        - var_true:
          list with variants like ['A2V', 'R3E,K5D']
    """
    seq_len = len(seq)
    amino_acids = list(aa_dict.values())
    # all residue indices in the sequence
    res_inds = np.arange(first_ind, first_ind + seq_len).astype(int)
    # all possible single mutations as NRX (NR=ind, X=amino acid)
    single = np.asarray(list(product(res_inds, amino_acids)))
    # as ONRX (O=original residue)
    single_mut = np.column_stack((seq[single[:, 0].astype(int) - first_ind], single))
    # remove A1A combinations
    sm_bool = single_mut[:, 0] != single_mut[:, 2]
    single_mut = single_mut[sm_bool]
    # single mutants as [A 1 K]
    single_mut_ind = np.arange(single_mut.shape[0])
    # all possible combinations between single mutations
    pos_pairs = np.asarray(list(combinations(single_mut_ind, 2)))
    # which single mutations do not feature the same residue (NR)
    pp_bool = single_mut[:, 1][pos_pairs[:, 0]] != single_mut[:, 1][pos_pairs[:, 1]]
    true_pairs = pos_pairs[pp_bool]
    # double mutants as [A 1 R L 4 K]
    double_muts = np.column_stack(
        (single_mut[true_pairs[:, 0]], single_mut[true_pairs[:, 1]])
    )

    # join single and double mutants as str and sort the double mutants
    single_mut_join = []
    double_mut_join = []
    for i in single_mut:
        i_join = "".join(i)
        single_mut_join.append(i_join)
    for i in double_muts:
        i_join = ",".join(np.sort(["".join(i[:3]), "".join(i[3:])]))
        double_mut_join.append(i_join)

    single_mut_join = np.asarray(single_mut_join)
    double_mut_join = np.asarray(double_mut_join)

    if tsv_path is not None and tsv_inds_path is not None:
        # read the data used in the test datasets
        tsv_data = np.asarray(pd.read_csv(tsv_path, delimiter="\t"))[:, 0]
        used_data = np.asarray(pd.read_csv(tsv_inds_path, header=None)).flatten()
        # get single mutants and same sorted double mutants
        used_sm = []
        used_dm = []
        for i in tsv_data[used_data]:
            i_split = i.strip().split(",")
            i_len = len(i_split)
            if i_len == 1:
                used_sm.append(i)
            elif i_len == 2:
                used_dm.append(",".join(np.sort(i_split)))

        used_sm = np.asarray(used_sm)
        used_dm = np.asarray(used_dm)

        # create dict date stores 0 for all mutants used in the test datasets
        sm_dict = dict(
            zip(used_sm.tolist(), np.zeros(used_sm.shape[0], dtype=int).tolist())
        )
        dm_dict = dict(
            zip(used_dm.tolist(), np.zeros(used_dm.shape[0], dtype=int).tolist())
        )

        # 1 for entries that are not in used_sm and 0 for entries that are in
        sm_hits = []
        for i in single_mut_join:
            try:
                sm_hits.append(sm_dict[i])
            except KeyError:
                sm_hits.append(1)
        sm_hits = np.asarray(sm_hits)

        dm_hits = []
        for i in double_mut_join:
            try:
                dm_hits.append(dm_dict[i])
            except KeyError:
                dm_hits.append(1)
        dm_hits = np.asarray(dm_hits)

        sm_true = single_mut_join[sm_hits.astype(bool)]
        dm_true = double_mut_join[dm_hits.astype(bool)]

        var_true = np.append(sm_true, dm_true)
    else:
        var_true = np.append(single_mut_join, double_mut_join)

    return var_true


def calc_pseudo_score(
    sequence: list[str],
    first_ind: int,
    variants: list[str],
    pdb_filepath: str,
    dist_th: int | float,
    alignment_base: str,
    alignment_path: str | None = None,
) -> np.ndarray[tuple[int], np.dtype[float]]:
    """calculates the pseudo scores to create pre training datasets
    :parameter
        - sequence:
          protein wild type sequence like ['A', 'V', 'L', 'I']
        - first_ind:
          index of the first residue in the protein
        - variants:
          variants for which the pseudo scores should be calculated
          ['A1S', 'R3K,L9T']
        - pdb_filepath:
          file path to the pdb file of the protein
        - dist_th:
          distance threshold used for model_interactions
        - alignment_base:
          name of the protein in the alignment file
        - alignment_path:
          path to the alignment file of the protein
    :return
        - pseudo_score:
          calculated pseudo scores for the given variants
    """

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
        sequence, alignment_base=protein, alignment_path=alignment_path
    )

    dist_m, factor, comb_bool = atom_interaction_matrix_d(pdb_filepath, dist_th)

    base = model_interactions(
        feature_to_encode="M{}M".format(first_ind),
        interaction_matrix=comb_bool,
        index_matrix=mat_index,
        factor_matrix=factor,
        distance_matrix=dist_m,
        dist_thrh=dist_th,
        first_ind=first_ind,
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

    pseudo_score = []
    for i in variants:
        psi = model_interactions(
            feature_to_encode=i,
            interaction_matrix=comb_bool,
            index_matrix=mat_index,
            factor_matrix=factor,
            distance_matrix=dist_m,
            dist_thrh=dist_th,
            first_ind=first_ind,
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

        inter_v = np.sum(np.abs(psi - base), axis=(0, 1))
        pseudo_score.append(np.sum(inter_v))

    pseudo_score = np.asarray(pseudo_score) / 100

    return pseudo_score


def create_pt_ds(file_path: str, variants: list[str], scores: list[float]) -> None:
    """stores pseudo scores int the same formate as the original datasets
    :parameter
        - file_path: str
          where to store the file
        - variants: list of str
          variants like ['A1S', 'R3K,L9T']
        - scores: list of floats
          pseudo scores of the variants
    :return
        - None
    """

    file = open(file_path, "w+")
    file.write("variant\tnum_mutations\tscore\n")
    for i, j in zip(variants, scores):
        file.write("{}\t{}\t{:0.8f}\n".format(i, len(i.split(",")), j))
    file.close()


if __name__ == "__main__":
    pass
    """
    protein = "avgfp"
    p_data = protein_settings(protein)
    first_ind = int(p_data["offset"])
    seq = np.asarray(list(p_data["sequence"]))
    size = 50

    runs = ["first", "second", "third"]
    for r in runs:
        if r == "first":
            pre = "fr"
        elif r == "second":
            pre = "sr"
        elif r == "third":
            pre = "tr"
        else:
            r = None
        for size in [50, 100, 250, 500, 1000, 2000, 6000]:
            new_variants = create_mutants(
                first_ind,
                seq,
                f"./nononsense/nononsense_{protein}.tsv",
                f"./nononsense/{r}_split_run/{protein}_even_splits/split_{size}/stest.txt",
            )

            chosen_var = np.random.choice(
                np.arange(len(new_variants)), replace=False, size=40000
            )

            ns = calc_pseudo_score(
                variants=new_variants[chosen_var],
                sequence=seq,
                pdb_filepath=f"./datasets/{protein}.pdb",
                alignment_base=protein,
                alignment_path=f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
                first_ind=first_ind,
                dist_th=20,
            )

            create_pt_ds(
                f"./datasets/pseudo_scores/{protein}/{protein}_{pre}_{size}.tsv",
                new_variants[chosen_var],
                ns,
            )
    """
