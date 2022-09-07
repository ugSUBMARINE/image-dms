import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.stats import pearsonr, spearmanr

from d4_generation import data_generator_vals, DataGenerator
from d4_interactions import (
    atom_interaction_matrix_d,
    check_structure,
    model_interactions,
)
from d4_utils import (
    protein_settings,
    hydrophobicity,
    h_bonding,
    sasa,
    charge,
    side_chain_length,
    aa_dict,
    read_blosum,
    aa_dict_pos,
)


def predict_score(
    protein_pdb: str,
    protein_seq: list[str],
    variant_s: list[str],
    model_filepath: str,
    dist_th: int | float,
    algn_path: str,
    algn_base: str,
    batch_size: int = 32,
    channel_num: int = 7,
    first_ind: int = 0,
) -> np.ndarray[tuple[int], np.dtype[float]]:
    """predicts scores of variants with provided trained model
    :parameter
        - protein_pdb:
          filepath to the proteins pdb file containing its structure
        - protein_seq:
          amino acid sequence of the protein ['A', 'V', 'L', 'I']
        - variant_s:
          variants for which the score should be predicted e.g.
          ['A1S', 'K3F,I9L']
        - model_filepath:
          file path to the trained model that should be loaded
          to be used for the predictions
        - dist_th:
          distance threshold used when training the model
        - algn_path:
          path to the multiple sequence alignment in clustalw format
        - algn_base:
          name of the wild type sequence in the alignment file
        - first_ind:
          index of the start of the protein sequence
        - batch_size:
          how many variants get predicted at once
        - channel_num:
          number of (interaction) matrices that encode a variant
        - first_ind:
          offset of the start of the sequence
          (when sequence doesn't start with residue 0)
    :return
        - pred:
          predicted scores for the variants"""
    # values needed as input for the DataGenerator
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
    ) = data_generator_vals(protein_seq, algn_path, algn_base)

    dist_m, factor, comb_bool = atom_interaction_matrix_d(protein_pdb, dist_th)
    # checks whether sequence in structure as wt_seq match
    check_structure(protein_pdb, comb_bool, protein_seq, silent=True)
    # DataGenerator parameters
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
        "dist_th": dist_th,
        "co_converted": co_converted,
        "co_table": co_table,
        "co_rows": co_rows,
        "shuffle": False,
        "train": False,
    }
    # loading the model
    model = keras.models.load_model(model_filepath)
    # use DataGenerator and model.predict only when more than 64 variants should
    # be predicted
    if len(variant_s) <= 64:
        pred = []
        for i in variant_s:
            pred += [
                float(
                    model(
                        np.asarray(
                            [
                                model_interactions(
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
                            ]
                        ),
                        training=False,
                    )
                )
            ]
        pred = np.asarray(pred)
    else:
        generator = DataGenerator(variant_s, np.zeros(len(variant_s)), **params)
        pred = np.asarray(model.predict(generator).flatten())

    if len(pred) == 1:
        pred = float(pred[0])
    # predicted score(s)
    return pred


def assess_performance(
    ground_truth: np.ndarray[tuple[int], np.dtype[int | float]],
    predicted_score: np.ndarray[tuple[int], np.dtype[int | float]],
    scatter_plot: bool = False,
):
    """calculates the error and correlation of predictions made by a trained model
    :parameter
        - ground_truth: ndarray or ints or floats
          true experiamentaly determined fitness scores
        - predicted_score: ndarray of ints or floats
          scores predicted by the neural network
    :return
        - None
    """
    sr, sp = spearmanr(ground_truth, predicted_score)
    pr, pp = pearsonr(ground_truth, predicted_score)
    mse = np.mean((ground_truth - predicted_score) ** 2)
    mae = np.mean(np.abs(ground_truth - predicted_score))

    print(
        "Pearson's R: {:0.4f} ({:0.4f})\n"
        "Spearman R: {:0.4f} ({:0.4f})\n"
        "MeanAbsoluteError: {:0.4f}\n"
        "MeanSquarredError: {:0.4f}".format(pr, pp, sr, sp, mae, mse)
    )

    if scatter_plot:
        plt.scatter(ground_truth, predict_score, color="forestgreen")
        plt.xlabel("ground truth")
        plt.ylabel("predicted score")
        plt.show()


if __name__ == "__main__":
    protein = "pab1"
    ps = protein_settings(protein)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # calculate the models performance

    dms_data = pd.read_csv(
        "./nononsense/nononsense_{}.tsv".format(protein), delimiter="\t"
    )

    dms_variants = np.asarray(dms_data[ps["variants"]])[:2000]
    dms_scores = np.asarray(dms_data[ps["score"]])[:2000]

    score = predict_score(
        f"./datasets/{protein}.pdb",
        list(ps["sequence"]),
        dms_variants,
        "./result_files/saved_models/pab1_fr_50_27_08_2022_100124/",
        20,
        f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
        protein,
        first_ind=int(ps["offset"]),
    )

    assess_performance(dms_scores, score)

    # ------------------------------------------------------------------------
    # accessing predicted scores

    voi = ["A128K", "R145L,K160T"]
    score = predict_score(
        f"./datasets/{protein}.pdb",
        list(ps["sequence"]),
        voi,
        "./result_files/saved_models/pab1_fr_50_27_08_2022_100124/",
        20,
        f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
        protein,
        first_ind=int(ps["offset"]),
    )
    for i, j in zip(voi, score):
        print(f"{i}: {j}")
