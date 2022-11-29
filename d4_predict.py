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
from d4_split import read_split_file

from d4_argpars import predict_dict


def predict_score(
    protein_pdb: str,
    protein_seq: list[str],
    variant_s: list[str],
    model_filepath: str,
    dist_th: int | float,
    algn_path: str | None = None,
    algn_base: str | None = None,
    batch_size: int = 32,
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

    # set number of channels based on presence of the alignment file
    channel_num = 7
    if algn_path is None:
        channel_num = 6

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


def recall_calc(
    protein: str,
    test_var_inds_file: str,
    model_filepath: str,
    steps: int = 100,
    test_size: int | None = None,
    N: int = 100,
):
    """calculates the recall percentage for a given network and protein dataset
    :parameter
        - protein:
          name of the protein
        - test_var_inds_file:
          file path to the test.txt file that contains the indices of samples
          for the nononsense_PROTEIN.tsv
        - model_filepath:
          file path to the trained model_filepath
        - steps:
          number of steps in range function that should test the
          recall percentage
        - test_size:
          max number of samples in one prediction
        - N:
          number of top samples (budget)
    :return
        - num
          list with sample sizes per data point
        - recall_perc
          percentage of samples in the top N predictions
    """
    ps = protein_settings(protein)
    dms_data = pd.read_csv(f"./nononsense/nononsense_{protein}.tsv", delimiter="\t")
    tvi = read_split_file(test_var_inds_file)

    dms_variants = np.asarray(dms_data[ps["variants"]])[tvi]
    dms_scores = np.asarray(dms_data[ps["score"]])[tvi]

    # predict the scores of the test data
    score = predict_score(
        f"./datasets/{protein}.pdb",
        list(ps["sequence"]),
        dms_variants,
        model_filepath,
        20,
        f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
        protein,
        first_ind=int(ps["offset"]),
    )

    # sort the scores of the true data for the best variants
    top_n_ground_trouth = dms_variants[np.argsort(dms_scores)[::-1]][:N]
    # variants sorted by prediction
    pred_sort = dms_variants[np.argsort(score)[::-1]]
    # variants randomly 'sorted'
    random_ind = np.arange(len(dms_scores))
    np.random.shuffle(random_ind)
    random_sort = dms_variants[random_ind]

    # calculates for different budgets how many of the predictions are truly in the
    # top N
    recall_perc = []
    num = []
    best_case = []
    random_case = []
    if test_size is None:
        test_size = len(tvi) + 1
    for i in range(10, test_size, steps):
        predicted_top_n = pred_sort[:i]
        random_top_n = random_sort[:i]
        # percentage of correctly recalls in the top N
        recall_perc.append(np.sum(np.isin(predicted_top_n, top_n_ground_trouth)) / N)
        # best case percentage
        best_calc = i / N
        if best_calc > 1.0:
            best_calc = 1.0
        best_case.append(best_calc)
        # random choices percentage
        random_case.append(np.sum(np.isin(random_top_n, top_n_ground_trouth)) / N)
        num.append(i)
    return num, recall_perc, random_case, best_case


if __name__ == "__main__":
    protein = "pab1"
    ps = protein_settings(protein)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # calculate the models performance
    """
    dms_data = pd.read_csv(
        f"nononsense/nononsense_{protein}.tsv", delimiter="\t"
    )

    dms_variants = np.asarray(dms_data[ps["variants"]])[:2000]
    dms_scores = np.asarray(dms_data[ps["score"]])[:2000]

    score = predict_score(
        f"./datasets/{protein}.pdb",
        list(ps["sequence"]),
        dms_variants,
        "./pub_result_files/saved_models/recall_whole_ds/"
        "nononsense_pab1_04_11_2022_094109/",
        20,
        f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
        protein,
        first_ind=int(ps["offset"]),
    )

    assess_performance(dms_scores, score)
    """
    # ------------------------------------------------------------------------
    # accessing predicted scores
    """
    voi = ["A128K", "R145L,K160T"]
    score = predict_score(
        protein_pdb=f"./datasets/{protein}.pdb",
        protein_seq=list(ps["sequence"]),
        variant_s=voi,
        model_filepath="./result_files/saved_models/pab1_fr_50_27_08_2022_100124/",
        dist_th=20,
        algn_path=f"./datasets/alignment_files/{protein}_1000_experimental.clustal",
        algn_base=protein,
        first_ind=int(ps["offset"]),
    )
    for i, j in zip(voi, score):
        print(f"{i}: {j}")
    """
    # ------------------------------------------------------------------------
    # calculate recall percentage
    """
    print(recall_calc(
        "gb1",
        "result_files/rr5/recall/recall_fract_splits/dense_net2/"
        "nononsense_gb1_28_09_2022_142206_splits0/test.txt",
        "result_files/saved_models/recall_fract_ds/dense_net2/"
        "nononsense_gb1_28_09_2022_142206/",
    ))
    """
    predict_score(**predict_dict())
