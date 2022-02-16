import numpy as np
import pandas as pd
import sys
import keras
import scipy.stats

from d4batch_distance_functions import aa_dict, atom_interaction_matrix_d, data_generator_vals, check_structure, \
    DataGenerator
from d4batch_driver import seq_dict

np.set_printoptions(threshold=sys.maxsize)


def create_augmentation(file, seq, first_ind, augment_num, num_mutations="num_mutations", variant="variant"):
    """creates augment_num random variants for each number of mutations if possible, and they are not already present
        in real data"""
    # wt sequence as list
    seq_list = list(seq)
    # which mutations are present in the file ie [1,2] = single and double mutants
    mutations = np.unique(file[num_mutations])
    # get rid of * mutations
    ast_bool = []
    for i in file[variant]:
        ast_bool += ["*" not in i]
    file = file[ast_bool]
    # all amino acids as one letter
    aa = list(aa_dict.values())
    new_mut_all = []
    for i in mutations:
        # data in file where variants with i mutations are present
        file_i = file[file[num_mutations] == i]
        res_j = []
        res_j_mut = []
        # get from each variant its position and the mutated residue
        for j in file_i[variant]:
            res_k = []
            res_k_mut = []
            for k in j.strip().split(","):
                res_k += [int(k[1:-1])]  # - first_ind]
                res_k_mut += [k[-1]]
            res_j += [res_k]
            res_j_mut += [res_k_mut]
        res_j = np.asarray(res_j).astype(int)
        # create as many random samples of i variants as set in augment_num ie [10, 2, 5, 9] and [A, L, K, R]
        t = []
        for u in range(i):
            # t += [np.random.choice(np.arange(first_ind, np.max(res_j) + 1), size=augment_num).tolist()]
            t += [np.random.choice(np.arange(first_ind, len(seq_list) + 1), size=augment_num).tolist()]
            t += [np.random.choice(aa, size=augment_num).tolist()]
        artificial = np.asarray(t).transpose()
        res_j = np.asarray(res_j)
        res_j_mut = np.asarray(res_j_mut)
        # same as t but the original data
        to = []
        for b in range(i):
            to += [res_j[:, b].tolist()]
            to += [res_j_mut[:, b].tolist()]
        original = np.asarray(to).transpose()
        # getting rid of the t variants that are already present in the original data
        n, c = np.unique(np.concatenate((artificial, original, original)), axis=0, return_counts=True)
        new_mut = n[c == 1]
        # getting the residue letter from the wt sequence to go from ['139' 'R' '155' 'A'] to ['A139R', 'S155A']
        for f in new_mut:
            fint = []
            for fi in np.arange(start=0, stop=len(f), step=2):
                fint += ["".join([seq_list[int(f[fi]) - first_ind], f[fi], f[fi + 1]])]
            new_mut_all += [",".join(fint)]
    return np.asarray(new_mut_all)


def split_dataframes(num_mutations, variant, score, pre_train_fract, val_fraction):
    """pt, td, vd = split_dataframes(num_mutations="num_mutations", variant="variant", score="score",
            pre_train_fract=0.3, val_fraction=0.15)"""
    # dataset to validate the model in the end
    validation_dataset = tsv_file.sample(frac=val_fraction, random_state=1)
    tsv_new = tsv_file.drop(validation_dataset.index)
    # dataset to train the model before training it with augmented data
    pre_train_dataset = []
    # dataset to tune the pre trained model
    tune_dataset = []
    column_names = []
    all_mutations = np.unique(tsv_new[num_mutations])
    for i in all_mutations:
        file_i = tsv_new[tsv_new[num_mutations] == i]
        file_i = file_i[[variant, num_mutations, score]]
        pre_i = file_i.sample(frac=pre_train_fract, random_state=1)
        remain = file_i.drop(pre_i.index)
        pre_train_dataset += pre_i.values.tolist()
        tune_dataset += remain.values.tolist()
        if i == 1:
            column_names += remain.columns.tolist()
    pre_train_dataset = pd.DataFrame(pre_train_dataset, columns=column_names)
    tune_dataset = pd.DataFrame(tune_dataset, columns=column_names)
    return pre_train_dataset, tune_dataset, validation_dataset


def validate_model(name, model, data_generator_params, features, labels=None, pseudo_labels_path=None, human_data=None,
                   num_mutations="num_mutations", variant="variant", score="score"):
    # import a trained model with features and labels and calculate the mae spearman r pearson r
    model = keras.models.load_model(model)
    data_generator = DataGenerator(np.asarray(features), np.zeros(len(features)), **data_generator_params)
    predicted_labels = model.predict(data_generator).flatten()
    if pseudo_labels_path is None:
        error = np.abs(predicted_labels - labels)
        pearson_r, pearson_r_p = scipy.stats.pearsonr(labels.astype(float), predicted_labels.astype(float))
        spearman_r, spearman_r_p = scipy.stats.spearmanr(labels.astype(float), predicted_labels.astype(float))
        return np.mean(error), error.std(), pearson_r, pearson_r_p, spearman_r, spearman_r_p
    else:
        # use trained model to generate labels for augmented data and create a new tsv with it
        # human_data: data which was really measured and not augmented df!
        mutations = []
        for i in features:
            mutations += [len(i.strip().split(","))]
        col = np.column_stack((features, mutations, predicted_labels))
        pl_df = pd.DataFrame(col, columns=[variant, num_mutations, score])
        con_df = pl_df.append(human_data[[variant, num_mutations, score]])
        # print(human_data[[variant, num_mutations, score]].columns)
        # print(pl_df.columns)
        pd.DataFrame.to_csv(con_df, pseudo_labels_path + "/" + name + ".tsv", sep="\t", index=False)


if __name__ == "__main__":
    # tsv_file = pd.read_csv("augmentation/pre_training_avgfp.tsv", delimiter="\t")
    pdb_ex = "datasets/pab1_rosetta_model.pdb"
    protein = "pab1"
    sequence = seq_dict()[protein]
    # augmented_features = create_augmentation(tsv_file, sequence, 1,  6000)
    # print(len(augmented_features))
    # pre_training_data = tsv_file.sample(frac=0.02, random_state=1)
    # validation_data = tsv_file.drop(pre_training_data.index)
    # print(pre_training_data, validation_data)
    # pd.DataFrame.to_csv(pre_training_data, "augmentation/pre_training_avgfp.tsv", sep="\t", index=False)
    # pd.DataFrame.to_csv(validation_data, "augmentation/validate_avgfp.tsv", sep="\t", index=False)

    hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
    cm_converted, ia_converted, mat_index = data_generator_vals(sequence)

    _, factor, comb_bool = atom_interaction_matrix_d(pdb_ex, dist_th=15, plot_matrices=False)
    check_structure(pdb_ex, comb_bool, list(sequence))

    dg_vals_pl = {'wild_type_seq': sequence,
                  'interaction_matrix': comb_bool,
                  'dim': comb_bool.shape,
                  'n_channels': 5,
                  'batch_size': 1,
                  'first_ind': 126,  # 126
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
    """
    # create augmented data
    pre_training_data = pd.read_csv("augmentation/pre_training_avgfp.tsv", delimiter="\t")
    validate_model(name="avgfp_augmented",
                   model="result_files/pre_training_avgfp_31_01_2022_0751/pre_training_avgfp_31_01_2022_0751",
                   data_generator_params=dg_vals_pl, features=augmented_features,
                   pseudo_labels_path="augmentation", human_data=pre_training_data, score="score_wt_norm")

    """

    # validate model
    val_data = pd.read_csv("pab1_augentation_1/validate_pab1.tsv", delimiter="\t")
    val_features = val_data["variant"]
    val_labels = val_data["score"]  # score score_wt_norm
    val_features = np.asarray(val_features)
    val_labels = np.asarray(val_labels)
    val_bool = []
    for i in val_features:
        val_bool += ["*" not in i]
    val_labels = val_labels[val_bool]
    val_features = val_features[val_bool]

    a, b, c, d, e, f = validate_model(name="pab1_pretraining",
                                      model="pab1_augentation_1/pre_training_pab1_27_01_2022_1536/pre_training_pab1_27_01_2022_1536",
                                      data_generator_params=dg_vals_pl, features=val_features, labels=val_labels)
    print("MAE: {}\nSTD: {}\nPearson's r: {}\nPearson's r p-value:{}\nSpearman r: {}\nSpearman r p-value: {}\n".
          format(str(a), str(b), str(c), str(d), str(e), str(f)))

    # print(tsv_file)
    # more_than_five = tsv_file[tsv_file["num_mutations"] > 5]
    # five_max = tsv_file[tsv_file["num_mutations"] <= 5]
    # sub_five_max = five_max.sample(n=2000, random_state=1)
    # five_max_rest = five_max.drop(sub_five_max.index)
    # print(five_max)
    # print(sub_five_max)
    # print(five_max_rest)
    # print(more_than_five)
    # pd.DataFrame.to_csv(pd.concat([more_than_five, five_max_rest], ignore_index=True),
    #                     "augmentation/validate_avgfp.tsv", sep="\t", index=False)
    # pd.DataFrame.to_csv(sub_five_max, "augmentation/pre_training_avgfp.tsv", sep="\t", index=False)
