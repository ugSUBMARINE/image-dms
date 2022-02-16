import numpy as np
import pandas as pd
from d4batch_distance_functions import aa_dict, atom_interaction_matrix_d, data_generator_vals, check_structure, \
    DataGenerator
from d4batch_driver import seq_dict


def predict(model, prediction_generator, prediction_generator_params, features):
    """returns predictions for given features with the use of a DataGenerator for the model and DataGenerator parameters
    as dict"""
    generator = prediction_generator(features, np.zeros(len(features)), **prediction_generator_params)
    prediction = model.predict(generator).flatten()
    return prediction


def recall(features, true_score, predicted_score, cutoff):
    """test how many of the features sorted according to their predicted scores are
            also in the top 'cutoff' true scores
        :returns number of correct in top 'cutoff' and the percentage"""
    features = np.asarray(features)
    predicted_score = np.asarray(predicted_score)
    true_score = np.asarray(true_score)
    predicted_order = features[np.argsort(np.asarray(predicted_score))[::-1]]
    true_order = features[np.argsort(np.asarray(true_score))[::-1]]
    hit = np.sum(np.isin(predicted_order[:cutoff], true_order[:cutoff]))
    return hit, hit / cutoff


def create_dataset_tsv(directory, name, tsv_to_split, split, pd_del="\t", num_mutations="num_mutations",
                       variant="variant", score="score"):
    """directory:where to save, name of the new document, tsv file to split, split size/ration"""
    original = pd.read_csv(tsv_to_split, delimiter=pd_del)
    train_dataset = []
    test_dataset = []
    column_names = []
    all_mutations = np.unique(original[num_mutations])
    num_diff_mutations = len(all_mutations)
    for i in all_mutations:
        file_i = original[original[num_mutations] == i]
        file_i = file_i[[variant, num_mutations, score]]
        if isinstance(split, float):
            pre_i = file_i.sample(frac=split)
        elif isinstance(split, int):
            if file_i.shape[0] < split:
                pre_i = pd.DataFrame()
            else:
                pre_i = file_i.sample(n=int(split / num_diff_mutations))
        else:
            print("invalid argument for split")
            return
        remain = file_i.drop(pre_i.index)
        train_dataset += pre_i.values.tolist()
        test_dataset += remain.values.tolist()
        if len(column_names) == 0:
            column_names += remain.columns.tolist()
    train_dataset = pd.DataFrame(train_dataset, columns=column_names)
    test_dataset = pd.DataFrame(test_dataset, columns=column_names)
    pd.DataFrame.to_csv(train_dataset, directory + "/" + "_train_" + name + ".tsv", sep="\t", index=False)
    pd.DataFrame.to_csv(test_dataset, directory + "/" + "_test_" + name + ".tsv", sep="\t", index=False)


if __name__ == "__main__":
    pdb_ex = "datasets/pab1_rosetta_model.pdb"
    protein = "pab1"
    sequence = seq_dict()[protein]
    hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
    cm_converted, ia_converted, mat_index = data_generator_vals(sequence)

    _, factor, comb_bool = atom_interaction_matrix_d(pdb_ex, dist_th=15, plot_matrices=False)
    check_structure(pdb_ex, comb_bool, list(sequence))

    dg_vals_pl = {'wild_type_seq': sequence,
                  'interaction_matrix': comb_bool,
                  'dim': comb_bool.shape,
                  'n_channels': 5,
                  'batch_size': 1,
                  'first_ind': 126,  # 126  1
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

    create_dataset_tsv("split_datasets", "bgl3", "datasets/bgl3.tsv", 0.8)
