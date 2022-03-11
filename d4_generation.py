import numpy as np
import tensorflow as tf
from tensorflow import keras
from timeit import default_timer as timer
from datetime import datetime

import os
import sys
import warnings
import logging
import gc

from d4_utils import create_folder, log_file, hydrophobicity, h_bonding, charge, sasa, side_chain_length, clear_log
from d4_stats import validate, validation, pearson_spearman
from d4_split import split_inds, create_split_file
from d4_interactions import atom_interaction_matrix_d, hydrophobicity_matrix, hbond_matrix, charge_matrix, \
    interaction_area, clashes, mutate_sequences, check_structure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=sys.maxsize)


def data_generator_vals(wt_seq):
    """returns values/ numpy arrays based on the wt_seq for the DataGenerator\n
        :parameter
            wt_seq: wild type sequence as list eg ['A', 'V', 'L']\n
        :returns
            - hm_pos_vals: ndarray of int\n
              values for interactions with valid hydrogen bonding partners\n
            - ch_good_vals: ndarray of float\n
              values representing +/- charge pairs\n
            - ch_mid_vals: ndarray of float\n
              values representing +/n or -/n charge pairs\n
            - ch_bad_vals: ndarray of float\n
              values representing +/+ or -/- charge pairs\n
            - hp_norm: float\n
              max value possible for hydrophobicity interactions\n
            - ia_norm: float\n
              max value possible for interaction ares interactions\n
            - hm_converted: ndarray of float\n
              wt_seq converted into hydrogen bonding values
            - hp_converted: ndarray of float\n
              wt_seq converted into hydrophobicity values
            - cm_converted: ndarray of float\n
              wt_seq converted into charge values
            - ia_converted: ndarray of float\n
              wt_seq converted into SASA values
            - mat_index: 2D ndarray of float\n
              symmetrical index matrix\n
            - cl_converted: ndarray of float\n
              wt_seq converted into side chain length values
            - cl_norm: float\n
              max value possible for interaction ares interactions\n"""

    hm_pos_vals = np.asarray([2, 3, 6, 9])

    ch_good_vals = np.asarray([-1., 4.])
    ch_mid_vals = np.asarray([-2., 2.])
    ch_bad_vals = np.asarray([1.])

    h_vals = list(hydrophobicity.values())
    hp_norm = np.abs(max(h_vals) - min(h_vals))
    ia_norm = max(list(sasa.values())) * 2
    cl_norm = 2 * max(side_chain_length.values())  # + dist_thr

    hm_converted = np.asarray(list(map(h_bonding.get, wt_seq)))
    hp_converted = np.asarray(list(map(hydrophobicity.get, wt_seq)))
    cm_converted = np.asarray(list(map(charge.get, wt_seq)))
    ia_converted = np.asarray(list(map(sasa.get, wt_seq)))
    cl_converted = np.asarray(list(map(side_chain_length.get, wt_seq)))

    wt_len = len(wt_seq)
    mat_size = wt_len * wt_len
    pre_mat_index = np.arange(mat_size).reshape(wt_len, wt_len) / (mat_size - 1)
    pre_mat_index = np.triu(pre_mat_index)
    mat_index = pre_mat_index + pre_mat_index.T - np.diag(np.diag(pre_mat_index))
    np.fill_diagonal(mat_index, 0)

    return hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
           cm_converted, ia_converted, mat_index, cl_converted, cl_norm


class DataGenerator(keras.utils.Sequence):
    """
    Generates n_channel x n x n matrices to feed them as batches to a network where n denotes len(wild type sequence)\n
    modified after 'https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'\n
    ...\n
    Attributes:\n
    - features: list of str\n
      features that should be encoded ec ['A2S,E3R' 'T6W']\n
    - labels: list of int / float\n
      the corresponding labels to the features\n
    - interaction_matrix: 2D ndarray of bool\n
      boolean matrix whether residues interact or not\n
    - dim: tuple\n
      dimensions of the matrices (len(wt_seq) x len(wt_seq))\n
    - n_channels: int\n
      number of matrices used\n
    - batch_size: int\n
      Batch size (if 1 gradient gets updated after every sample in training)\n
    - first_ind: int\n
      index of the start of the protein sequence\n
    - hm_converted: ndarray of floats\n
      wt sequence h-bonding encoded\n
    - hm_pos_vals: ndarray of ints\n
      valid values for h-bonding residues\n
    - factor: 2D ndarray of floats\n
      1 - norm(distance) for all residues in the interaction matrix\n
    - hp_converted: ndarray of floats\n
      wt sequence hydrophobicity encoded\n
    - hp_norm: int or float\n
      max possible value for hydrophobicity change\n
    - cm_converted: ndarray of floats\n
      wt sequence charge encoded\n
    - ch_good_vals: int\n
      values for +-, nn interactions\n
    - ch_mid_vals: int\n
      values for n+, n- interactions\n
    - ch_bad_vals: int\n
      values for --, ++ interactions\n
    - ia_converted: ndarray of floats\n
      wt sequence interaction area encoded\n
    - ia_norm: float\n
      max value for interaction area change\n
    - mat_index: 2D ndarray of ints\n
      symmetrical index matrix (for adjacency matrix) that represents the position of each interaction in the matrices
    - cl_converted: ndarray of floats\n
      wild type sequence clash encoded
    - cl_norm: float\n
      normalization value for the clash matrix
    - dist_mat 2D ndarray of floats\n
      ture distances between all residues
    - dist_th
      maximum distance to be counted as interaction
    - shuffle: bool, (optional - default True)\n
      if True data gets shuffled after every epoch\n
    - train: bool, (optional - default True\n
      if True Generator returns features and labels (use turing training) else only features\n
    """

    def __init__(self, features, labels, interaction_matrix, dim, n_channels, batch_size, first_ind,
                 hm_converted, hm_pos_vals, factor, hp_converted, hp_norm, cm_converted, ch_good_vals, ch_mid_vals,
                 ch_bad_vals, ia_converted, ia_norm, mat_index, cl_converted, cl_norm, dist_mat, dist_th, shuffle=True,
                 train=True):
        self.features, self.labels = features, labels
        self.interaction_matrix = interaction_matrix
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
        self.cl_converted = cl_converted
        self.cl_norm = cl_norm
        self.dist_mat = dist_mat
        self.dist_th = dist_th
        self.shuffle = shuffle
        self.train = train

    def __len__(self):
        """number of batches per epoch"""
        return int(np.floor(len(self.features) / self.batch_size))  # ceil before

    def __getitem__(self, idx):
        """Generate one batch of data"""
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

    def __batch_variants(self, features_to_encode, corresponding_labels):
        """creates interaction matrices of variants for a batch"""
        batch_features = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_labels = np.empty(self.batch_size, dtype=float)

        for ci, i in enumerate(features_to_encode):
            # hydrogen bonging
            cur_hb = mutate_sequences(self.hm_converted, i, h_bonding, self.first_ind)
            part_hb = hbond_matrix(self.interaction_matrix, cur_hb, self.hm_pos_vals) * self.factor

            # hydrophobicity
            cur_hp = mutate_sequences(self.hp_converted, i, hydrophobicity, self.first_ind)
            part_hp = hydrophobicity_matrix(self.interaction_matrix, cur_hp, self.hp_norm) * self.factor

            # charge
            cur_cm = mutate_sequences(self.cm_converted, i, charge, self.first_ind)
            part_cm = charge_matrix(self.interaction_matrix, cur_cm, self.ch_good_vals, self.ch_mid_vals,
                                    self.ch_bad_vals) * self.factor
            # interaction area
            cur_ia = mutate_sequences(self.ia_converted, i, sasa, self.first_ind)
            part_ia = interaction_area(self.interaction_matrix, self.ia_converted, cur_ia,
                                       self.ia_norm) * self.factor

            # clashes
            cur_cl = mutate_sequences(self.cl_converted, i, side_chain_length, self.first_ind)
            part_cl = clashes(self.interaction_matrix, self.cl_converted, cur_cl, self.cl_norm, self.dist_mat,
                              dist_thr=self.dist_th) * self.factor

            # interaction position
            position = self.mat_index * self.factor

            batch_features[ci] = np.stack((part_hb, part_hp, part_cm, part_ia, part_cl, position), axis=2)
            batch_labels[ci] = corresponding_labels[ci]
        return batch_features, batch_labels


class SaveToFile(keras.callbacks.Callback):
    """writes training stats in a temp file
     ...\n
    Attributes:\n
    - features: str\n
      path where the temp.csv file should be saved
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        log_string = "{},{:0.4f},{:0.4f}".format(str(epoch), logs["loss"], logs["val_loss"])
        log_file(self.filepath, write_str=log_string, optional_header="epoch,loss,mae,val_loss,val_mae")


class CustomPrint(keras.callbacks.Callback):
    """prints custom stats during training\n
    ...\n
    Attributes:\n
    - epoch_print: int, (optional - default 1)\n
      interval at which loss and the change in loss should be printed\n
    - epoch_stat_print: int, (optional - default 10)\n
      interval at which best train epoch, the best validation epoch and the difference in the loss between them
      should be printed\n
    """

    def __init__(self, epoch_print=1, epoch_stat_print=10):
        self.epoch_print = epoch_print
        self.best_loss = np.Inf
        self.bl_epoch = 0
        self.best_val_loss = np.Inf
        self.bvl_epoch = 0
        self.latest_loss = 0.
        self.latest_val_loss = 0.
        self.epoch_stat_print = epoch_stat_print

    def on_epoch_end(self, epoch, logs=None):
        cur_loss = logs["loss"]
        cur_val_loss = logs["val_loss"]

        if epoch % self.epoch_print == 0:
            print("Epoch {} -- loss: {:0.4f}   val loss: {:0.4f}  --  loss change: {:0.4f}   val loss change: {:0.4f}\n"
                  .format(str(epoch), cur_loss, cur_val_loss, cur_loss - self.latest_loss,
                          cur_val_loss - self.latest_val_loss))

        self.latest_loss = cur_loss
        self.latest_val_loss = cur_val_loss

        if cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.bl_epoch = epoch
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.bvl_epoch = epoch

        if epoch % self.epoch_stat_print == 0 and epoch > 0:
            d = np.abs(self.best_loss - self.best_val_loss)
            if d != 0. and self.best_val_loss != 0.:
                dp = (d / self.best_val_loss) * 100
            else:
                dp = np.nan
            print("Best train epoch: {}\nBest validation epoch: {}\ndelta: {:0.4f} (equals {:0.2f}% of val loss)\n"
                  .format(str(self.bl_epoch), str(self.bvl_epoch), d, dp))

    def on_train_end(self, logs=None):
        print("overall best epochs")
        print("best training epoch: {} with a loss of {:0.4f}".format(str(self.bl_epoch), self.best_loss))
        print("best validation epoch: {} with a loss of {:0.4f}\n".format(str(self.bvl_epoch), self.best_val_loss))


class ClearMemory(keras.callbacks.Callback):
    """clears garbage collection and clears session after each epoch\n
        ...\n
        Attributes:\n
        None\n
        """

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def run_all(architecture_name, model_to_use, optimizer, tsv_file, pdb_file, wt_seq, number_mutations, variants, score,
            dist_thr, channel_num, max_train_mutations, training_epochs, test_num, first_ind, r_seed=None,
            deploy_early_stop=True, es_monitor="val_loss", es_min_d=0.01, es_patience=20, es_mode="auto",
            es_restore_bw=True, load_trained_model=None, batch_size=64, save_fig=None, show_fig=False,
            write_to_log=True, silent=False, extensive_test=False, save_model=False, load_trained_weights=None,
            no_nan=True, settings_test=False, p_dir="", split_def=None, validate_training=False, lr=0.001,
            transfer_conv_weights=None, train_conv_layers=False, write_temp=False, split_file_creation=False,
            use_split_file=None):
    """ runs all functions to train a neural network\n
    :parameter\n
    - architecture_name: str\n
      name of the architecture\n
    - model_to_use: function object\n
      function that returns the model\n
    - optimizer: Optimizer object
      optimizer to be used\n
    - tsv_file: str\n
      path to tsv file containing dms data of the protein of interest\n
    - pdb_file: str\n
      path to pdb file containing the structure\n
    - wt_seq: str\n
      wt sequence of the protein of interest\n
    - number_mutations: str\n
      how the number of mutations column is named\n
    - variants:str\n
      name of the variant column\n
    - score: str\n
      name of the score column\n
    - dist_thr: int or float\n
      threshold distances between any side chain atom to count as interacting\n
    - channel_num: int\n
      number of channels\n
    - max_train_mutations: int or None\n
      int specifying maximum number of mutations per sequence to be used for training\n
      None to use all mutations for training\n
    - training_epochs: int\n
      number of epochs used for training the model\n
    - test_num: int\n
      number of samples for the test after the model was trained\n
    - first_ind: int\n
      offset of the start of the sequence (when sequence doesn't start with residue 0)
    - r_seed: None, int, (optional - default None)\n
      numpy random seed\n
    - deploy_early_stop: bool, (optional - default True)\n
      whether early stop during training should be enabled (Ture) or not (False)\n
            - es_monitor: str, (optional - default 'val_loss')\n
              what to monitor to determine whether to stop the training or not\n
            - es_min_d: float, (optional - default 0.01)\n
              min_delta - min difference in es_monitor to not stop training\n
            - es_patience: int, (optional - default 20)\n
              number of epochs the model can try to get a es_monitor > es_min_d before stopping\n
            - es_mode: str, (optional - default 'auto')\n
              direction of quantity monitored in es_monitor\n
            - es_restore_bw: bool, (optional - default True)\n
              True stores the best weights of the training - False stores the last\n
    - batch_size: int, (optional - default 64)\n
      after how many samples the gradient gets updated\n
    - load_trained_model: str or None, (optional - default None)\n
      path to an already trained model\n
    - save_fig: None, (optional - default None)\n
      any to save figures in result_path\n
    - show_fig: bool, (optional - default False)\n
      True to show figures\n
    - write_to_log: bool, (optional - default True)\n
      if True writes all parameters used in the log file - !should be always enabled!\n
    - silent: bool, (optional - default False)\n
      True to print stats in the terminal\n
    - extensive_test: bool, (optional - default False):
      if True more test are done and more detailed plots are created
    - save_model: bool, (optional - default False)\n
      Ture to saves the model\n
    - load_trained_weights: str or None, (optional - default None)\n
      path to model of who's weights should be used None if it shouldn't be used\n
    - no_nan: bool, (optional - default True)\n
      True terminates training on nan\n
    - settings_test: bool, (optional - default False)\n
      Ture doesn't train the model and only executes everything of the function that is before fit
    - p_dir: str, (optional - default '')\n
      path to where the results, figures and log_file should be saved
    - split_def: list of int/float or None, (optional - default None)\n
      specifies the split for train, tune, test indices\n
            - float specifies fractions of the whole dataset
              eg [0.25, 0.25, 0.5] creates a train and tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
            - int specifies the different number of samples per dataset
              eg [50,50,100] leads to a train and a tune dataset with 50 entries each and a test dataset of 100
              if the whole dataset contains 200 entries\n
    - validate_training: bool, (optional - default False)\n
      if True validation of the training will be performed
    - lr: float, (optional - default 0.001)\n
      learning rate (how much the weights can change during an update)
    - transfer_conv_weights: str or None, (optional - default None)\n
      path to model who's weights of it's convolution layers should be used for transfer learning (needs to have the
      same architecture for the convolution part as the newly build model (model_to_use)\n
    - train_conv_layers: bool, (optional - default False)\n
      if True convolution layers are trainable - only applies whe transfer_conv_weights is not None\n
    - write_temp: bool, (optional - default False)\n
      if True writes mae and loss of each epoch to the temp.csv in result_files
    - split_file_creation: bool, (optional - default False)\n
      if True creates a directory containing train.txt, tune.txt and test.txt files that store the indices of the
      rows used from the tsv file\n
    - use_split_file: None or str, (optional - default None)\n
      if not None this needs the file_path to a directory containing splits specifying
      the 'train', 'tune', 'test' indices - these files need to be named 'train.txt', 'tune.txt' and 'test.txt'
      otherwise splits according to split_def will be used\n
    :return\n
        None\n
    """
    try:
        # dictionary with argument names as keys and the input as values
        arg_dict = locals()

        # path of the directory where results are stored
        result_dir = os.path.join(p_dir, "result_files")
        # path where the temp_file is located
        temp_path = os.path.join(result_dir, "temp.csv")
        # path where the log_file is located
        log_file_path = os.path.join(result_dir, "log_file.csv")

        # create result dir if it doesn't exist
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        # getting the proteins name
        p_name = os.path.split(tsv_file)[1].split(".")[0]

        # creating a "unique" name for protein
        time_ = str(datetime.now().strftime("%d_%m_%Y_%H%M%S")).split(" ")[0]
        name = "{}_{}".format(p_name, time_)
        print(name)

        # clear temp file from previous content or creates it if it doesn't exist
        clear_log(temp_path, name + "\n")

        # clear error.log from previous run
        clear_log(os.path.join(result_dir, "error.log"))

        # resets all state generated by keras
        tf.keras.backend.clear_session()
        if not write_to_log:
            warnings.warn("Write to log file disabled - not recommend behavior", UserWarning)

        # set random seed
        if r_seed is not None:
            np.random.seed(r_seed)
            tf.random.set_seed(r_seed)

        # creates a directory where plots will be saved
        if save_fig is not None:
            result_path = create_folder(result_dir, name)
            if save_fig is not None:
                save_fig = result_path

        if not settings_test:
            # writes used arguments to log file
            if write_to_log:
                header = "name," + ",".join(list(arg_dict.keys())) + ",training_time_in_min"
                prep_values = []
                for i in list(arg_dict.values()):
                    if type(i) == list:
                        try:
                            prep_values += ["".join(i)]
                        except TypeError:
                            prep_values += ["".join(str(i)).replace(",", "_").replace(" ", "")]
                    else:
                        prep_values += [str(i)]
                values = name + "," + ",".join(prep_values) + ",nan"
                log_file(log_file_path, values, header)

        starting_time = timer()
        # creating a list of the wt sequence string e.g. 'AVL...'  -> ['A', 'V', 'L',...]
        wt_seq = list(wt_seq)

        # split dataset
        ind_dict, data_dict = split_inds(file_path=tsv_file, variants=variants, score=score,
                                         number_mutations=number_mutations, split=split_def,
                                         split_file_path=use_split_file, test_name="stest")

        # Create files with the corresponding indices of the train, tune and test splits
        if split_file_creation:
            create_split_file(p_dir=result_dir, name=name, train_split=ind_dict["train"], tune_split=ind_dict["tune"],
                              test_split=ind_dict["test"])

        # data to train on
        train_data = data_dict["train_data"]
        train_labels = data_dict["train_labels"]
        train_mutations = data_dict["train_mutations"]

        # data to validate during training
        test_data = data_dict["tune_data"]
        test_labels = data_dict["tune_labels"]
        test_mutations = data_dict["tune_mutations"]

        # data the model has never seen
        unseen_data = data_dict["test_data"]
        unseen_labels = data_dict["test_labels"]
        unseen_mutations = data_dict["test_mutations"]

        # possible values and encoded wt_seq (based on different properties) for the DataGenerator
        hm_pos_vals, ch_good_vals, ch_mid_vals, ch_bad_vals, hp_norm, ia_norm, hm_converted, hp_converted, \
        cm_converted, ia_converted, mat_index, cl_converted, cl_norm = data_generator_vals(wt_seq)

        # distance-, factor- and interaction matrix
        dist_m, factor, comb_bool = atom_interaction_matrix_d(pdb_file, dist_th=dist_thr, plot_matrices=show_fig)

        # checks whether sequence in the pdb and the wt_seq match
        check_structure(pdb_file, comb_bool, wt_seq)

        # neural network model function
        model = model_to_use(wt_seq, channel_num)

        # load weights to model
        if load_trained_weights is not None:
            old_model = keras.models.load_model(load_trained_weights)
            model.set_weights(old_model.get_weights())

        # loads a model defined in load_trained_model
        if load_trained_model is not None:
            model = keras.models.load_model(load_trained_model)

        # load weights of a models convolutional part to a model that has a convolution part with the same architecture
        # but maybe a different/ not trained classifier
        if transfer_conv_weights is not None:
            # loads model and its weights
            trained_model = keras.models.load_model(transfer_conv_weights)
            temp_weights = [layer.get_weights() for layer in trained_model.layers]

            # which layers are conv layers (or not dense or flatten since these are sensitive to different input size)
            transfer_layers = []
            for i in range(len(trained_model.layers)):
                if i > 0:
                    l_name = trained_model.layers[i].name
                    if any(["dense" in l_name, "flatten" in l_name]):
                        pass
                    else:
                        transfer_layers += [i]

            # Transfer weights to new model
            for i in transfer_layers:
                model.layers[i].set_weights(temp_weights[i])
                if train_conv_layers is False:
                    model.layers[i].trainable = False

            # summary of the new model
            model.summary()

        model.compile(optimizer(learning_rate=lr), loss="mean_absolute_error", metrics=["mae"], run_eagerly=True)

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

        # save mae and loss to temp file
        if write_temp:
            all_callbacks += [SaveToFile(temp_path)]

        # clear Session after each epoch
        # all_callbacks += [ClearMemory()]

        # custom stats print
        all_callbacks += [CustomPrint(epoch_print=1, epoch_stat_print=10)]

        # parameters for the DatGenerator
        params = {'interaction_matrix': comb_bool,
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
                  'cl_converted': cl_converted,
                  'cl_norm': cl_norm,
                  'dist_mat': dist_m,
                  'dist_th': dist_thr,
                  'shuffle': True,
                  'train': True}

        # DataGenerator for training and the validation during training
        training_generator = DataGenerator(train_data, train_labels, **params)
        validation_generator = DataGenerator(test_data, test_labels, **params)

        # create test data for the test_generator
        if len(unseen_mutations) > 0:
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

        test_params = {'interaction_matrix': comb_bool,
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
                       'cl_converted': cl_converted,
                       'cl_norm': cl_norm,
                       'dist_mat': dist_m,
                       'dist_th': dist_thr,
                       'shuffle': False,
                       'train': False}

        test_generator = DataGenerator(t_data, np.zeros(len(t_labels)), **test_params)

        if not settings_test:
            # training
            history = model.fit(training_generator, validation_data=validation_generator, epochs=training_epochs,
                                use_multiprocessing=True, workers=12, callbacks=[all_callbacks], verbose=0)

            end_time = timer()

            # adds training time to result_files and replaces the nan time
            log_f = open(log_file_path, "r")
            prev_log = log_f.readlines()
            log_f.close()
            log_cont_len = len(prev_log)
            w_log = open(log_file_path, "w+")
            for ci, i in enumerate(prev_log):
                if len(prev_log) > 1:
                    if log_cont_len - ci == 1:
                        loi = i.strip().split(",")
                        loi[-1] = str(np.round((end_time - starting_time) / 60, 0))
                        w_log.write(",".join(loi) + "\n")
                    else:
                        w_log.write(i)
            w_log.close()

            # saves model in result path
            if save_model:
                model.save(create_folder(result_dir, name), name)

            # training and validation plot of the training
            if validate_training:
                try:
                    val_val, _, _ = validate(validation_generator, model, history, name,
                                             max_train_mutations, save_fig_v=save_fig, plot_fig=show_fig)
                except ValueError:
                    val_val = "nan"
                    log_file(file_path=os.path.join(result_dir, "log_file.csv"), write_str="nan in training history")
            else:
                val_val = "nan"

            # calculating pearsons' r and spearman r for the test dataset
            try:
                mae, pearsonr, pp, spearmanr, sp = pearson_spearman(model, test_generator, t_labels)
            except ValueError:
                mae, pearsonr, pp, spearmanr, sp = "nan", "nan", "nan", "nan", "nan"

            print("{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n{:<12s}{:0.4f}\n"
                  .format("MAE", mae, "PearsonR", pearsonr, "PearsonP", pp, "SpearmanR", spearmanr, "SpearmanP", sp))

            # creating more detailed plots
            if extensive_test:
                validation(model=model, generator=test_generator, labels=t_labels, v_mutations=t_mutations,
                           p_name=p_name, test_num=test_num, save_fig=save_fig, plot_fig=show_fig, silent=silent)

            # data for the result file
            result_string = ",".join([name, architecture_name, str(len(train_data)), str(len(test_data)),
                                      str(np.round(mae, 4)), str(np.round(pearsonr, 4)), str(np.round(pp, 4)),
                                      str(np.round(spearmanr, 4)), str(np.round(sp, 4))])
            # writing results to the result file
            log_file(os.path.join(result_dir, "results.csv"), result_string,
                     "name,architecture,train_data_size,test_data_size,mae,pearson_r,pearson_p,spearman_r,spearman_p")
        
        gc.collect()
        del model

    except Exception as e:
        # writing exception to error.log
        result_dir = os.path.join(p_dir, "result_files")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt='%d/%m/%Y %I:%M:%S %p',
                            handlers=[logging.FileHandler(os.path.join(result_dir, "error.log")),
                                      logging.StreamHandler(sys.stdout)])
        logging.exception(e)


if __name__ == "__main__":
    pass
