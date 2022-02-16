import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import os
from timeit import default_timer as timer
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from d4batch_distance_functions import protein_settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""## load data"""

path_to_file = "avgfp_augmentation_1/pre_training_avgfp.tsv"

"""## read data"""

# reading the tsv data file with format seqID sequence score
raw_data = pd.read_csv(path_to_file, delimiter="\t")

# getting the proteins name
if "/" in path_to_file:
    p_name = path_to_file.strip().split("/")[-1].split(".")[0]
    if "_" in p_name:
        p_name = p_name.split("_")[-1]
else:
    p_name = path_to_file.strip().split(".")[0]
    if "_" in p_name:
        p_name = p_name.split("_")[-1]

"""##columns to choose"""

print()
print(" - ".join(raw_data.columns.tolist()))
print()
# check whether there are multiple columns where a score is listed
sc = 0
for i in list(raw_data):
    if "score" in i:
        sc += 1
if sc > 1:
    print("*** multiple scores are given - choose appropriate one ***")
    print()
print("raw data length:", len(raw_data))
print()
print("protein name:", p_name)

"""## parameters"""

protein_name = p_name
protein_attributes = protein_settings(protein_name)

# how the number of mutations column is named
number_mutations = protein_attributes["number_mutations"]
# name of the variant column
variants = protein_attributes["variants"]
# name of the score column
score = protein_attributes["score"]
wt_seq = list(protein_attributes["sequence"])

save_fig = False
show_fig = False

# maximum number of mutations per sequence to be used for training (None to use all mutations for training)
max_train_mutations = None
# how much of the data should be used for training
train_split = 0.8
# number of epochs used for training the model
training_epochs = 200  # 200
# batch size
batch_size_ex = 32  # 32 64
# number of test after the model was trained
test_num = 15000
# whether to save the model or not
save_model = False
# compute integrated gradients
gradients = False
# numpy random seed
r_seed = 1
# plot how different scheduler function look over epochs
plot_scheduler = False
# plot model architecture
plot_model = False
# load an already trained model
load_model = False
# creates a text file with all settings
create_settings_file = True
# terminate training if nan occurred
no_nan = True
# early stop parameters , min_delta, patience, mode, restore_best_weights
deploy_early_stop = True
# what to monitor to determine whether to stop the training or not
es_monitor = "val_loss"
# min_delta min difference in es_monitor to not stop training
es_min_d = 0.01
# number of epochs the model can try to get a es_monitor > es_min_d before stopping
es_patience = 30
# direction of quantity monitored in es_monitor
es_mode = "auto"
# True stores the best weights of the training - False stores the last
es_restore_bw = True
# True runs everything but no training
test_settings = False

time_ = str(datetime.now().strftime("%d_%m_%Y_%H%M")).split(" ")[0]
name = "{}_{}".format(p_name, time_)

"""## data compatibility check"""
starting_time = timer()

"""
# checking each variant whether there is a mutation with an "*" and removing it
rd_bool = []
for i in raw_data[variants]:
    rd_bool += ["*" not in i]
raw_data = raw_data[rd_bool]
print("*** {} rows had to be excluded due to incompatibility ***".format(np.sum(np.invert(rd_bool))))
"""
"""## mutation frequency"""

# all variants in a list
v = raw_data[variants].tolist()
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
# checking the indexing of the sequence
first_ind = int(pro_seq_sorted[0][1])
if first_ind != 1:
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
            print(
                "*** residues between {} and {} not mutated***".format(str(pro_seq_inds[i]), str(pro_seq_inds[i + 1])))

if gap_count != len(wt_seq) - len(pro_seq_sorted):
    print("\n")
    print("*** wt sequence doesn't match the sequence reconstructed from the mutation file ***")
    print("    Sequence constructed from mutations:\n   ", "".join(pro_seq_sorted[:, 0]))
elif gap_count > 0:
    fill = pro_seq_inds.copy().astype(object)
    offset = 0
    for i in gaps:
        for j in i:
            fill = np.insert(fill, j - offset, "_")
            offset += 1

    under_fill = []
    rec_seq = []
    for i in fill:
        if i == "_":
            rec_seq += ["-"]
            under_fill += ["*"]
        else:
            rec_seq += [wt_seq[int(i) - 1]]
            under_fill += [" "]

    r_seq_str = "".join(rec_seq)
    w_seq_str = "".join(wt_seq)
    uf = "".join(under_fill)
    print()
    print("reconstructed sequence\nwild type sequence\ngap indicator\n")
    for i in range(0, len(wt_seq), 80):
        print(r_seq_str[i:i + 80])
        print(w_seq_str[i:i + 80])
        print(uf[i:i + 80])
        print()

# histogram for how often a residue site was part of a mutation
fig = plt.figure(figsize=(10, 6))
plt.hist(x=np.asarray(pre_seq)[:, 1].astype(int), bins=np.arange(first_ind, len(pro_seq) + 1 + first_ind),
         color="forestgreen")
plt.xlabel("residue index")
plt.ylabel("number of mutations")
plt.title(name)
if save_fig:
    plt.savefig("mutation_histogram_" + name)
if show_fig:
    plt.show()

"""## data split"""

np.random.seed(r_seed)
var_and_score = np.asarray(raw_data[[variants, score, number_mutations]])
np.random.shuffle(var_and_score)
if max_train_mutations is None:
    border = int(len(var_and_score) * train_split)
    train_data = var_and_score[:, 0][:border]
    train_labels = var_and_score[:, 1][:border]
    train_mutations = var_and_score[:, 2][:border]

    test_data = var_and_score[:, 0][border:]
    test_labels = var_and_score[:, 1][border:]
    test_mutations = var_and_score[:, 2][border:]

    print("train data size:", train_data.shape)
    print("test data size:", test_data.shape)

    unseen_mutations = None
    unseen_labels = None
    unseen_data = None
else:
    vs_un = var_and_score[var_and_score[:, 2] <= max_train_mutations]
    vs_up = var_and_score[var_and_score[:, 2] > max_train_mutations]
    border = int(len(vs_un) * train_split)

    train_data = vs_un[:, 0][:border]
    train_labels = vs_un[:, 1][:border]
    train_mutations = vs_un[:, 2][:border]

    test_data = vs_un[:, 0][border:]
    test_labels = vs_un[:, 1][border:]
    test_mutations = vs_un[:, 2][border:]

    unseen_data = vs_up[:, 0]
    unseen_labels = vs_up[:, 1]
    unseen_mutations = vs_up[:, 2]
    print("train data size:", train_data.shape)
    print("test data size:", test_data.shape)
    print("unseen data size:", unseen_data.shape)

"""## processing"""

aa_token = dict(
    zip(['*', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
        np.arange(21)))

wt_seq_encoded = list(map(aa_token.get, wt_seq))

"""## sequence data generator """


class SequenceDataGenerator(keras.utils.Sequence):

    def __init__(self, features, labels, wts, ecd, dim, batch_size, shuffle=True, train=True):
        self.features, self.labels = features, labels
        self.wts = wts
        self.ecd = ecd
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train

    def __len__(self):
        return int(np.ceil(len(self.features) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size:(idx + 1) *
                                                      self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) *
                                                    self.batch_size]

        f, l = self.__batch_variants(batch_x, batch_y)
        if self.train:
            return f, l
        else:
            return f

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idx = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.idx)

    def get_variation(self, var, wt_enc, enc_dict):
        """converting the sequence with its variation into an encoded form
        var: variation(s) like ['A1S', 'L5K']
        wt_ecn: encoded wt seq (2D numpy array)
        enc_dict: dict that was used to encode the wt seq"""
        base_seq = wt_enc.copy()
        vars = var.strip().split(",")
        for i in vars:
            base_seq[int(i[1:-1]) - first_ind] = enc_dict[i[-1]]
        return base_seq
        # return np.c_[base_seq, positions]

    def __batch_variants(self, features_to_encode, corresponding_labels):
        """creates encoded variants for a batch"""
        batch_features = np.empty((self.batch_size, self.dim))
        batch_labels = np.empty((self.batch_size), dtype=float)
        for ci, i in enumerate(features_to_encode):
            batch_features[ci] = self.get_variation(i, self.wts, self.ecd)
            batch_labels[ci] = corresponding_labels[ci]
        return batch_features, batch_labels


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


vocab_size = 21  # Only consider the top 20k words 21
maxlen = len(wt_seq)  # Only consider the first 200 words of each movie review

embed_dim = 32  # Embedding size for each token  24
num_heads = 6  # Number of attention heads  8
ff_dim = 64  # Hidden layer size in feed forward network inside transformer  64

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(256, activation="leaky_relu")(x)  # 256
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="leaky_relu")(x)  # 128
x = layers.Dropout(0.2)(x)
for i in range(2):
    x = layers.Dense(1024, activation="leaky_relu")(x)
x = layers.Dense(64, activation="leaky_relu")(x)  # 64
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="leaky_relu")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(tfa.optimizers.AdamW(weight_decay=0.0001, learning_rate=0.001, beta_2=0.97), loss="mean_absolute_error",
              metrics=["mae"])

all_callbacks = []
if no_nan:
    all_callbacks += [tf.keras.callbacks.TerminateOnNaN()]
# deploying early stop parameters
if deploy_early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=es_monitor,
        min_delta=es_min_d,
        patience=es_patience,
        mode=es_mode,
        restore_best_weights=es_restore_bw)
    all_callbacks += [es_callback]

# sequence training

params = {'wts': wt_seq_encoded,
          'ecd': aa_token,
          'dim': len(wt_seq),
          'batch_size': batch_size_ex,
          'shuffle': True,
          'train': True}

training_generator = SequenceDataGenerator(train_data, train_labels, **params)
validation_generator = SequenceDataGenerator(test_data, test_labels, **params)

np.random.seed = r_seed
if unseen_mutations is not None:
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

test_params = {'wts': wt_seq_encoded,
               'ecd': aa_token,
               'dim': len(wt_seq),
               'batch_size': 1,
               'shuffle': False,
               'train': False}

# test_generator = SequenceDataGenerator(t_data, t_labels, **test_params)

if not test_settings:
    history = model.fit(training_generator, validation_data=validation_generator,
                        epochs=training_epochs, use_multiprocessing=True, workers=12,
                        callbacks=[all_callbacks])
    ending_time = timer()
    print("total time used for training in minutes:", str(np.around((ending_time - starting_time)/60, 3)))

    val_data = pd.read_csv("avgfp_augmentation_1/validate_avgfp.tsv", delimiter="\t")
    t_data = np.asarray(val_data[variants])
    t_labels = np.asarray(val_data[score])
    test_generator = SequenceDataGenerator(t_data, t_labels, **test_params)
    predicted_labels = model.predict(test_generator).flatten()
    error = np.abs(predicted_labels - t_labels)
    try:
        pearson_r, pearson_r_p = scipy.stats.pearsonr(t_labels.astype(float), predicted_labels.astype(float))
        spearman_r, spearman_r_p = scipy.stats.spearmanr(t_labels.astype(float), predicted_labels.astype(float))
        print("MAE: {}\nSTD: {}\nPearson's r: {}\nPearson's r p-value:{}\nSpearman r: {}\nSpearman r p-value: {}\n".
              format(str(np.mean(error)), str(error.std()), str(pearson_r), str(pearson_r_p), str(spearman_r),
                     str(spearman_r_p)))
    except ValueError:
        print("Invalid loss")


