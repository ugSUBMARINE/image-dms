import argparse, configparser
import os

from d4_utils import protein_settings, dotdict
from d4_config import retrieve_args


def arg_dict(p_dir: str = "") -> dict:
    """creates a parameter dict for run_all with the use of argparse
    :parameter
        - p_dir:
          directory where the dataset folder are stored and all the .py files are stored
    :return
        - d:
          dictionary specifying all parameters for run_all in d4_cmd_driver.py
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        required=False,
        default=None,
        help="to use config file as arguments use the file path to the config file in"
        "this argument",
    )
    parser.add_argument(
        "-pn",
        "--protein_name",
        type=str,
        required=False,
        default=None,
        help="str: name of the protein in the protein settings file",
    )
    parser.add_argument(
        "-qn",
        "--query_name",
        type=str,
        required=False,
        default=None,
        help="str: name of the wild type sequence in the protein alignment file",
    )
    parser.add_argument(
        "-af",
        "--alignment_file",
        type=str,
        required=False,
        default=None,
        help="str: alignemt file for the protein of interest",
    )
    parser.add_argument(
        "-tw",
        "--transfer_conv_weights",
        type=str,
        required=False,
        default=None,
        help="str: file path to a suitable trained network to transfer its convolution"
        "layer weights to the new model",
    )
    parser.add_argument(
        "-tl",
        "--train_conv_layers",
        action="store_true",
        help="set flag to train the convolution layer else they are set to "
        "trainable=False",
    )
    parser.add_argument(
        "-te",
        "--training_epochs",
        type=int,
        required=False,
        default=100,
        help="int: number of training epochs",
    )
    parser.add_argument(
        "-ep",
        "--es_patience",
        type=int,
        required=False,
        default=20,
        help="number of epochs the model can try to decrease its es_monitor value for "
        "at least min_delta before",
    )
    parser.add_argument(
        "--es_monitor",
        type=str,
        required=False,
        default="val_loss",
        help="which value should be monitored for es_patience",
    )
    parser.add_argument(
        "--es_min_d",
        type=float,
        required=False,
        default=0.01,
        help="how much the es_monitor needs to change over es_patience epochs",
    )
    parser.add_argument(
        "--es_mode",
        type=str,
        required=False,
        default="auto",
        help="direction of es_monitor",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="after how many samples the gradient gets updated",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="how much the weights can change during an update",
    )
    parser.add_argument(
        "-s0",
        "--split0",
        type=float,
        required=False,
        default=0.8,
        help="size of the training data set - can be either a fraction or a number of"
        "samples",
    )
    parser.add_argument(
        "-s1",
        "--split1",
        type=float,
        required=False,
        default=0.2,
        help="size of the tune data set - can be either a fraction or a number of"
        "samples",
    )
    parser.add_argument(
        "-s2",
        "--split2",
        type=float,
        required=False,
        default=0.0,
        help="size of the test data set - can be either a fraction or a number of"
        "samples",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        default="sep_conv_mix",
        help="input the name of the model function from d4_models",
    )
    parser.add_argument(
        "-sm",
        "--save_model",
        action="store_true",
        help="set flag to save the model after training",
    )
    parser.add_argument(
        "-st",
        "--settings_test",
        action="store_true",
        help="set flag doesn't train the model and only executes everything of"
        "the function that is before model.fit()",
    )
    parser.add_argument(
        "-dt",
        "--dist_thr",
        type=float,
        required=False,
        default=20,
        help="threshold distances between any side chain atom to count as interacting",
    )
    parser.add_argument(
        "-lw",
        "--load_trained_weights_path",
        type=str,
        required=False,
        default=None,
        help="path to model of who's weights should be used None if it "
        "shouldn't be used",
    )
    parser.add_argument(
        "-lm",
        "--load_trained_model_path",
        type=str,
        required=False,
        default=None,
        help="path to an already trained model or None to not load a model",
    )
    parser.add_argument(
        "-mm",
        "--max_train_mutations",
        type=int,
        required=False,
        default=None,
        help="int specifying maximum number of mutations per sequence to be "
        "used for training -None to use all mutations for training",
    )
    parser.add_argument(
        "-tn",
        "--test_num",
        type=int,
        required=False,
        default=5000,
        help="number of samples for the test after the model was trained",
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        required=False,
        default=None,
        help="numpy and tensorflow random seed",
    )
    parser.add_argument(
        "-sf",
        "--save_figures",
        action="store_true",
        help="set flag to store plots in result_files",
    )
    parser.add_argument(
        "-pf",
        "--show_figures",
        action="store_true",
        help="set flag to show figures",
    )
    parser.add_argument(
        "-se",
        "--silent_execution",
        action="store_false",
        help="set flag to print stats in the terminal",
    )
    parser.add_argument(
        "-et",
        "--extensive_test",
        action="store_true",
        help="set flag so more tests are done and more detailed plots are created",
    )
    parser.add_argument(
        "-es",
        "--deploy_early_stop",
        action="store_false",
        help="set flag if early stop during training should be disabled",
    )
    parser.add_argument(
        "-nn",
        "--no_nan",
        action="store_false",
        help="set flag to terminate training on nan",
    )
    parser.add_argument(
        "-wf",
        "--write_temp",
        action="store_true",
        help="set flag to write mae, loss and time per epoch of each epoch to the "
        "temp.csv in result_files",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        required=False,
        default="Adam",
        help="input an optimizer name from tf.keras.optimizers",
    )
    parser.add_argument(
        "-pd",
        "--p_dir",
        type=str,
        required=False,
        default=p_dir,
        help="path to the projects content root - default=''",
    )
    parser.add_argument(
        "-fc",
        "--split_file_creation",
        action="store_true",
        help="set flag to create directory containing train.txt, tune.txt"
        " and test.txt files that store the indices of the rows used from the tsv file "
        "during training, validating and testing",
    )
    parser.add_argument(
        "-uf",
        "--use_split_file",
        type=str,
        required=False,
        default=None,
        help="if not None this needs the file_path to a directory containing "
        "splits specifying the 'train', 'tune', 'test' indices - these files need "
        "to be named 'train.txt', 'tune.txt' and 'test.txt' otherwise splits of the "
        "tsv file according to split_def will be used",
    )
    parser.add_argument(
        "-nm",
        "--number_mutations",
        type=str,
        required=False,
        default=None,
        help="how the number of mutations column is named -"
        "required when protein_name is not defined",
    )
    parser.add_argument(
        "-v",
        "--variants",
        type=str,
        required=False,
        default=None,
        help="name of the variant column - required when protein_name is not defined",
    )
    parser.add_argument(
        "-s",
        "--score",
        type=str,
        required=False,
        default=None,
        help="name of the score column - required when protein_name is not defined",
    )
    parser.add_argument(
        "-wt",
        "--wt_seq",
        type=str,
        required=False,
        default=None,
        help="wt sequence of the protein of interest eg. 'AVL...' - "
        "required when protein_name is not defined - required when protein_name "
        "is not defined",
    )
    parser.add_argument(
        "-fi",
        "--first_ind",
        type=int,
        required=False,
        default=None,
        help="offset of the start of the sequence (when sequence doesn't start "
        "with residue 0) - required when protein_name is not defined",
    )
    parser.add_argument(
        "-tp",
        "--tsv_filepath",
        type=str,
        required=False,
        default=None,
        help="path to tsv file containing dms data of the protein of interest - "
        "required when tsv file is not stored in /datasets or protein_name is not "
        "given or the file is not named protein_name.tsv",
    )
    parser.add_argument(
        "-pp",
        "--pdb_filepath",
        type=str,
        required=False,
        default=None,
        help="path to pdb file containing the structure data of the protein "
        "of interest - required when pdb file is not stored in ./datasets or "
        "protein_name is not given or the file is not named protein_name.pdb",
    )
    parser.add_argument(
        "-vt",
        "--validate_training",
        action="store_true",
        help="validates training and either shows the validation plot if "
        "show_figures flag is set or saves them if save_figures flag is set",
    )
    parser.add_argument(
        "-rb",
        "--restore_bw",
        action="store_false",
        help="set flag to not store the best weights but the weights of the last "
        "training epoch",
    )
    parser.add_argument(
        "-wl",
        "--write_to_log",
        action="store_false",
        help="set flag to not write settings usd for training to the log file -"
        "NOT recommended",
    )
    parser.add_argument(
        "-da",
        "--data_aug",
        action="store_true",
        help="set flag to use data augmentation",
    )
    parser.add_argument(
        "-cl",
        "--clear_error_log",
        action="store_true",
        help="set flag to clear error log before run",
    )
    parser.add_argument(
        "-rd",
        "--reduce",
        action="store_true",
        help="set flag to use size reducing intro layer",
    )
    parser.add_argument(
        "-j",
        "--jit_compile",
        action="store_false",
        help="set flag to not use jit_compile from tensorflow",
    )

    args = parser.parse_args()
    cf_path = args.config_file

    if cf_path is not None:
        args = dotdict(retrieve_args(cf_path))

    p_dir = args.p_dir
    if p_dir is None:
        p_dir = ""

    protein_name = args.protein_name

    if args.split2 > 0.0:
        split_def_ex = [args.split0, args.split1, args.split2]
    else:
        split_def_ex = [args.split0, args.split1]

    # check when protein name is None
    nm = args.number_mutations
    v = args.variants
    s = args.score
    wt = args.wt_seq
    fi = args.first_ind

    if protein_name is None:
        if not all(
            [
                nm is not None,
                v is not None,
                s is not None,
                wt is not None,
                fi is not None,
            ]
        ):
            raise ValueError(
                "If protein_name is not given 'number_mutations', 'variants', "
                "'score', 'wt_seq' and 'first_ind' must be given as input"
            )
        wt_seq_ex = list(wt)
        number_mutations_ex = args.number_mutations
        variants_ex = args.variants
        score_ex = args.score
        first_ind_ex = args.first_ind
    else:
        protein_attributes = protein_settings(
            protein_name, os.path.join(p_dir, "datasets/protein_settings_ori.txt")
        )
        number_mutations_ex = protein_attributes["number_mutations"]
        variants_ex = protein_attributes["variants"]
        score_ex = protein_attributes["score"]
        wt_seq_ex = list(protein_attributes["sequence"])
        first_ind_ex = int(protein_attributes["offset"])

    if args.tsv_filepath is None and protein_name is not None:
        tsv_ex = os.path.join(p_dir, "datasets", "{}.tsv".format(protein_name.lower()))
    elif args.tsv_filepath is not None:
        tsv_ex = args.tsv_filepath
    else:
        raise ValueError("Either protein_name or tsv_filepath must be given as input")

    if args.pdb_filepath is None and protein_name is not None:
        pdb_ex = os.path.join(p_dir, "datasets", "{}.pdb".format(protein_name.lower()))
    elif args.pdb_filepath is not None:
        pdb_ex = args.pdb_filepath
    else:
        raise ValueError("Either protein_name or pdb_filepath must be given as input")

    tsv_ex = os.path.join(p_dir, tsv_ex)
    pdb_ex = os.path.join(p_dir, pdb_ex)
    align_ex = None
    if args.alignment_file is not None:
        align_ex = os.path.join(p_dir, args.alignment_file)

    # checking whether the files exist
    if not os.path.isfile(tsv_ex):
        raise FileNotFoundError(
            "tsv file path is incorrect - file '{}' doesn't exist".format(str(tsv_ex))
        )
    if not os.path.isfile(pdb_ex):
        raise FileNotFoundError(
            "pdb file path is incorrect - file '{}' doesn't exist".format(str(pdb_ex))
        )
    if align_ex is not None and not os.path.isfile(align_ex):
        raise FileNotFoundError(
            "alignment file path is incorrect - file '{}' doesn't exist".format(
                str(args.alignment_file)
            )
        )

    d = {
        "p_dir": p_dir,
        "algn_path": align_ex,
        "algn_bl": args.query_name,
        "transfer_conv_weights": args.transfer_conv_weights,
        "train_conv_layers": args.train_conv_layers,
        "training_epochs": args.training_epochs,
        "es_patience": args.es_patience,
        "es_monitor": args.es_monitor,
        "es_mode": args.es_mode,
        "es_min_d": args.es_min_d,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "split_def": split_def_ex,
        "save_model": args.save_model,
        "settings_test": args.settings_test,
        "dist_thr": args.dist_thr,
        "load_trained_weights": args.load_trained_weights_path,
        "load_trained_model": args.load_trained_model_path,
        "max_train_mutations": args.max_train_mutations,
        "test_num": args.test_num,
        "r_seed": args.random_seed,
        "save_fig": args.save_figures,
        "show_fig": args.show_figures,
        "silent": args.silent_execution,
        "extensive_test": args.extensive_test,
        "deploy_early_stop": args.deploy_early_stop,
        "no_nan": args.no_nan,
        "write_temp": args.write_temp,
        "number_mutations": number_mutations_ex,
        "variants": variants_ex,
        "score": score_ex,
        "wt_seq": wt_seq_ex,
        "first_ind": first_ind_ex,
        "tsv_file": tsv_ex,
        "pdb_file": pdb_ex,
        "optimizer": args.optimizer,
        "model_to_use": args.architecture,
        "split_file_creation": args.split_file_creation,
        "use_split_file": args.use_split_file,
        "validate_training": args.validate_training,
        "es_restore_bw": args.restore_bw,
        "write_to_log": args.write_to_log,
        "daug": args.data_aug,
        "clear_el": args.clear_error_log,
        "reduce": args.reduce,
        "jit": args.jit_compile,
    }
    return d


def predict_dict(p_dir: str = "") -> dict:
    """creates a parameter dict for predictions with the use of argparse
    :parameter
        - p_dir:
          directory where the datasets are stored
    :return
        - d:
          dictionary specifying all parameters for predictions in d4_prediction.py
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-pf",
        "--protein_pdb",
        type=str,
        required=True,
        help="str: filepath to the pdb file of the protein of interest",
    )
    parser.add_argument(
        "-wt",
        "--wt_seq",
        type=str,
        required=True,
        help="str: wt sequence of the protein of interest eg. 'AVL...' - ",
    )
    parser.add_argument(
        "-va",
        "--variant_s",
        type=str,
        required=True,
        help="str: variant(s) of interest like 'A1S,R3K' for one variant or "
        "'A1S,R3K_F9I' for multiple variants",
    )
    parser.add_argument(
        "-mf",
        "--model_filepath",
        type=str,
        required=True,
        help="str: filepath to the model that was trained on data of the protein of "
        "interest",
    )
    parser.add_argument(
        "-af",
        "--alignment_file",
        type=str,
        required=False,
        default=None,
        help="str: alignment file for the protein of interest",
    )
    parser.add_argument(
        "-qn",
        "--query_name",
        type=str,
        required=False,
        default=None,
        help="str: name of the wild type sequence in the protein alignment file",
    )
    parser.add_argument(
        "-dt",
        "--dist_thr",
        type=float,
        required=False,
        default=20,
        help="float: threshold distances between any side chain atom to count as "
        "interacting",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="int: after how many samples the gradient gets updated",
    )
    parser.add_argument(
        "-fi",
        "--first_ind",
        type=str,
        required=False,
        default=None,
        help="int: offset of the start of the sequence (when sequence doesn't start "
        "with residue 0)",
    )
    args = parser.parse_args()
    # checking whether the files exist
    if not os.path.isfile(args.protein_pdb):
        raise FileNotFoundError(
            "pdb file path is incorrect - file '{}' doesn't exist".format(
                str(args.protein_pdb)
            )
        )
    if not os.path.isdir(args.model_filepath):
        raise FileNotFoundError(
            "model file path is incorrect - file '{}' doesn't exist".format(
                str(args.model_filepath)
            )
        )
    if args.alignment_file is not None and not os.path.isfile(args.alignment_file):
        raise FileNotFoundError(
            "alignment file path is incorrect - file '{}' doesn't exist".format(
                str(args.alignment_file)
            )
        )
    # creating list of variants to match the needed input for predictions
    variants = args.variant_s
    if "_" in variants:
        variants = variants.split("_")
    else:
        variants = [variants]

    pred_dict = {
        "protein_pdb": args.protein_pdb,
        "protein_seq": args.wt_seq,
        "variant_s": variants,
        "model_filepath": args.model_filepath,
        "dist_th": args.dist_thr,
        "algn_path": args.alignment_file,
        "algn_base": args.query_name,
        "batch_size": args.batch_size,
        "first_ind": args.first_ind,
    }

    return pred_dict


def pretrain_dict(p_dir: str = "") -> dict:
    """creates a parameter dict for creating pretrainging data with the use of argparse
    :parameter
        - p_dir:
          directory where the datasets are stored
    :return
        - d:
          dictionary specifying all parameters for predictions in d4_prediction.py
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-pn",
        "--p_name",
        type=str,
        required=True,
        help="name of the protein (also in the alignment file and setting file)",
    )
    parser.add_argument(
        "-pf",
        "--pdb_file",
        type=str,
        required=True,
        help="file path to the pdb file of the protein",
    )
    parser.add_argument(
        "-ap",
        "--algn_path",
        type=str,
        required=False,
        help="file path to the alignment file of the protein"
        "(can be omitted but should be used)",
    )
    parser.add_argument(
        "-pd",
        "--p_data",
        action="store_false",
        help="set flag if protein data is not in the protein_settings_ori.txt file",
    )
    parser.add_argument(
        "-pi",
        "--p_firstind",
        type=int,
        required=False,
        default=None,
        help="index of the first amino acid, overwrite p_data here if given",
    )
    parser.add_argument(
        "-ps",
        "--p_seq",
        type=str,
        required=False,
        default=None,
        help="amino acid sequence of the protein, overwrite p_data here if given",
    )
    parser.add_argument(
        "-ts",
        "--tsv_path",
        type=str,
        required=False,
        default=None,
        help="path to the tsv file with true assay scores if these variants should"
        "not be used in pretraining",
    )
    parser.add_argument(
        "-ti",
        "--testind_path",
        type=str,
        required=False,
        default=None,
        help="path to file with indices of variants that should be excluded for the"
        "pretraining score generation",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        required=False,
        default=None,
        help="path where the pseudo score tsv file should be stored - default is"
        "'datasets/pseudo_scores/P_NAME'",
    )
    parser.add_argument(
        "-a",
        "--add",
        type=str,
        required=False,
        default=None,
        help="string that should be added to the filename",
    )
    parser.add_argument(
        "-n",
        "--num_var",
        type=int,
        required=False,
        default=40000,
        help="number of variants that should be generated",
    )

    parser.add_argument(
        "-nv",
        "--name_var",
        type=str,
        required=False,
        default="variant",
        help="name of the variant column in the tsv file",
    )
    parser.add_argument(
        "-nm",
        "--name_nmut",
        type=str,
        required=False,
        default="num_mutations",
        help="name of the num_mutations column in the tsv file",
    )
    parser.add_argument(
        "-ns",
        "--name_score",
        type=str,
        required=False,
        default="score",
        help="name of the score column in the tsv file",
    )

    args = parser.parse_args()
    # checking whether the files exist
    if not os.path.isfile(args.pdb_file):
        raise FileNotFoundError(
            "pdb file path is incorrect - file '{}' doesn't exist".format(
                str(args.pdb_file)
            )
        )
    if args.algn_path is not None and not os.path.isfile(args.algn_path):
        raise FileNotFoundError(
            "alignment file path is incorrect - file '{}' doesn't exist".format(
                str(args.algn_path)
            )
        )
    if args.tsv_path is not None and not os.path.isfile(args.tsv_path):
        raise FileNotFoundError(
            "tsv file path is incorrect - file '{}' doesn't exist".format(
                str(args.tsv_path)
            )
        )
    if args.testind_path is not None and not os.path.isfile(args.testind_path):
        raise FileNotFoundError(
            "test ind file path is incorrect - file '{}' doesn't exist".format(
                str(args.testind_path)
            )
        )
    if args.out_path is not None and not os.path.isdir(args.out_path):
        raise FileNotFoundError(
            "out_path directory path is incorrect - dir '{}' doesn't exist".format(
                str(args.out_path)
            )
        )

    pret_dict = {
        "p_name": args.p_name,
        "pdb_file": args.pdb_file,
        "algn_path": args.algn_path,
        "p_data": args.p_data,
        "p_firstind": args.p_firstind,
        "p_seq": args.p_seq,
        "tsv_path": args.tsv_path,
        "testind_path": args.testind_path,
        "out_path": args.out_path,
        "add": args.add,
        "num_var": args.num_var,
        "name_var": args.name_var,
        "name_nmut": args.name_nmut,
        "name_score": args.name_score,
    }

    return pret_dict


if __name__ == "__main__":
    pass
