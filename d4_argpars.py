import argparse
import tensorflow as tf
from d4_models import simple_model, simple_model_norm, simple_model_imp, simple_stride_model_wide, \
    create_simple_model, simple_model_gap, simple_stride_model_test, shrinking_res, inception_res, \
    deeper_res, res_net, vgg, simple_longer, simple_stride_model
from d4_utils import protein_settings


def arg_dict():
    """creates a parameter dict for run_all with the use of argparse"""
    pos_models = [simple_model, simple_model_norm, simple_model_imp, simple_stride_model_wide, create_simple_model,
                  simple_model_gap, simple_stride_model_test, shrinking_res, inception_res, deeper_res, res_net, vgg,
                  simple_longer, simple_stride_model]

    pos_optimizer = [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]

    model_str = []
    for ci, i in enumerate(pos_models):
        model_str += [str(ci) + " " + str(i).split(" ")[1] + "\n"]

    opt_str = []
    for ci, i in enumerate(pos_optimizer):
        opt_str += [str(ci) + " " + str(i).split(" ")[1] + "\n"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_name", type=str, required=True, help="protein_name")
    parser.add_argument("--transfer_conv_weights", type=str, required=False, default=None, help="transfer_conv_weights")
    parser.add_argument("--train_conv_layers", type=bool, required=False, default=False, help="train_conv_layers")
    parser.add_argument("--training_epochs", type=int, required=False, default=100, help="training_epochs")
    parser.add_argument("--es_patience", type=int, required=False, default=20, help="es_patience")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help="batch_size")
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001, help="learning_rate")
    parser.add_argument("--split0", type=float, required=False, default=0.8, help="split 0")
    parser.add_argument("--split1", type=float, required=False, default=0.2, help="split 1")
    parser.add_argument("--split2", type=float, required=False, default=0.0, help="split 2")
    parser.add_argument("--architecture", type=int, required=False, default=0, help=" ".join(model_str))
    parser.add_argument("--save_model", type=bool, required=False, default=False, help="save_model")
    parser.add_argument("--settings_test", type=bool, required=False, default=False, help="settings_test")
    parser.add_argument("--dist_thr", type=float, required=False, default=20, help="dist_thr")
    parser.add_argument("--channel_num", type=int, required=False, default=6, help="channel_num")
    parser.add_argument("--load_trained_weights_path", type=str, required=False, default=None,
                        help="load_trained_weights_path")
    parser.add_argument("--load_trained_model_path", type=str, required=False, default=None,
                        help="load_trained_model_path")
    parser.add_argument("--max_train_mutations", type=int, required=False, default=None, help="max_train_mutations")
    parser.add_argument("--test_num", type=int, required=False, default=5000, help="test_num")
    parser.add_argument("--random_seed", type=int, required=False, default=None, help="random_seed")
    parser.add_argument("--save_figures", type=int, required=False, default=None, help="save_figures")
    parser.add_argument("--show_figures", type=bool, required=False, default=False, help="show_figures")
    parser.add_argument("--silent_execution", type=bool, required=False, default=True, help="silent_execution")
    parser.add_argument("--extensive_test", type=bool, required=False, default=False, help="extensive_test")
    parser.add_argument("--deploy_early_stop", type=bool, required=False, default=True, help="deploy_early_stop")
    parser.add_argument("--no_nan", type=bool, required=False, default=True, help="no_nan")
    parser.add_argument("--write_temp", type=bool, required=False, default=False, help="write_temp")
    parser.add_argument("--optimizer", type=int, required=False, default=0, help="\n".join(opt_str))
    parser.add_argument("--p_dir", type=str, required=False, default="./result_files",
                        help="p_dir")
    parser.add_argument("--split_file_creation", type=bool, required=False, default=False, help="split_file_creation")
    parser.add_argument("--use_split_file", type=str, required=False, default=None, help="use_split_file")

    args = parser.parse_args()
    protein_name = args.protein_name
    p_dir_ex = args.p_dir
    transfer_conv_weights_ex = args.transfer_conv_weights
    train_conv_layers_ex = args.train_conv_layers
    training_epochs_ex = args.training_epochs
    es_patience_ex = args.es_patience
    batch_size_ex = args.batch_size
    lr_ex = args.learning_rate

    if args.split2 > 0.:
        split_def_ex = [args.split0, args.split1, args.split2]
    else:
        split_def_ex = [args.split0, args.split1]

    architecture = pos_models[args.architecture]
    save_model_ex = args.save_model
    settings_test_ex = args.settings_test
    dist_thr_ex = args.dist_thr
    channel_num_ex = args.channel_num
    load_trained_weights_ex = args.load_trained_weights_path
    load_trained_model_ex = args.load_trained_model_path
    max_train_mutations_ex = args.max_train_mutations
    test_num_ex = args.test_num
    r_seed_ex = args.random_seed
    save_fig_ex = args.save_figures
    show_fig_ex = args.show_figures
    silent_ex = args.silent_execution
    ex_test = args.extensive_test
    deploy_early_stop_ex = args.deploy_early_stop
    no_nan_ex = args.no_nan
    write_temp_ex = args.write_temp
    protein_attributes = protein_settings(protein_name)
    architecture_name = architecture.__code__.co_name
    number_mutations_ex = protein_attributes["number_mutations"]
    variants_ex = protein_attributes["variants"]
    score_ex = protein_attributes["score"]
    wt_seq_ex = list(protein_attributes["sequence"])
    first_ind_ex = int(protein_attributes["offset"])
    tsv_ex = "./datasets/{}.tsv".format(protein_name.lower())
    pdb_ex = "./datasets/{}.pdb".format(protein_name.lower())
    opt = pos_optimizer[args.optimizer]
    split_file_creation_ex = args.split_file_creation
    use_split_file_ex = args.use_split_file

    d = {"p_dir": p_dir_ex,
         "transfer_conv_weights": transfer_conv_weights_ex,
         "train_conv_layers": train_conv_layers_ex,
         "training_epochs": training_epochs_ex,
         "es_patience": es_patience_ex,
         "batch_size": batch_size_ex,
         "lr": lr_ex,
         "split_def": split_def_ex,
         "save_model": save_model_ex,
         "settings_test": settings_test_ex,
         "dist_thr": dist_thr_ex,
         "channel_num": channel_num_ex,
         "load_trained_weights": load_trained_weights_ex,
         "load_trained_model": load_trained_model_ex,
         "max_train_mutations": max_train_mutations_ex,
         "test_num": test_num_ex,
         "r_seed": r_seed_ex,
         "save_fig": save_fig_ex,
         "show_fig": show_fig_ex,
         "silent": silent_ex,
         "extensive_test": ex_test,
         "deploy_early_stop": deploy_early_stop_ex,
         "no_nan": no_nan_ex,
         "write_temp": write_temp_ex,
         "number_mutations": number_mutations_ex,
         "variants": variants_ex,
         "score": score_ex,
         "wt_seq": wt_seq_ex,
         "first_ind": first_ind_ex,
         "tsv_file": tsv_ex,
         "pdb_file": pdb_ex,
         "architecture_name": architecture_name,
         "optimizer": opt,
         "model_to_use": architecture,
         "split_file_creation": split_file_creation_ex,
         "use_split_file": use_split_file_ex}
    return d


if __name__ == "__main__":
    pass