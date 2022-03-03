import argparse
import tensorflow as tf
from d4_models import simple_model, simple_model_norm, simple_model_imp, simple_stride_model_wide, \
    create_simple_model, simple_model_gap, simple_stride_model_test, shrinking_res, inception_res, \
    deeper_res, res_net, vgg, simple_longer, simple_stride_model
from d4_utils import protein_settings


def arg_dict():
    """creates am parameter dict for run all with the use of argparse"""
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
    parser.add_argument("--pn", type=str, required=True, help="protein name")
    parser.add_argument("--pd", type=str, required=False, default="./result_files",
                        help="p_dir")
    parser.add_argument("--tcw", type=str, required=False, default=None, help="transfer_conv_weights")
    parser.add_argument("--tcl", type=bool, required=False, default=False, help="train_conv_layers")
    parser.add_argument("--ep", type=int, required=False, default=100, help="training_epochs")
    parser.add_argument("--pa", type=int, required=False, default=20, help="es_patience")
    parser.add_argument("--bs", type=int, required=False, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="learning_rate")
    parser.add_argument("--s1", type=float, required=False, default=0.8, help="learning_rate")
    parser.add_argument("--s2", type=float, required=False, default=0.2, help="learning_rate")
    parser.add_argument("--s3", type=float, required=False, default=0.0, help="learning_rate")
    parser.add_argument("--ac", type=int, required=False, default=0, help=" ".join(model_str))
    parser.add_argument("--sm", type=bool, required=False, default=False, help="save_model")
    parser.add_argument("--st", type=bool, required=False, default=False, help="settings_test")
    parser.add_argument("--dt", type=float, required=False, default=20, help="dist_thr")
    parser.add_argument("--cn", type=int, required=False, default=6, help="channel_num")
    parser.add_argument("--ltw", type=str, required=False, default=None, help="load_trained_weights_path")
    parser.add_argument("--ltm", type=str, required=False, default=None, help="load_trained_model_path")
    parser.add_argument("--mtm", type=int, required=False, default=None, help="max_train_mutations")
    parser.add_argument("--tn", type=int, required=False, default=5000, help="test_num")
    parser.add_argument("--rs", type=int, required=False, default=None, help="random seed")
    parser.add_argument("--saf", type=int, required=False, default=None, help="save figures")
    parser.add_argument("--shf", type=bool, required=False, default=False, help="show figures")
    parser.add_argument("--se", type=bool, required=False, default=True, help="silent execution")
    parser.add_argument("--et", type=bool, required=False, default=False, help="extensive test")
    parser.add_argument("--des", type=bool, required=False, default=True, help="deploy_early_stop")
    parser.add_argument("--nne", type=bool, required=False, default=True, help="no_nan")
    parser.add_argument("--wt", type=bool, required=False, default=False, help="write_temp")
    parser.add_argument("--opt", type=int, required=False, default=0, help="\n".join(opt_str))

    args = parser.parse_args()
    protein_name = args.pn
    p_dir_ex = args.pd
    transfer_conv_weights_ex = args.tcw
    train_conv_layers_ex = args.tcl
    training_epochs_ex = args.ep
    es_patience_ex = args.pa
    batch_size_ex = args.bs
    lr_ex = args.lr

    if args.s3 > 0.:
        split_def_ex = [args.s1, args.s2, args.s3]
    else:
        split_def_ex = [args.s1, args.s2]

    architecture = pos_models[args.ac]
    save_model_ex = args.sm
    settings_test_ex = args.st
    dist_thr_ex = args.dt
    channel_num_ex = args.cn
    load_trained_weights_ex = args.ltw
    load_trained_model_ex = args.ltm
    max_train_mutations_ex = args.mtm
    test_num_ex = args.tn
    r_seed_ex = args.rs
    save_fig_ex = args.saf  # None to not save figures or eg 1 to save
    show_fig_ex = args.shf
    silent_ex = args.se
    ex_test = args.et
    deploy_early_stop_ex = args.des
    no_nan_ex = args.nne
    write_temp_ex = args.wt
    protein_attributes = protein_settings(protein_name)
    architecture_name = architecture.__code__.co_name
    number_mutations_ex = protein_attributes["number_mutations"]
    variants_ex = protein_attributes["variants"]
    score_ex = protein_attributes["score"]
    wt_seq_ex = list(protein_attributes["sequence"])
    first_ind_ex = int(protein_attributes["offset"])
    tsv_ex = "./datasets/{}.tsv".format(protein_name.lower())
    pdb_ex = "./datasets/{}.pdb".format(protein_name.lower())
    opt = pos_optimizer[args.opt]

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
         "model_to_use": architecture}
    return d


if __name__ == "__main__":
    pass
