import configparser
import os


def retrieve_args(ini_filepath: str) -> dict:
    """reads the supplied ini file, checks for data types and returns args as
        dict for run_all
    :parameter
    - ini_filepath:
      file path to the ini file
    :return
    - converted_args:
      dict with all args from the ini file converted to their data type
    """

    config = configparser.ConfigParser()
    config.read(ini_filepath)

    # convert args from str to right data type
    converted_args = {}
    config_data = config["run args"]
    for i in config_data:
        i_data = config_data[i]
        # converte None
        if i_data.upper() == "NONE" or i_data == "":
            converted_args[i] = None
        elif any(map(str.isdigit, i_data)):
            # converte to int
            if i_data.isdigit():
                converted_args[i] = int(i_data)
            else:
                # convert to float
                try:
                    converted_args[i] = float(i_data)
                # convert to str
                except ValueError:
                    converted_args[i] = i_data
        else:
            # convert to bool
            try:
                converted_args[i] = config_data.getboolean(i)
            except ValueError:
                converted_args[i] = i_data

    none_type = type(None)
    type_check = {
        "architecture": str,
        "protein_name": [str, none_type],
        "optimizer": str,
        "tsv_filepath": str,
        "pdb_filepath": str,
        "wt_seq": str,
        "number_mutations": str,
        "variants": str,
        "score": str,
        "dist_thr": [int, float],
        "max_train_mutations": [int, none_type],
        "training_epochs": int,
        "test_num": int,
        "first_ind": int,
        "alignment_file": str,
        "query_name": str,
        "random_seed": [type(None), int],
        "deploy_early_stop": bool,
        "es_monitor": str,
        "es_min_d": float,
        "es_patience": int,
        "es_mode": str,
        "restore_bw": bool,
        "load_trained_model_path": [str, none_type],
        "batch_size": int,
        "save_figures": [str, none_type],
        "show_figures": bool,
        "write_to_log": bool,
        "silent_execution": bool,
        "extensive_test": bool,
        "save_model": bool,
        "load_trained_weights_path": [str, none_type],
        "no_nan": bool,
        "settings_test": bool,
        "p_dir": [str, none_type],
        "validate_training": bool,
        "learning_rate": float,
        "transfer_conv_weights": [str, none_type],
        "train_conv_layers": bool,
        "write_temp": bool,
        "split_file_creation": bool,
        "use_split_file": [str, none_type],
        "data_aug": bool,
        "clear_error_log": bool,
        "split0": float,
        "split1": float,
        "split2": float,
        "reduce": bool,
        "jit": bool,
    }

    # check if all inputs got converted to the right type
    for k, v in converted_args.items():
        check = type_check[k]
        cur_type = type(v)
        check_pass = True
        if type(check) == list:
            if cur_type not in check:
                check_pass = False
        else:
            if not cur_type == check:
                check_pass = False
        if not check_pass:
            raise TypeError(
                "{} got converted to type '{}' but needs type"
                "‘{}‘- please check input".format(k, cur_type, check)
            )

    return converted_args


if __name__ == "__main__":
    print(retrieve_args("./datasets/config_files/config.ini"))
