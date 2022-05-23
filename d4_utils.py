import os
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf

from d4_models import simple_model, simple_model_norm, simple_model_imp, create_simple_model, simple_model_gap, \
    simple_stride_model_test, shrinking_res, inception_res, deeper_res, res_net, vgg, simple_longer, \
    simple_stride_model, get_conv_mixer_256_8


def protein_settings(protein_name, data_path="./datasets/protein_settings_ori.txt"):
    """gets different setting for the protein of interest from the protein_settings file\n
        :parameter
            - protein_name: str\n
              name of the protein in the protein_settings file
            - data_path: str\n
              path to the protein_settings.txt file
        :return
            - protein_settings_dict: dict\n
              dictionary containing sequence, score, variants, number_mutations, offset column names
              :key sequence, score, variants, number_mutations, offset\n"""
    # all data of the different proteins
    settings = pd.read_csv(data_path, delimiter=",")
    # for which name to look for in the file
    protein_name = protein_name.lower()
    # getting only the rows containing data of the protein of interest
    content = np.asarray(settings[settings["name"] == protein_name][["attribute", "value"]])
    # creating a dict from the key and data columns
    protein_settings_dict = dict(zip(content[:, 0], content[:, 1]))
    return protein_settings_dict


def create_folder(parent_dir, dir_name, add=""):
    """creates directory for current experiment\n
        :parameter
            - parent_dir: str\n
              path where the new directory should be created\n
            - dir_name: str\n
              name of the new directory\n
            - add: str, (optional - default "")\n
              add to the name of the new directory\n
        :return
            - path: str\n
              path where the folder was created\n"""
    # replace "/" in the directory name to avoid the creation of a deeper folder
    if "/" in dir_name:
        warnings.warn("’/’ in dir_name was removed to avoid the creation of a deeper folder")
        dir_name = dir_name.replace("/", "_").replace("\\", "_")
    directory = dir_name + add
    # create the file if it doesn't exist already
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def log_file(file_path, write_str, optional_header=""):
    """reads previous contend and writes it and additional logs info's specified in write_str to log file\n
        :parameter
            - file_path: str\n
              path to log file\n
            - write_str: str\n
              string that should be written to the log file\n
            - optional_header: str, (optional - default "")\n
              optional header to indicate the column names (',' separated)\n
        :return
            None"""
    try:
        # write header to log file if it's empty
        log_file_read = open(file_path, "r")
        prev_log = log_file_read.readlines()
        log_file_read.close()
        if len(list(prev_log)) == 0:
            prev_log = optional_header + "\n"
    except FileNotFoundError:
        # if file doesn't exist what to write to the file as header
        if len(optional_header) > 0:
            prev_log = optional_header + "\n"
        else:
            prev_log = optional_header
    # open or create the file, write previous content to it if there was any and append it with write_str
    log_file_write = open(file_path, "w+")
    for i in prev_log:
        log_file_write.write(i)
    log_file_write.write(write_str + "\n")
    log_file_write.close()


def compare_get_settings(run_name1, run_name2=None,
                         file_path1="./result_files/log_file.csv",
                         file_path2="./result_files/log_file.csv",
                         column_to_search1="name",
                         column_to_search2="name"):
    """prints the settings/ results used in a certain run in an easy readable form or compares to different runs and
        prints the differences\n
        can also be used to display the differences in the results from results.csv of two runs\n
        :parameter
            - run_name1: str\n
              name of the row of interest\n
            - run_name2: str or None, (optional - None)\n
              name of the row to compare with\n
            - file_path1: str, optional\n
              path to the file that should be parsed\n
            - file_path2: str, optional\n
              path to the file that should be parsed for comparison (can be the same or a different one than
              file_path1\n
            - column_to_search1: str, (optional - default 'name')\n
              specifies the column in which the run_name1 should be searched\n
            - column_to_search2: str, (optional - default 'name')\n
              specifies the column in which the run_name2 should be searched\n
        :return
            None"""
    data1 = pd.read_csv(file_path1, delimiter=",")
    # name of the columns
    data_fields1 = data1.columns
    # data of the run_name1
    roi1 = data1[data1[column_to_search1] == run_name1]
    roi1 = roi1.values[0]

    if run_name2 is None:
        # print data in easy readable form
        for i, j in zip(data_fields1, roi1):
            print("{:25}: {}".format(i, j))
    else:
        # same es for run_name1 above
        data2 = pd.read_csv(file_path2, delimiter=",")
        data_fields2 = data2.columns
        roi2 = data2[data2[column_to_search2] == run_name2]
        roi2 = roi2.values[0]
        # which data fields are not the same
        if len(data_fields1) != len(data_fields2):
            raise LookupError("files contain different number of headers and therefore can't be compared")
        non_matching_field = data_fields1[np.invert(data_fields1 == data_fields2)]
        if len(non_matching_field) > 0:
            nm_str = ",".join(non_matching_field)
            warnings.warn("data fields do not match at: " + nm_str)
        # where the runs are different
        diff = roi1 != roi2
        diff_ind = np.where(diff)[0]
        for i in diff_ind:
            print(data_fields1[i])
            print(roi1[i], "---", roi2[i])
            print()


def get_func(name):
    """creates a function from a string\n
        :parameter
            - name:str\n
              name of the function of interest\n
        :return
            - method: function object\n
              the function object ot the function of interest"""
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(name)
    return method


def run_dict(run_name, column_to_search="name", data_path="./result_files/log_file.csv"):
    """creates a dictionary from data_path that can be used as input for the run_all at d4batch_driver.py\n
        :parameter
            - run_name: str\n
              name of the run whose parameters should be used\n
            - column_to_search: str, (optional - "name")\n
              specifies the column in which the run_name should be searched\n
            - file_path: str, optional\n
              path to the file that should be parsed\n
            - opt: class object\n
              optimizer to use\n
        :return
            - pre_dict: dict\n
              dictionary containing run_all parameters"""

    # data for the dictionary
    data = pd.read_csv(data_path, delimiter=",")
    # row of interest
    roi = data[data[column_to_search] == run_name]

    # dictionary with not all strings converted
    pre_dict = dict(zip(list(roi.columns), roi.values[0]))
    pre_keys = list(pre_dict.keys())
    pre_values = list(pre_dict.values())

    # convert the strings that are not the data type they should be into their respective type
    for i in range(len(pre_dict)):
        value_i = pre_values[i]
        value_i_type = type(value_i)
        if not any([value_i_type == int, value_i_type == bool, value_i_type == float]):
            if value_i.isdecimal():
                pre_dict[pre_keys[i]] = int(value_i)
            elif value_i == "None":
                pre_dict[pre_keys[i]] = None
            # to extract the correct split list
            elif "[" in value_i:
                new_split_list = []
                split_list = value_i[1:-1].split("_")
                for j in split_list:
                    if j.isdecimal():
                        new_split_list += [int(j)]
                    else:
                        new_split_list += [float(j)]
                pre_dict[pre_keys[i]] = new_split_list
            else:
                vi_split = value_i.split(" ")
                if len(vi_split) > 1 and all(["<" in value_i, ">" in value_i, "function" in value_i]):
                    pre_dict[pre_keys[i]] = get_func(vi_split[1])
                elif "optimizer" in value_i:
                    opt_name = value_i.replace(">", "").replace("<", "").replace("'", "").split(".")[-1]
                    pre_dict[pre_keys[i]] = getattr(tf.keras.optimizers, opt_name)

    # deletes entries that are not used in run_all
    del pre_dict["name"]
    del pre_dict["training_time_in_min"]
    return pre_dict


def clear_log(file_path, text=None):
    """clears or creates log file\n
        :parameter
            - file_path: str\n
              path ot log file\n
            - text: str or None, (optional - default None)\n
              text that should be written to the file if None nothing gets written to the file\n"""
    a = open(file_path, "w+")
    if text is not None:
        a.write(text)
    a.close()


def remove_csv_column(file_path, col_name=None):
    """removes one or more columns of a csv file\n
        :parameter
            - file_path: str\n
              path to csv file where columns should be removed\n
            - col_name: tuple of one or multiple strings\n
              column header(s) that should be removed\n
        :return
            None
        """
    # file content
    content = pd.read_csv(file_path, delimiter=",")
    # file content without the columns that should be removed
    new_content = content.drop(columns=[*col_name], axis=1)
    # write to file
    new_content.to_csv(file_path, index=False)


def read_blosum():
    """read blosum matrix file from https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt and modify it\n
        :parameter
            None
        :returns
            - blosum_matrix: 2D ndarray of ints\n
              blosum matrix with 0 filled diagonal
            - blosum_keys: list of str\n
              which row/ column in the matrix corresponds to which amino acid
            """
    # read the file
    data = open("datasets/BLOSUM62.txt", "r")
    lines = data.readlines()
    data.close()
    line_counter = 0
    value_lines = []
    keys = []
    for i in lines:
        line = i.strip()
        if not line.startswith("#"):
            # extract which amino acid is in which row / column
            if line_counter == 0:
                keys += line.split("  ")[:-4]
            else:
                # get the matrix and get rid of spaces
                data_line = line.split(" ")
                read_data_line = []
                for k in data_line:
                    kstrip = k.strip()
                    if len(kstrip) > 0:
                        read_data_line += [kstrip]
                value_lines += [read_data_line[1:-4]]
            line_counter += 1
    # create 2d ndarray an fill diagonal with 0
    blosum_matrix = np.asarray(value_lines, dtype=int)
    blosum_matrix = blosum_matrix[:-4]
    np.fill_diagonal(blosum_matrix, 0)
    blosum_keys = np.asarray(keys)
    return blosum_matrix, blosum_keys


def present_matrix(matrix, col_row):
    """prints a given square matrix where columns and rows have the same one letter head in an easy readable way\n
        :parameter
            - matrix: 2D ndarray of ints or floats\n
              substitution matrix\n
            - col_row: list of str\n
              name of the columns/ rows (usually the amino acid one-letter code)\n
        :return
            None"""

    max_len = 0
    for i in matrix.flatten().astype(str):
        l = len(i)
        if l > max_len:
            max_len = l
    inter_cr = []
    for i in col_row:
        dcr = max_len - len(i)
        inter_cr += [" " * dcr + i]
    print(" " * (max_len - 1), " ".join(inter_cr))

    for ci, i in enumerate(matrix.astype(str)):
        inter_i = [col_row[ci]]
        for j in i:
            dlj = max_len - len(j)
            inter_i += [" " * dlj + j]
        print(" ".join(inter_i))


def star_message():
    """prints start message\n"""
    print("""
                                o===o     o    o 
                                ||  \\\   ||   || 
                                ||   OO   o====O 
                                ||  //        || 
                                o===o         oo 
                        """)
    thy = "- .... .- -. -.- ....... -.-- --- ..- ....... ..-. --- .-. ....... ..- ... .. -. --. ....... -.. ....-"
    hw = ".-- . ....... .... --- .--. . ....... .. - ....... .-- .. .-.. .-.. ....... . -. .... .- -. -.-. . " \
         "....... -.-- --- ..- .-. ....... .-. . ... . .- .-. -.-. ...."
    print(thy)
    print(hw)


aa_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
           "VAL": "V", "LEU": "L", "ILE": "I", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "TRP": "W",
           "TYR": "Y", "THR": "T"}

hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9,
                  'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                  'W': -0.9, 'Y': -1.3}

# neutral 0, negatively charged -1, positively charged 1
charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
          'Q': 0, 'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
"""
+- -1
++  1
--  1
nn  0
n+  0
n-  0
"""

# charge = {'A': 2., 'C': 2., 'D': -1., 'E': -1., 'F': 2., 'G': 2., 'H': 1., 'I': 2., 'K': 1., 'L': 2., 'M': 2.,
# 'N': 2., 'P': 2., 'Q': 2., 'R': 1., 'S': 2., 'T': 2., 'V': 2., 'W': 2., 'Y': 2.}

# hydrogen bonding capability 0 no hydrogen bonding, 1 acceptor, 2 donor, 3 donor and acceptor
h_bonding = {'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 3, 'I': 0, 'K': 2, 'L': 0, 'M': 0, 'N': 3, 'P': 0,
             'Q': 3, 'R': 2, 'S': 3, 'T': 3, 'V': 0, 'W': 2, 'Y': 3}

# surface accessible side chain area
sasa = {'A': 75, 'C': 115, 'D': 130, 'E': 161, 'F': 209, 'G': 0, 'H': 180, 'I': 172, 'K': 205, 'L': 172, 'M': 184,
        'N': 142, 'P': 134, 'Q': 173, 'R': 236, 'S': 95, 'T': 130, 'V': 143, 'W': 254, 'Y': 222}

# amino acid side chain length from CA to the furthest side chain atom
side_chain_length = {'A': 1.53832,
                     'C': 2.75909,
                     'D': 3.66044,
                     'E': 4.99565,
                     'F': 5.17121,
                     'G': 0,
                     'H': 4.47560,
                     'I': 4.03327,
                     'K': 6.36891,
                     'L': 3.97704,
                     'M': 5.34885,
                     'N': 3.44372,
                     'P': 2.44674,
                     'Q': 5.04045,
                     'R': 8.27652,
                     'S': 2.48484,
                     'T': 2.64210,
                     'V': 2.66271,
                     'W': 5.94311,
                     'Y': 6.53028}

if __name__ == "__main__":
    pass
    # compare_get_settings("avgfp_09_03_2022_134211", "avgfp_09_03_2022_132158")
    # run_dict("bgl3_06_03_2022_215803")
    # compare_get_settings("nononsense_pab1_22_03_2022_093849", "nononsense_pab1_22_03_2022_094448")
    # compare_get_settings("nononsense_pab1_21_03_2022_195748", "nononsense_pab1_17_03_2022_073620", file_path1="nononsense/second_split_run/log_results/pab1_log_file.csv",file_path2="nononsense/first_split_run/logs_results_cnn/pab1_log_file.csv")
    # compare_get_settings("nononsense_avgfp_12_03_2022_080540")

    star_message()
