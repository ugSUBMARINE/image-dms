import numpy as np
import pandas as pd
import os
import tensorflow as tf
from d4_models import *


def protein_settings(protein_name):
    """gets different setting for the protein of interest from the protein_settings file\n
        :parameter
            protein_name: str\n
            name of the protein in the protein_settings file
        :return
            protein_settings_dict: dict\n
            dictionary containing sequence, score, variants, number_mutations, offset column names
            :key sequence, score, variants, number_mutations, offset\n"""
    settings = pd.read_csv("datasets/protein_settings_ori.txt", delimiter=",")
    protein_name = protein_name.lower()
    content = np.asarray(settings[settings["name"] == protein_name][["attribute", "value"]])
    protein_settings_dict = dict(zip(content[:, 0], content[:, 1]))
    return protein_settings_dict


def create_folder(parent_dir, dir_name, add=""):
    """creates directory for current experiment\n
        :parameter
            parent_dir: str\n
            path where the new directory should be created\n
            dir_name: str\n
            name of the new directory\n
            add: str optional\n
            add to the name of the new directory\n"""
    if "/" in dir_name:
        dir_name = dir_name.replace("/", "_")
    directory = dir_name + add
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    return path


def log_file(file_path, write_str, optional_header=""):
    """reads and writes log info's to log file\n
        :parameter
            file_path: str\n
            path to log life\n
            write_str: str\n
            string that should be written to the log file\n
            optional_header: str, optional\n
            optional header to indicate the column names (',' separated)\n"""
    try:
        log_file_read = open(file_path, "r")
        prev_log = log_file_read.readlines()
        log_file_read.close()
        if len(list(prev_log)) == 0:
            prev_log = optional_header + "\n"
    except FileNotFoundError:
        if len(optional_header) > 0:
            prev_log = optional_header + "\n"
        else:
            prev_log = optional_header
    log_file_write = open(file_path, "w+")
    for i in prev_log:
        log_file_write.write(i)
    log_file_write.write(write_str + "\n")
    log_file_write.close()


def get_settings(run_name, file_path="result_files/log_file.csv", column_to_search="name"):
    """prints the settings/ results used in a certain run in an easy readable form
        :parameter
            run_name: str\n
            name of the row of interest\n
            file_path: str, optional\n
            path to the file that should be parsed\n
            column_to_search: str, optional\n
            specifies the column in which the run_name should be searched\n
        :return
            None"""
    data = pd.read_csv(file_path, delimiter=",")
    data_fields = data.columns
    roi = data[data[column_to_search] == run_name]
    for i, j in zip(data_fields, roi.values[0]):
        print("{:25}: {}".format(i, j))


def run_dict(run_name, column_to_search="name", data_path="result_files/log_file.csv"):
    """creates a dictionary from data_path that can be used as input for the run_all at d4batch_driver.py\n
        :parameter
            run_name: str\n
            name of the run whose parameters should be used\n
            column_to_search: str, optional\n
            specifies the column in which the run_name should be searched\n
            file_path: str, optional\n
            path to the file that should be parsed\n
        :return
            par: dict\n
            dictionary containing run_all parameters"""

    def get_func(name):
        """creates a function from a string\n
            :parameter
                name:str\n
                name of the function of interest\n
            :return
                method: function object\n
                the function object ot the function of interest"""
        possibles = globals().copy()
        possibles.update(locals())
        method = possibles.get(name)
        return method

    # data for the dictionary
    data = pd.read_csv(data_path, delimiter=",")
    roi = data[data[column_to_search] == run_name]

    # dictionary with not all strings converted
    par = dict(zip(list(roi.columns), roi.values[0]))
    par_k = list(par.keys())
    par_v = list(par.values())

    # convert the strings that are not the data type they should be into their respective type
    for i in range(len(par)):
        vi = par_v[i]
        vi_type = type(vi)
        if any([vi_type == int, vi_type == bool, vi_type == float]):
            pass
        else:
            if vi == "None":
                par[par_k[i]] = None
            elif "[" in vi:
                new_split_list = []
                split_list = vi[1:-1].split("_")
                for j in split_list:
                    if j.isdecimal():
                        new_split_list += [int(j)]
                    else:
                        new_split_list += [float(j)]
                par[par_k[i]] = new_split_list
            else:
                vi_split = vi.split(" ")
                if len(vi_split) > 1 and all(["<" in vi, ">" in vi, "function" in vi]):
                    par[par_k[i]] = get_func(vi_split[1])
                elif "optimizer" in vi:
                    par[par_k[i]] = tf.keras.optimizers.Adam

    # deletes entries that are not used in run_all
    del par["name"]
    del par["training_time_in_min"]
    return par


aa_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
           "VAL": "V", "LEU": "L", "ILE": "I", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "TRP": "W",
           "TYR": "Y", "THR": "T"}

hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9,
                  'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                  'W': -0.9, 'Y': -1.3}

# neutral 0, negatively charged -1, positively charged 1
charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
          'Q': 0, 'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}

# hydrogen bonding capability 0 no hydrogen bonding, 1 acceptor, 2 donor, 3 donor and acceptor
h_bonding = {'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 3, 'I': 0, 'K': 2, 'L': 0, 'M': 0, 'N': 3, 'P': 0,
             'Q': 3, 'R': 2, 'S': 3, 'T': 3, 'V': 0, 'W': 2, 'Y': 3}

# surface accessible side chain area
sasa = {'A': 75, 'C': 115, 'D': 130, 'E': 161, 'F': 209, 'G': 0, 'H': 180, 'I': 172, 'K': 205, 'L': 172, 'M': 184,
        'N': 142, 'P': 134, 'Q': 173, 'R': 236, 'S': 95, 'T': 130, 'V': 143, 'W': 254, 'Y': 222}

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
    # get_settings("pab1_02_03_2022_094555")

