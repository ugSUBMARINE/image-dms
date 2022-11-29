import os
from typing import Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def split_inds(
    file_path: str,
    variants: str,
    score: str,
    number_mutations: str,
    split: None | list[int | float] = None,
    remove_nonsense: bool = True,
    silent: bool = False,
    split_file_path: Union[None, str, dict] = None,
    train_name: str = "train",
    tune_name: str = "tune",
    test_name: str = "test",
) -> tuple[dict, dict]:
    """get indices of variants that don't feature a nonsense mutation
    :parameter
        - file_path:
          path to the tsv file of interest
        - variants:
          how the variants column is labeled in the tsv file
        - score:
          how the score column is labeled in the tsv file
        - number_mutations:
          how the number_mutations column is labeled in the tsv file
        - split:
          specifies the split for train, tune, test indices
            - float specifies fractions of the whole dataset
              eg [0.25, 0.25, 0.5] creates a train and tune dataset with 50 entries
              each and a test dataset of 100
              if the whole dataset contains 200 entries
            - int specifies the different number of samples per dataset
              eg [50,50,100] leads to a train and a tune dataset with 50 entries
              each and a test dataset of 100
              if the whole dataset contains 200 entries
            - None uses [0.8, 0.15, 0.05] as split
        - remove_nonsense:
          True removes indices of nonsense mutations of all possible indices to
          choose from
        - silent:
          if True doesn't print split sizes
        - split_file_path:
            - None
              the splits get created according to split
            - str
              splits get created according to the splits specified in the file in the
              directory - filenames without their file extensions need to be specified
              in train_name, tune_name and test_name
            - dict
              splits get created according to the specification in the dict -
              train_name, tune_name and test_name specify the keys for the dictionary
              how the train tune and test keys are named
        - train_name, tune_name, test_name:
          names of the train, tune and test data files - without their file extension
          e.g. 'train.txt' needs 'train'
    :returns
        - data_dict: dict
          dictionary containing the arrays with indices for the three data splits
          :key 'train', 'tune', 'test'
        - data: dict
          dictionary containing the arrays with variants (data), scores (labels) and
          number of mutations (mutations) for the train, tune and test splits
          prefix = ['train', 'tune', 'test']
          :key prefix_data, prefix_labels, prefix_mutations"""

    raw_data = pd.read_csv(file_path, delimiter="\t")
    # extract variants column
    variants_raw = np.asarray(raw_data["variant"])
    # check which variant doesn't feature a nonsense mutation
    if remove_nonsense:
        no_ast_bool = []
        for i in variants_raw:
            var = i.strip()
            no_ast_bool.append("*" not in var)
    else:
        no_ast_bool = np.ones(len(variants_raw)).astype(bool)

    if split_file_path is None:
        # get all possible indices of all rows and shuffle them
        possible_inds = np.arange(len(raw_data))[no_ast_bool]
        np.random.shuffle(possible_inds)
        # number of rows
        pi_len = len(possible_inds)

        # check inputs
        if any([isinstance(split, list), split is None]):
            if split is not None and len(split) < 2:
                raise ValueError(
                    "split needs to contain at least 2 inputs or needs to be None"
                )
        if split is None:
            split = [int(np.ceil(pi_len * 0.8)), int(np.floor(pi_len * 0.15))]
        elif isinstance(split, list):
            if len(split) == 3:
                if not all([split[0] > 0.0, split[1] > 0.0]):
                    raise ValueError("train and tune split need to be > 0.")
                if all([split[0] <= 1.0, split[1] <= 1.0, split[2] <= 1.0]):
                    if np.sum(split) > 1.0:
                        raise ValueError("sum of split fractions can't be > 1.")
                    split = [
                        int(np.ceil(pi_len * split[0])),
                        int(np.floor(pi_len * split[1])),
                    ]
                elif all([split[0] >= 1.0, split[1] >= 1.0, split[2] >= 1.0]):
                    split[0] = int(split[0])
                    split[1] = int(split[1])
                    split[2] = int(split[2])
                else:
                    raise ValueError(
                        "split needs to be either all > 1 to be used as split size"
                        " or all < 1 to be used as fraction"
                    )
            elif len(split) == 2:
                if not all([split[0] > 0.0, split[1] > 0.0]):
                    raise ValueError("both splits need to be > 0.")
                if all([split[0] <= 1.0, split[1] <= 1.0]):
                    if np.sum(split) > 1.0:
                        raise ValueError("sum of split fractions can't be > 1.")
                    split = [
                        int(np.ceil(pi_len * split[0])),
                        int(np.floor(pi_len * split[1])),
                    ]
                elif all([split[0] >= 1.0, split[1] >= 1.0]):
                    split[0] = int(split[0])
                    split[1] = int(split[1])
                else:
                    raise ValueError(
                        "split needs to be either all > 1 to be used as split size"
                        " or all < 1 to be used as fraction"
                    )
            else:
                raise ValueError("split as list needs to contain either 2 or 3 items")
        else:
            raise ValueError(
                "Incorrect split input needs to be list containing 'float' or 'int'"
                "or needs to be None"
            )

        # split indices in separate data sets for train, tune, test
        train = possible_inds[: split[0]]
        tune = possible_inds[split[0] : split[0] + split[1]]
        try:
            test = possible_inds[split[0] + split[1] : split[0] + split[1] + split[2]]
        except IndexError:
            test = possible_inds[split[0] + split[1] :]
    elif type(split_file_path) == str:
        split_dict = read_split_dir(split_file_path)
        train = np.asarray(split_dict[train_name])
        tune = np.asarray(split_dict[tune_name])
        test = np.asarray(split_dict[test_name])
    elif type(split_file_path) == dict:
        train = np.asarray(split_file_path[train_name])
        tune = np.asarray(split_file_path[tune_name])
        test = np.asarray(split_file_path[test_name])
    else:
        raise ValueError(
            "Incorrect input for split_file_path - expected None, str or dict but got"
            "{} instead".format(type(split_file_path))
        )

    if not silent:
        print(
            "size train split: {}\nsize tune split: {}\nsize test split: {}".format(
                len(train), len(tune), len(test)
            )
        )
    # dict containing the used indices to create the splits of the tsv file
    data_dict = {"train": train, "tune": tune, "test": test}

    # locating the data specified with the indices
    train_dataset = raw_data.iloc[train]
    tune_dataset = raw_data.iloc[tune]
    test_dataset = raw_data.iloc[test]

    # storing the data that was specified in ndarrays to return them as a dict
    train_data = np.asarray(train_dataset[variants])
    train_labels = np.asarray(train_dataset[score])
    train_mutations = np.asarray(train_dataset[number_mutations])

    tune_data = np.asarray(tune_dataset[variants])
    tune_labels = np.asarray(tune_dataset[score])
    tune_mutations = np.asarray(tune_dataset[number_mutations])

    test_data = np.asarray(test_dataset[variants])
    test_labels = np.asarray(test_dataset[score])
    test_mutations = np.asarray(test_dataset[number_mutations])

    # dict containing the data, labels and number of mutations of the different splits
    data = {
        "train_data": train_data,
        "train_labels": train_labels,
        "train_mutations": train_mutations,
        "tune_data": tune_data,
        "tune_labels": tune_labels,
        "tune_mutations": tune_mutations,
        "test_data": test_data,
        "test_labels": test_labels,
        "test_mutations": test_mutations,
    }

    return data_dict, data


def create_split_file(
    p_dir: str,
    name: str,
    train_split: list[int] | np.ndarray[tuple[int], np.dtype[int]],
    tune_split: list[int] | np.ndarray[tuple[int], np.dtype[int]],
    test_split: list[int] | np.ndarray[tuple[int], np.dtype[int]],
) -> None:
    """creates train tune and test split txt files in a directory called 'splits'
    :parameter
        - p_dir:
          where the splits' directory should be created
        - name:
          name of the protein
        - train_split, tune_split, test_split:
          lists that contain the indices of the splits that should be written to the
          corresponding files"""

    def open_and_write(
        file_path: str,
        fname: str,
        data: list[int] | np.ndarray[tuple[int], np.dtype[int]],
    ) -> None:
        """writes splits to file
        :parameter
            - file_path:
              where the file should be created
            - name:
              name of the file
            - data:
              data that should be written to file"""
        file = open(file_path + "/" + fname + ".txt", "w+")
        for i in data:
            file.write(str(i) + "\n")
        file.close()

    # create new split directory
    check = os.path.join(p_dir.strip(), name + "_splits")
    last_add = 0
    while os.path.isdir(check + str(last_add)):
        last_add += 1
    new_path = check + str(last_add)
    os.mkdir(new_path)

    # write splits to file
    open_and_write(new_path, "train", train_split)
    open_and_write(new_path, "tune", tune_split)
    open_and_write(new_path, "test", test_split)


def read_split_file(file_path: str) -> np.ndarray[tuple[int], np.dtype[int]]:
    """parses txt file that contains the indices for a split (one index per row) and
    returns the indices as ndarray
    :parameter
        - file_path:
          path where the file is stored
    :return
        - split_ind:
          contains the split indices of the parsed txt file"""
    split_file = open(file_path, "r")
    content = split_file.readlines()
    split_ind = []
    for i in content:
        split_ind.append(int(i.strip()))
    split_file.close()
    return np.asarray(split_ind)


def read_split_dir(file_path: str) -> dict:
    """reads train.txt, tune.txt, test.txt files from a directory and returns their
    index content as list of lists
    :parameter
        - file_path:
          path to directory where the three files are stored
    :return
        - splits_dict:
          dictionary containing the content of the files in file_path with their
          file names as keys and their content as values"""
    if os.path.isdir(file_path):
        s_dir, _, files = list(os.walk(file_path))[0]
        splits = []
        names = []
        if len(files) == 3:
            for i in files:
                names.append(str(i).split(".")[0])
                splits.append(read_split_file(os.path.join(s_dir, i)))
            splits_dict = dict(zip(names, splits))
            return splits_dict
        else:
            raise FileNotFoundError(
                "Wrong number of files to create train, tune, test index list - "
                f"3 needed but {len(files)} are given"
            )
    else:
        raise FileNotFoundError(
            f"Directory '{file_path}' containing the split files doesn't exist"
        )


def create_inds(
    data_path: str, train_size: int, tune_size: int = 5000, test_size: int = 5000
) -> tuple[
    np.ndarray[tuple[int], np.dtype[int]],
    np.ndarray[tuple[int], np.dtype[int]],
    np.ndarray[tuple[int], np.dtype[int]],
]:
    """creates split indices that can be used to split a tsv files rows into
    train tune and test datasets
    :parameter
    - data_path:
      location of the tsv file for which the splits should be produced
    - train_size:
      size of the train split
    - tune_size, test_size:
      sizes of the tune and test split
    :returns
    - train, tune, test: ndarray
      arrays containing the split indices for each split"""
    data_file = pd.read_csv(data_path, delimiter="\t")
    # all possible indices
    pos_inds = np.arange(len(data_file))
    np.random.shuffle(pos_inds)
    # split indices into the respective splits
    train = pos_inds[:train_size]
    tune = pos_inds[train_size : train_size + tune_size]
    test = pos_inds[train_size + tune_size : train_size + tune_size + test_size]
    return train, tune, test


def create_txt(
    file_path: str,
    data: list | np.ndarray[tuple[int], np.dtype[int | float | str]],
    name: str,
) -> None:
    """writes data (indices) to .txt file with one index per row
    :parameter
        - file_path: str
          where the file should be stored
        - data: list or list like
          data that should be written per line to the .txt file
        - name: str
          name of the file that should be created
    :return
        None"""
    # creates dir if it doesn't already exist
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    a = open(os.path.join(file_path, name), "w+")
    for i in data:
        a.write(str(i).strip() + "\n")
    a.close()


if __name__ == "__main__":
    pass
    """
    for protein_name in ["avgfp", "pab1", "gb1"]:
        base_dir = "nononsense"
        data_dir = os.path.join(base_dir, "nononsense_{}.tsv".format(protein_name))
        split_dir = os.path.join(base_dir, "{}_even_splits".format(protein_name))
        os.mkdir(split_dir)
        for i in [50, 100, 250, 500, 1000, 2000, 6000]:
            tr, tu, te = create_inds(data_dir, i)
            target_path = os.path.join(split_dir, "split_" + str(i))
            create_txt(target_path, tr, "train.txt")
            create_txt(target_path, tu, "tune.txt")
            create_txt(target_path, te, "stest.txt")
    """
