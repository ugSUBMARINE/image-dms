# Example workflow

This directory provides an example of how to structure data, format files, and use the code in this repository.

For a detailed description of the file structure please take a look at [this README](https://github.com/ugSUBMARINE/image-dms/blob/8dbec0f0785f123129922a77f465c7abfd904c17/README.md?plain=1#L17C1-L17C1).

**This example assumes it is run from within the base directory of this repository (`image-dms`).**

## Pre-training dataset creation (optional)
This will create a dataset containing pseudo-scores in the `out_path` end will be similar to the output shown in  `gb1None.tsv`.

```sh
python3 d4_pt_dataset.py --p_name gb1 --pdb_file example/gb1.pdb --algn_path example/gb1_1000_experimental.clustal --p_data --out_path example/ --p_firstind 0 --p_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --name_var variant --name_nmut num_mutations --name_score score
```

For a detailed description of the used parameters run 
```sh
python3 d4_pt_dataset.py -h
```

## Pre-training a network (optional)
This will pre-train a model on the pseudo-scores in `gb1None.tsv`, store a model with the best weights based on the validation statistics and one from the end of the training which has the suffix `_end` in `result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/`, will store the used parameters in `result_files/log_file.csv` and the training statistics in `result_files/results.csv`.
It can be used in the next step as a starting point for training a network on experimentally determined data.

```sh
python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model
```

For a detailed description of the used parameters run 
```sh
python3 d4_cmd_driver.py -h
```

## Training a network with experimental data
In order to train a network on experimentally determined data (`gb1_small_clean.csv` in this case) use one of the methods mentioned below. 

This will train a network on a 0.7-0.15-0.05 training-validation-test split and save the trained model in `result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/` as well as the used parameters in `result_files/log_file.csv` and the training and test statistics in `result_files/results.csv`.

### Training without using a pre-trained model
```sh
python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model
```

### Training by using a pre-trained model
*CAUTION check the correct time stamp for the pre-trained model*
```sh
python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model --transfer_conv_weights result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/ --train_conv_layers
```

## Making predictions
*CAUTION check the correct time stamp for the trained model*

To make predictions for the variants `G1W` as well as `K42G,I24P` and get the predicted fitness score.

```sh
d4_predict.py --model_filepath result_files/saved_models/gb1None_10_01_2024_110517_end/ --protein_pdb example/gb1.pdb --alignment_file example/gb1_1000_experimental.clustal --query_name gb1 --variant_s G1W_K42G,I24P --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0
```

If only positive scores should be shown, one can pipe the output through the following commands:

```bash
| awk -F '_' '{print $1} $2 > 0' | grep _
```
 can be added if only positive scores shoud be shown (assumes a UNIX machine)
