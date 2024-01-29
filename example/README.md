# Example workflow

This directory provides an example of how to structure data, format files, and use the code in this repository.

For a detailed description of the file structure please take a look at [this README](https://github.com/ugSUBMARINE/image-dms/blob/8dbec0f0785f123129922a77f465c7abfd904c17/README.md?plain=1#L17C1-L17C1).

**This example assumes it is run from within the base directory of this repository (`image-dms`).**

## Pre-training dataset creation (optional)
For a detailed description of the used parameters run `python3 d4_pt_dataset.py`

`python3 d4_pt_dataset.py --p_name gb1 --pdb_file example/gb1.pdb --algn_path example/gb1_1000_experimental.clustal --p_data --p_firstind 0 --p_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --out_path example/ --name_var variant --name_nmut num_mutations --name_score score`

## Pre-training a network (optional)
For a detailed description of the used parameters run `python3 d4_cmd_driver.py`

`python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model `

This will save the network trained on the pre-training dataset created in the previous step. It can be used in the next step as a starting point for training a network on experimentally determined data.
The saved network can be found in `result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/` where the time stamp will depend on the time the training was started.

## Training a network with experimental data
In order to train a network on experimentally determined data use one of the methods mentioned below. 

This will train a network on a 0.7-0.15-0.05 training-validation-test split and save the trained model in `result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/`

### Training without using a pre-trained model
`python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model`

### Training by using a pre-trained model
`python3 d4_cmd_driver.py --query_name gb1 --alignment_file example/gb1_1000_experimental.clustal --tsv_filepath example/gb1None.tsv --pdb_filepath example/gb1.pdb --number_mutations num_mutations --variants variant --score score --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0 --training_epochs 100 --split0 0.7 --split1 0.15 --split2 0.05 --save_model --transfer_conv_weights result_files/saved_models/gb1None_DD_MM_YYYY_HHMMSS/ --train_conv_layers`

## Making predictions
To make predictions for the variants `G1W` as well as `K42G,I24P` and get the predicted fitness score.
`d4_predict.py --model_filepath result_files/saved_models/gb1None_10_01_2024_110517_end/ --protein_pdb example/gb1.pdb --alignment_file example/gb1_1000_experimental.clustal --query_name gb1 --variant_s G1W_K42G,I24P --wt_seq MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE --first_ind 0`
(`| awk -F '_' '{print $1} $2 > 0' | grep _` can be added if only positive scores shoud be shown [assumes a UNIX machine])

