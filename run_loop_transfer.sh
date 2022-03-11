#!/bin/bash
for i in nononsense/avgfp_even_splits/split_50 nononsense/avgfp_even_splits/split_100 nononsense/avgfp_even_splits/split_250 nononsense/avgfp_even_splits/split_500; do
    python3 -Wdefault ./d4batch_driver.py  --protein_name avgfp  --training_epochs 100 --batch_size 32 --architecture 2 --random_seed 1 --use_split_file "${i}" --tsv_filepath nononsense/nononsense_avgfp.tsv  --transfer_conv_weights trained_simple_model_d6/pab1_25_02_2022_090840/pab1_25_02_2022_090840
done
