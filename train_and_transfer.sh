#!/bin/bash

p_name="avgfp"
path="nononsense/"$p_name"_even_splits/split_"
tsv="nononsense/nononsense_"$p_name".tsv"

transfer=True
normal=False
train_sizes=(50 100 250 500 1000 2000 6000)


if [ "$normal" = True ]; then
  for i in "${train_sizes[@]}"; do
    if [ -d "$path$i" ];
    then
    python3 -Wdefault ./d4batch_driver.py --protein_name "$p_name" --training_epochs 100 --batch_size 32 --architecture 2 --random_seed 1 --use_split_file "$path$i" --tsv_filepath "$tsv"
    else
    echo "Failed to locate folder $path$i"
    fi
  done
fi

if [ "$transfer" = True ]; then
  for i in "${train_sizes[@]}"; do
    if [ -d "$path$i" ];
    then
    python3 -Wdefault ./d4batch_driver.py --protein_name "$p_name" --training_epochs 100 --batch_size 32 --architecture 2 --random_seed 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights ./trained_simple_model_d6/pab1_25_02_2022_090840/pab1_25_02_2022_090840
    else
    echo "Failed to locate folder $path$i"
    fi
  done
fi
