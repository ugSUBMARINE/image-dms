#!/bin/bash

names=("avgfp" "pab1" "gb1")
split=("first" "second" "third") 
train_sizes=(50 100 250 500 1000 2000 6000)
batch_size=32
epochs=100


for p_name in "${names[@]}"; do
  tsv="nononsense/nononsense_"$p_name".tsv"
  echo "$p_name"
  echo "$tsv"
  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
              echo $tsv
              echo $path$i
              p=$(python code/regression.py --dataset_name $p_name --net_file network_specs/cnns/cnn-5xk3f128.yml --epochs $epochs --dataset_file $tsv --split_dir $path$i --early_stopping | tail -n 10)
              echo "$p_name,$k,$i" >> "../var_size_results.txt"
              echo "$p" >> "../var_size_results.txt"
              echo "" >> "../var_size_results.txt"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
  done
done
