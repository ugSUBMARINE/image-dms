#!/bin/bash

names=("avgfp" "pab1" "gb1")
split=("first" "second" "third") 
schemes=("anh_scan" "position_aware")

for sch in "${schemes[@]}";do
    for p_name in "${names[@]}"; do
        tsv="nononsense_"$p_name".tsv"
        echo "$p_name"
        echo "$tsv"
        for k in "${split[@]}"; do
            path="position_data/"$sch"/"$sch"_"$p_name"_"$k
        echo $path
        
        if [ -d "$path" ]; then
            p=$(python code/regression.py --dataset_name $p_name --net_file network_specs/cnns/cnn-5xk3f128.yml --epochs 100 --dataset_file "position_data/$tsv" --split_dir $path --early_stopping | tail -n 10)
            echo "$p_name,$sch" >> "../results.csv"
            echo "$p" >> "../results.csv"
            echo "" >> "../results.csv"
        else
          echo "Failed to locate folder $path"
        fi
        done
    done
done
