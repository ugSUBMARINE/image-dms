#!/bin/bash

names=("avgfp" "pab1" "gb1")
split=("first" "second" "third") 
# schemes=("position_aware")
schemes=("anh_scan")
batch_size=32
epochs=100
l_rate=0.001

for architecture in sep_conv_mix simple_model_imp;do # sep_conv_mix simple_model_imp;do
    for sch in "${schemes[@]}";do
        for p_name in "${names[@]}"; do
            algn_file="datasets/alignment_files/"$p_name"_1000_experimental.clustal"
            tsv="nononsense/nononsense_"$p_name".tsv"
            if [ $p_name == "avgfp" ]; then
              red=-rd
            else
              red= 
            fi
            echo "$p_name"
            echo "$tsv"
            for k in "${split[@]}"; do
                path="nononsense/"$sch"/"$sch"_"$p_name"_"$k
            echo $path
            if [ -d "$path" ]; then
              python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path" --tsv_filepath "$tsv" -wf -lr "$l_rate" $red -j
            else
              echo "Failed to locate folder $path"
            fi
            done
        done
    done
    echo "" >> result_files/results.csv
    echo "" >> result_files/log_file.csv
done
