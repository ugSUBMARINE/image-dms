#!/bin/bash
protein=pab1
for run in "fr" "sr" "tr";do
    for size in 50 100 250 500 1000 2000 6000;do
        tsv_path=./datasets/pseudo_scores/"$protein"_"$run"_"$size".tsv 
        python3 -Wdefault ./d4_cmd_driver.py -pn pab1 -qn pab1 -af ./datasets/alignment_files/"$protein"_1000_experimental.clustal -te 100 -bs 64 -a dense_net2 -rs 1 --tsv_filepath $tsv_path -wf -cn 7 -s0 0.8 -s1 0.15 -s2 0.05 -sm
    done
done

