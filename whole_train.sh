#!/bin/bash
for p_name in "pab1" "gb1" ;do  # "pab1" "gb1" "avgfp" ;do
    tsv_path="nononsense/nononsense_"$p_name".tsv"
    if [ $p_name == "avgfp" ]; then
        red=-rd
    else
        red= 
    fi
    if [ -f $tsv_path ];
    then
    # python3 -Wdefault ./d4_cmd_driver.py -pn $p_name -qn $p_name -af ./datasets/alignment_files/"$p_name"_1000_experimental.clustal -te 100 -bs 64 -a simple_model_imp -rs 1 --tsv_filepath $tsv_path -wf -cn 7 -s0 0.8 -s1 0.15 -s2 0.05 -sm -fc
    # python3 -Wdefault ./d4_cmd_driver.py -pn $p_name -qn $p_name -af ./datasets/alignment_files/"$p_name"_1000_experimental.clustal -te 100 -bs 64 -a dense_net2 -rs 1 --tsv_filepath $tsv_path -wf -cn 7 -s0 0.8 -s1 0.15 -s2 0.05 -sm -fc
    python3 -Wdefault ./d4_cmd_driver.py -pn $p_name -qn $p_name -af ./datasets/alignment_files/"$p_name"_1000_experimental.clustal -te 100 -bs 64 -a sep_conv_mix -rs 1 --tsv_filepath $tsv_path -wf -cn 7 -s0 0.8 -s1 0.15 -s2 0.05 -sm -fc $red
    else
    echo "Failed to locate $tsv_path"
    fi
done
