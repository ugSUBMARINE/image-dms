#!/bin/bash
p_name=avgfp
architecture=sep_conv_mix
if [ $p_name == "avgfp" ]; then
  red=-rd
else
  red= 
fi
for run in "fr" "sr" "tr" ;do # "fr" "sr" "tr" 
    for size in 50 100 250 500 1000 2000 6000;do
        tsv_path="datasets/pseudo_scores/"$p_name"/"$p_name"_"$run"_"$size".tsv" 
        if [ -f $tsv_path ];
        then
        python3 -Wdefault ./d4_cmd_driver.py -pn $p_name -qn $p_name -af ./datasets/alignment_files/"$p_name"_1000_experimental.clustal -te 70 -bs 32 -a "$architecture" -rs 1 --tsv_filepath $tsv_path -wf -cn 7 -s0 0.8 -s1 0.15 -s2 0.05 -sm $red
        else
        echo "Failed to locate $tsv_path"
        fi
    done
done

