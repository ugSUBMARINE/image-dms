#!/bin/bash
architectures=("simple_model_imp" "dense_net2" "sep_conv_mix")
for rand_S in 1 42 96; do
    for p_name in "pab1" "gb1" "avgfp" ;do
        for arch in "${architectures[@]}"; do
            for setting in 0 1 2 3; do
                pre_weigths="result_files/saved_models/"$arch"_pretrained_"$p_name"/"$p_name"_fr_50_"*
                echo $p_name
                echo $arch
                echo $setting
                if [ $setting == 0 ]; then
                    pre_weigths=
                    pret=
                    trl=
                    daug=
                elif [ $setting == 1 ]; then
                    pret=-tw
                    trl=-tl
                    daug=
                elif [ $setting == 2 ]; then
                    pre_weigths=
                    pret=
                    trl=
                    daug=-da
                else
                    pret=-tw
                    trl=-tl
                    daug=-da
                fi
                tsv_path="nononsense/nononsense_"$p_name".tsv"
                if [ $p_name == "avgfp" ]; then
                    red=-rd
                else
                    red= 
                fi
                if [ -f $tsv_path ];
                then
                python3 -Wdefault ./d4_cmd_driver.py -pn $p_name -qn $p_name -af ./datasets/alignment_files/"$p_name"_1000_experimental.clustal -te 100 -bs 64 -a $arch -rs $rand_S --tsv_filepath $tsv_path -wf --use_split_file "nononsense/single_double_avgfp/avgfp_splits0/" $red $pret $pre_weigths $trl $daug
                else
                echo "Failed to locate $tsv_path"
                fi
            done
        done
    done
done
