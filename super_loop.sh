#!/bin/bash

names=("avgfp")  # ("avgfp" "pab1" "gb1")
split=("first") # ("first" "second" "third") 
train_sizes=(50 100 250 500 1000 2000 6000)
batch_size=32
architecture=sep_conv_mix
epochs=100
channels=7
simple=1
simple_trans_nt=0
simple_trans_t=1
aug=0
aug_trans_nt=0
aug_trans_t=0

for p_name in "${names[@]}"; do
  if [ $p_name == "avgfp" ]; then
    weigths="result_files/saved_models/avgfp_fr_50_25_10_2022_203303/" 
    l_rate=0.001
  elif [ $p_name == "pab1" ]; then 
    weigths="result_files/saved_models/pab1_23_07_2022_105600/" 
    l_rate=0.001
  else
    weigths="result_files/saved_models/gb1_22_07_2022_184057/" 
    l_rate=0.001
  fi 
  algn_file="datasets/alignment_files/"$p_name"_1000_experimental.clustal"
  tsv="nononsense/nononsense_"$p_name".tsv"
  echo "$p_name"
  echo $weigths
  echo "$tsv"

  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
    if [ $simple == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
    if [ $simple_trans_nt == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
    if [ $simple_trans_t == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
  done

  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
    if [ $aug == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf -da -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
    if [ $aug_trans_nt == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf -da -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
    if [ $aug_trans_t == 1 ]; then
        for i in "${train_sizes[@]}"; do
          if [ -d "$path$i" ];
          then
          python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf -da -cn $channels -lr "$l_rate"
          else
          echo "Failed to locate folder $path$i"
          fi
        done
    fi
  done
done
