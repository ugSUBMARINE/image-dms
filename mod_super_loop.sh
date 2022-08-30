#!/bin/bash

names=("pab1")  # ("avgfp" "pab1" "gb1")
split=("first" "second" "third") 
train_sizes=(50 100 250 500 1000 2000 6000)
batch_size=32
architecture=dense_net2
epochs=100
channels=7
simple=1
simple_trans_nt=1
simple_trans_t=1
aug=1
aug_trans_nt=1
aug_trans_t=1
l_rate=0.001


for p_name in "${names[@]}"; do
  algn_file="datasets/alignment_files/"$p_name"_1000_experimental.clustal"
  tsv="nononsense/nononsense_"$p_name".tsv"
  echo "$p_name"
  echo "$tsv"

  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
      if [ $k == "first" ]; then
        pre_weigths="result_files/saved_models/"$p_name"_fr_"
      elif [ $k == "second" ]; then 
        pre_weigths="result_files/saved_models/"$p_name"_sr_"
      else
        pre_weigths="result_files/saved_models/"$p_name"_tr_"
      fi 
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
          weigths=$pre_weigths$i"_"*
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
          weigths=$pre_weigths$i"_"*
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
          weigths=$pre_weigths$i"_"*
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
          weigths=$pre_weigths$i"_"*
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
