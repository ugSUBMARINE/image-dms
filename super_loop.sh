#!/bin/bash

names=("pab1") # "gb1" "avgfp")
split=("first" "second" "third") 
train_sizes=(50 100 250 500 1000 2000 6000)
batch_size=32
architecture=15
epochs=100
channels=7

for p_name in "${names[@]}"; do
  if [ $p_name == "avgfp" ]; then
    weigths="simple_model_imp_rr3/pab1_21_04_2022_123419"
  else
    weigths="result_files/saved_models/pab1_07_07_2022_165023/"
    algn_file="~//Downloads/pab1_1000_experimental.clustal"
  fi
  tsv="nononsense/nononsense_"$p_name".tsv"
  echo "$p_name"
  echo $weigths
  echo "$tsv"

  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
  done

  for k in "${split[@]}"; do
    path="nononsense/"$k"_split_run/"$p_name"_even_splits/split_"
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf -da -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf -da -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -qn "$p_name" -af $algn_file -te $epochs -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf -da -cn $channels
      else
      echo "Failed to locate folder $path$i"
      fi
    done
  done
done
