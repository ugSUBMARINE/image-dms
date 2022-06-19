#!/bin/bash

names=("pab1" "gb1" "avgfp")
split=("first" "second" "third")
train_sizes=(50 100 250 500 1000 2000 6000)
batch_size=32
architecture=2

for p_name in "${names[@]}"; do
  if [ $p_name == "avgfp" ]; then
    weigths="simple_model_imp_rr3/pab1_21_04_2022_123419"
  else
    weigths="simple_model_imp_rr3/avgfp_21_04_2022_125134"
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
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf
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
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" -wf -da
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -wf -da
      else
      echo "Failed to locate folder $path$i"
      fi
    done
    for i in "${train_sizes[@]}"; do
      if [ -d "$path$i" ];
      then
      python3 -Wdefault ./d4_cmd_driver.py -pn "$p_name" -te 100 -bs "$batch_size" -a "$architecture" -rs 1 --use_split_file "$path$i" --tsv_filepath "$tsv" --transfer_conv_weights $weigths -tl -wf -da
      else
      echo "Failed to locate folder $path$i"
      fi
    done
  done
done
