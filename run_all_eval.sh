#!/bin/bash

DS_LIST="
$DATASETS_DIR/128kset/test_books3_129_128k-1000k.jsonl
"

# $DATASETS_DIR/128kset/test_pg19_20_130k-360k.jsonl
# $DATASETS_DIR/128kset/val_long_46_128k-700k.jsonl
# $DATASETS_DIR/128kset/proof_pile_36_128k-560k.jsonl

for ds in $DS_LIST
do
  echo "======== Evaluating $ds"
  # python ./eval.py --print --infer_type lstd --min_token_len 128 --max_token_len 256 --step_token_len 128 --model_name_or_path $MODELS_DIR/Llama-2-7b-chat-hf --test_data $ds
  python ./eval.py --print --infer_type base --min_token_len 128 --max_token_len 256 --step_token_len 128 --model_name_or_path $MODELS_DIR/Llama-2-7b-chat-hf --test_data $ds
  # python ./eval.py --print --infer_type eagle --min_token_len 128 --max_token_len 256 --step_token_len 128 --model_name_or_path $MODELS_DIR/Llama-2-7b-chat-hf --test_data $ds --eagle_path $MODELS_DIR/EAGLE-llama2-chat-7B
done

