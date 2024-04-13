#!/bin/bash

DS_LIST="
$DATASETS_DIR/128kset/test_books3_129_128k-1000k.jsonl
"
#$DATASETS_DIR/128kset/test_pg19_20_130k-360k.jsonl
#$DATASETS_DIR/128kset/val_long_46_128k-700k.jsonl
#$DATASETS_DIR/128kset/proof_pile_36_128k-560k.jsonl

for ds in $DS_LIST
do
  echo "======== Evaluating $ds"

  # 16k cache Yi
  python ./eval.py --recent_size 16384 --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-6B-200K --test_data $ds --min_token_len 4096 --max_token_len 102400
  python ./eval.py --recent_size 16384 --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-6B-200K --test_data $ds --min_token_len 4096 --max_token_len 102400

  python ./eval.py --recent_size 16384  --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-9B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  python ./eval.py --recent_size 16384  --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-9B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536

  python ./eval.py --recent_size 16384 --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-34B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  python ./eval.py --recent_size 16384 --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-34B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  
  
  
  # basic
  #python ./eval.py --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-6B-200K --test_data $ds --min_token_len 4096 --max_token_len 102400
  #python ./eval.py --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-6B-200K --test_data $ds --min_token_len 4096 --max_token_len 102400
  
  #python ./eval.py --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yarn-Llama-2-7b-128k --test_data $ds --min_token_len 4096 --max_token_len 102400
  #python ./eval.py --infer_type base --print --model_name_or_path $MODELS_DIR/Yarn-Llama-2-7b-128k --test_data $ds --min_token_len 4096 --max_token_len 102400
  
  python ./eval.py --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yarn-Llama-2-13b-128k --test_data $ds --min_token_len 4096 --max_token_len 65536
  python ./eval.py --infer_type base --print --model_name_or_path $MODELS_DIR/Yarn-Llama-2-13b-128k --test_data $ds --min_token_len 4096 --max_token_len 65536
  
  #python ./eval.py --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-9B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  #python ./eval.py --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-9B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  
  #python ./eval.py --infer_type lstd --print --model_name_or_path $MODELS_DIR/Yi-34B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536
  python ./eval.py --infer_type base --print --model_name_or_path $MODELS_DIR/Yi-34B-200K --test_data $ds --min_token_len 4096 --max_token_len 65536

done


