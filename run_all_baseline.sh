#!/bin/bash

M_LIST="
$MODELS_DIR/LWM-Text-Chat-1M
$MODELS_DIR/Yarn-Llama-2-7b-128k
$MODELS_DIR/Yarn-Llama-2-13b-128k
"

OUTDIR=./baseline_output

# main

# test lveval main
DS_LIST="
$DATASETS_DIR/LVEval/loogle_SD_mixup/loogle_SD_mixup_128k.jsonl
$DATASETS_DIR/LVEval/loogle_CR_mixup/loogle_CR_mixup_128k.jsonl
$DATASETS_DIR/LVEval/loogle_MIR_mixup/loogle_MIR_mixup_128k.jsonl
"
for ds in $DS_LIST
do
  for mp in $M_LIST
  do
      echo "======== Evaluating lveval main, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"
      python ./eval_prompt.py --infer_type base --print --max_token_len 102400 --max_gen_len 512 $METRIX_CMD --algo 6 &>> $OUTDIR/test_lveval_main.log

  done
done


# test qa and sum
DS_LIST="
$DATASETS_DIR/InfiniteBench/longbook_sum_eng.jsonl
$DATASETS_DIR/InfiniteBench/longbook_qa_eng.jsonl
"
for ds in $DS_LIST
do
  for mp in $M_LIST
  do
      echo "======== Evaluating qa and sum main, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"
      python ./eval_prompt.py --infer_type base --print --max_token_len 130072 --max_gen_len 512 $METRIX_CMD --algo 6 &>> $OUTDIR/test_qa_sum_main.log
  done
done


# test text generation main
DS_LIST="
$DATASETS_DIR/128kset/proof-pile.jsonl
$DATASETS_DIR/128kset/test_books3_129_128k-1000k.jsonl
"
# 32k~128k
# 64k~128k
# above 128k
for ds in $DS_LIST
do
  for mp in $M_LIST
  do
      echo "======== Evaluating text generation, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"

      python ./eval.py --infer_type base --print --step_token_len 8192 --min_token_len 32768 --max_token_len 102400 --max_gen_len 512 $METRIX_CMD &>> $OUTDIR/test_txt_gen_32_100_main.log
      python ./eval.py --infer_type base --print --step_token_len 8192 --min_token_len 65536 --max_token_len 130072 --max_gen_len 512 $METRIX_CMD &>> $OUTDIR/test_txt_gen_64_128_main.log
  done
done


#
# vice
#


# test lveval vice
DS_LIST="
$DATASETS_DIR/LVEval/hotpotwikiqa_mixup/hotpotwikiqa_mixup_128k.jsonl
$DATASETS_DIR/LVEval/multifieldqa_en_mixup/multifieldqa_en_mixup_128k.jsonl
$DATASETS_DIR/LVEval/factrecall_en/factrecall_en_128k.jsonl
"

# vice models
M_LIST_2="
$MODELS_DIR/Yi-6B-200K
$MODELS_DIR/Yi-9B-200K
$MODELS_DIR/Yi-34B-200K
"
for ds in $DS_LIST
do
  for mp in $M_LIST_2
  do
      echo "======== Evaluating lveval vice, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"
      python ./eval_prompt.py --infer_type base --print --max_token_len 102400 --max_gen_len 512 $METRIX_CMD --algo 6 &>> $OUTDIR/test_lveval_vice.log

  done
done


# test qa and sum vice
DS_LIST="
$DATASETS_DIR/InfiniteBench/longbook_sum_eng.jsonl
$DATASETS_DIR/InfiniteBench/longbook_qa_eng.jsonl
"
for ds in $DS_LIST
do
  for mp in $M_LIST_2
  do
      echo "======== Evaluating qa and sum main, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"
      python ./eval_prompt.py --infer_type base --print --max_token_len 130072 --max_gen_len 512 $METRIX_CMD --algo 6 &>> $OUTDIR/test_qa_sum_vice.log
  done
done

# test text generation vice
DS_LIST="
$DATASETS_DIR/128kset/proof-pile.jsonl
$DATASETS_DIR/128kset/test_books3_129_128k-1000k.jsonl
"
# 32k~128k
# 64k~128k
# above 128k
for ds in $DS_LIST
do
  for mp in $M_LIST
  do
      echo "======== Evaluating text generation, ds $ds, mp $mp"

      METRIX_CMD="--model_name_or_path $mp --test_data $ds --max_test_count 100 --output_dir $OUTDIR"

      python ./eval.py --infer_type base --print --step_token_len 8192 --min_token_len 32768 --max_token_len 102400 --max_gen_len 512 $METRIX_CMD &>> $OUTDIR/test_txt_gen_32_100_vice.log
      python ./eval.py --infer_type base --print --step_token_len 8192 --min_token_len 65536 --max_token_len 130072 --max_gen_len 512 $METRIX_CMD &>> $OUTDIR/test_txt_gen_64_128_vice.log
  done
done



# TODO: lade, ssd


