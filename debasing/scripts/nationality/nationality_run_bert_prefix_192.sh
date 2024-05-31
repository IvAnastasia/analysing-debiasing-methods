#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u debias.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "masked_lm" \
	--prompt_model "prefix_tuning" \
	--pre_seq_len 192 \
	--train_file "data/nationality_corpus.txt" \
	--max_seq_length 320 \
	--line_by_line \
	--bias_type "nationality" \
	--cda_mode "partial" \
	--output_dir "checkpoints/nationality-bert-prefix-tune-192" \
	--do_train \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-3 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 1 \
	--overwrite_output_dir \
	> nationality_run_bert_prefix_192.out 2>&1