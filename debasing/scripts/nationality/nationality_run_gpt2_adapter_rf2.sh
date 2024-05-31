#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u debias_adapter.py \
	--model_name_or_path "gpt2" \
	--task_type "causal_lm" \
	--train_file "data/nationality_corpus.txt" \
	--max_seq_length 1024 \
	--line_by_line \
	--bias_type "nationality" \
	--cda_mode "partial" \
	--output_dir "checkpoints/nationality-gpt2-adapter-rf2" \
	--do_train \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--learning_rate 5e-3 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 1 \
	--adapter_config "seq_bn" \
#	--adapter_reduction_factor 2 \
	> nationality_run_gpt2_adapter_rf2.out 2>&1