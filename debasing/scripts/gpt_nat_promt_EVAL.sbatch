

python evaluate.py --model_name_or_path "checkpoints/nationality-gpt2-prompt-tune-384" --prompt_model "prompt_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''