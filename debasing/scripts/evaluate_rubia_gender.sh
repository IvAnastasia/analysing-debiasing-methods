

# bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# bert-prefix-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# bert-prompt-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# bert-adapter-rf48
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/bert-adapter-rf48/masked_lm" --adapter_config "seq_bn" --adapter_reduction_factor 48 --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''


# gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''
# gpt2-prefix-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# gpt2-prompt-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# gpt2-adapter-rf48
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/gpt2-adapter-rf48/causal_lm" --adapter_config "seq_bn" --adapter_reduction_factor 48 --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# SentDebias
# for bert
cd experiments
python sentence_debias_subspace.py --model "BertModel" --model_name_or_path "bert-base-uncased" --bias_type "gender"
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# for gpt2
cd experiments
python sentence_debias_subspace.py --model "GPT2Model" --model_name_or_path "gpt2" --bias_type "gender"
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''


# SelfDebias
# for bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''

# for gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "rubia" --bias_type "gender" --seed 42 --output_dir ''
