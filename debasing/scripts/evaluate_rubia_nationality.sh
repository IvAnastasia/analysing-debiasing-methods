
# bert 
# note that the results are also stored in checkpoints/bert-base-uncased/results, which is the same as gender and religion; you may manually change the directory name to avoid confliction
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-bert-prefix-tune-192
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/nationality-bert-prefix-tune-192" --prompt_model "prefix_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-bert-prompt-tune-192
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/nationality-bert-prompt-tune-192" --prompt_model "prompt_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-bert-adapter-rf4
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/nationality-bert-adapter-rf4/masked_lm" --adapter_config "seq_bn" --adapter_reduction_factor 4 --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''


# gpt2
# note that the results are also stored in checkpoints/gpt2/results, which is the same as gender and religion; you may manually change the directory name to avoid confliction
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-gpt2-prefix-tune-384
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/nationality-gpt2-prefix-tune-384" --prompt_model "prefix_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-gpt2-prompt-tune-384
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/nationality-gpt2-prompt-tune-384" --prompt_model "prompt_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# nationality-gpt2-adapter-rf2
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/nationality-gpt2-adapter-gpt2-rf2/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 2 --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''


# SentDebias
# for bert
cd experiments
python sentence_debias_subspace.py --model "BertModel" --model_name_or_path "bert-base-uncased" --bias_type "nationality"
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# for gpt2
cd experiments
python sentence_debias_subspace.py --model "GPT2Model" --model_name_or_path "gpt2" --bias_type "nationality"
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# SelfDebias
# for bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

# for gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "rubia" --bias_type "nationality" --seed 42 --output_dir ''

