#! /bin/bash
#SBATCH --job-name="gpt_prompt_rel"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-8:0
#SBATCH --mail-user=iv.anastasiia02@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="religion_run_gpt_prompt_rf4"%j.out
#SBATCH --error="religion_run_gpt_prompt_rf4.err"%j.out

module load Python/Anaconda_v03.2023

python -u debias.py --model_name_or_path "gpt2" --task_type "causal_lm" --prompt_model "prompt_tuning" --pre_seq_len 384 --train_file "data/religion_corpus.txt"	--max_seq_length 640 --line_by_line --bias_type "religion" --cda_mode "partial" --output_dir "checkpoints/religion-gpt2-prompt-tune-384" --do_train --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 5e-1 --num_train_epochs 2 --save_strategy "no" --evaluation_strategy "epoch" --seed 42 --down_sample 1

