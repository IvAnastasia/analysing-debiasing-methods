#! /bin/bash
#SBATCH --job-name="bert_adapter_gender"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-10:0
#SBATCH --mail-user=iv.anastasiia02@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="gender_run_bert_adapter_rf4"%j.out
#SBATCH --error="gender_run_bert_adapter_rf4.err"%j.out
module load Python/Anaconda_v03.2023

python -u debias_adapter.py --model_name_or_path "bert-base-uncased" --task_type "masked_lm" --train_file "data/gender_corpus.txt" --max_seq_length 512 --line_by_line --bias_type "gender" --cda_mode "partial" --output_dir "checkpoints/gender-bert-adapter-rf48" --do_train --per_device_train_batch_size 16 --learning_rate 5e-4 --num_train_epochs 2 --save_strategy "no" --evaluation_strategy "epoch" --seed 42 --down_sample 1 --adapter_config "seq_bn"



