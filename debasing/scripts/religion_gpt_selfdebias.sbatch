#! /bin/bash
#SBATCH --job-name="gpt2_religion_self"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-2:0
#SBATCH --mail-user=iv.anastasiia02@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="gpt2_religion_self_rf4"%j.out
#SBATCH --error="gpt2_religion_self_rf4.err"%j.out
source deactivate
source /home/adivanova_2/miniconda3/bin/activate


python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "rubia" --bias_type "religion" --seed 42 --output_dir ''