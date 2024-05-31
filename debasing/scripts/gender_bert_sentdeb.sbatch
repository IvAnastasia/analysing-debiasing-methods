#! /bin/bash
#SBATCH --job-name="bert_gender_sent"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-2:0
#SBATCH --mail-user=iv.anastasiia02@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="bert_gender_sent_rf4"%j.out
#SBATCH --error="bert_gender_sent_rf4.err"%j.out
source deactivate
source /home/adivanova_2/miniconda3/bin/activate

python sentence_debias_subspace.py --model "BertModel" --model_name_or_path "bert-base-uncased" --bias_type "gender"