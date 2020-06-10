#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#turnoff SBATCH --gpus=1
#SBATCH --priority=allennlp_hipri
#SBATCH --exclude=allennlp-server1,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturating-will

python main.py
