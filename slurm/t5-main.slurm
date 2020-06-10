#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate t5
cd /home/willm/code/saturated-sgd
python t5_main.py
