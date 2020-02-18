#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/saturated-sgd/configs/prune-eval.jsonnet

percent=0.2

dir=/tmp/willm/saturated-sgd/prune-eval/$percent
PERCENT=$percent allennlp train $conf -s $dir --include-package src
