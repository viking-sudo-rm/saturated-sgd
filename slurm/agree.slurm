#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/saturated-sgd/configs/agree.jsonnet

n_layers=2
drop=0.0

dir=/tmp/willm/saturated-sgd/agree/${n_layers}-drop-$drop
N_LAYERS=$n_layers DROP=$drop allennlp train $conf -s $dir --include-package src
