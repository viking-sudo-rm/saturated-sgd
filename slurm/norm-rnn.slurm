#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --partition=allennlp_hipri
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
conf=configs/norm-rnn.jsonnet

scale=1000.0
dir=/net/nfs2.corp/allennlp/willm/models/norm-rnn/$scale

SCALE=$scale allennlp train $conf -s $dir --include-package src
