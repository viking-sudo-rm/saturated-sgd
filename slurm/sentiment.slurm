#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
conf=configs/sentiment.jsonnet

# model=roberta-base
# dir=/tmp/willm/saturated-sgd/sentiment/${model}
# MODEL=$model allennlp train $conf -s $dir --include-package src

conf=configs/sentiment-lstm.jsonnet
dir=/tmp/willm/saturated-sgd/sentiment/lstm
allennlp train $conf -s $dir --include-package src
