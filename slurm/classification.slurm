#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
dataset=amazon

# model=roberta-base
# dir=/tmp/willm/saturated-sgd/sentiment/${model}
# MODEL=$model allennlp train $conf -s $dir --include-package src

model=roberta-base
optim=adamw
dir=/tmp/willm/saturated-sgd/classification/$dataset/lstm-$optim
conf=configs/classification/lstm.jsonnet
DATASET=$dataset OPTIM=$optim allennlp train $conf -s $dir --include-package=src

# model=roberta-base
# percent=0.2
# mode=RANDOM
# norm=WEIGHTS
# conf=configs/prune-sentiment.jsonnet
# dir=/tmp/willm/saturated-sgd/sentiment/$model-$mode-$norm-$percent
# MODEL=$model PERCENT=$percent MODE=$mode NORM=$norm allennlp train $conf -s $dir --include-package=src
