#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --partition=allennlp_hipri
#SBATCH --exclude=allennlp-server1,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
# Options for dataset are: ag  amazon  chemprot  citation_intent  hyperpartisan_by_article  imdb  rct-20k  sciie
dataset=amazon

# AHHHHHH I'M TRAPPED WILL HELP
# model=roberta-base
# dir=/tmp/willm/saturated-sgd/sentiment/${model}
# MODEL=$model allennlp train $conf -s $dir --include-package src

model=roberta-base
optim=adamw
percent=0.0
dir=/tmp/willm/saturated-sgd/classification/$dataset/lstm-$optim-ortho
conf=configs/classification/lstm.jsonnet
DATASET=$dataset OPTIM=$optim PERCENT=$percent allennlp train $conf -s $dir --include-package=src

# model=roberta-base
# percent=0.2
# mode=RANDOM
# norm=WEIGHTS
# conf=configs/prune-sentiment.jsonnet
# dir=/tmp/willm/saturated-sgd/sentiment/$model-$mode-$norm-$percent
# MODEL=$model PERCENT=$percent MODE=$mode NORM=$norm allennlp train $conf -s $dir --include-package=src
