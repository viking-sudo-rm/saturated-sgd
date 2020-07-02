#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
conf=configs/norm-sgd/agree.jsonnet

# dir=/tmp/willm/norm-sgd/agree
# TODO: run this experiment with fixed random seed.

optimizer=floor_sgd
lr=1e-3
min_step=1e-6
dir=/net/nfs2.corp/allennlp/willm/models/norm-sgd/agree/$optimizer-lr=$lr-min=$min_step

RNN=$rnn OPTIMIZER=$optimizer LR=$lr MIN_STEP=$min_step allennlp train $conf -s $dir --include-package src
