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
conf=/home/willm/code/saturated-sgd/configs/max_difference.jsonnet

n_hid=5
dir=/tmp/willm/saturated-sgd/$n_hid

N_HID=$n_hid allennlp train $conf -s $dir --include-package src
