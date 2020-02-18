#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/saturated-sgd/configs/prune-agree.jsonnet

percent=0.0
ste=false

if [ "$ste" = "true" ]; then
    dir=/tmp/willm/saturated-sgd/prune-agree/${percent}-ste
else
    dir=/tmp/willm/saturated-sgd/prune-agree/$percent
fi

PERCENT=$percent STE=$ste allennlp train $conf -s $dir --include-package src

# A baseline: just train with smaller number of activations.

# source /home/willm/anaconda3/bin/activate allennlp
# conf=/home/willm/code/saturated-sgd/configs/prune-agree-control.jsonnet

# n_hid=80

# dir=/tmp/willm/saturated-sgd/prune-agree/baseline-$n_hid
# N_HID=$n_hid allennlp train $conf -s $dir --include-package src