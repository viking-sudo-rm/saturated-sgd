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
conf=/home/willm/code/saturated-sgd/configs/anbn.jsonnet

for n_hid in {1..64}
do
    mkdir /tmp/willm/saturated-sgd/anbn/$n_hid
    for trial in {1..10}
    do
        dir=/tmp/willm/saturated-sgd/anbn/$n_hid/$trial
        N_HID=$n_hid allennlp train $conf -s $dir --include-package src
    done
done
