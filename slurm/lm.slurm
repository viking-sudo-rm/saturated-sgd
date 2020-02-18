#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/saturated-sgd/configs/lm.jsonnet

optim=adamw
c_drop=0.5 #0.5
l2=1.2e-6 # 1.2e-6

dir=/tmp/willm/saturated-sgd/lm/penn/${optim}-${c_drop}-${l2}
OPTIM=$optim C_DROP=$c_drop L2=$l2 allennlp train $conf -s $dir --include-package src
