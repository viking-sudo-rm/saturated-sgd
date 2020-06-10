#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1
#SBATCH --exclude=allennlp-server1,allennlp-server2,allennlp-server3
#SBATCH --partition=allennlp_hipri

source /home/willm/anaconda3/bin/activate allennlp
cd /home/willm/code/saturated-sgd
conf=/home/willm/code/saturated-sgd/configs/frozen_finetune.jsonnet

# FIXME: Why doesn't XLNet work here?
# bert-base-cased, roberta-base, t5-base, xlnet-base-cased
model=xlnet-base-cased
dir=/net/nfs2.corp/allennlp/willm/models/frozen-finetune/$model

MODEL=$model allennlp train $conf -s $dir \
    --include-package=src.data \
    --include-package=src.modules.pretrained_transformer_embedder_frozen \
    --include-package=src.models \
    --include-package=src.metrics