
echo "p=$p"

allennlp evaluate /tmp/willm/saturated-sgd/prune-agree/0.0/model.tar.gz \
    /net/nfs.corp/allennlp/willm/data/rnn_agr_simple/numpred.val \
    -o "{\"model.seq2seq_encoder.percent\":$p, \"model.seq2seq_encoder.prune_saturated\": false}" \
    --include-package=src

# allennlp evaluate /tmp/willm/saturated-sgd/lm/penn/adamw-0.5-1.2e-6/model.tar.gz \
#     /net/nfs.corp/allennlp/willm/data/penn/valid.txt \
#     -o "{\"model.contextualizer.percent\": $p}" \
#     --include-package=src
