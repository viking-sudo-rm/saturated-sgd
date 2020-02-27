path=/home/willm/code/

for idx in 1 3
do
    to_server="allennlp-server${idx}.corp.ai2"
    rsync -avz ${path}/saturated-sgd willm@${to_server}:${path}
done
