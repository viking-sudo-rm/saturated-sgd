#! /bin/sh

servers="vision-server6.corp.ai2"
clouds=""

for server in $servers; do
  echo "Syncing with $server"
  rsync -avz \
    --exclude .DS_Store \
    --exclude __pycache__ \
    --exclude .direnv \
    --exclude .envrc \
    --exclude data/datasets \
    --exclude .git \
    --exclude .vscode \
    --exclude venv \
    --exclude runs \
    ../saturating-will vivekr@${server}:~
done  
