#!/bin/bash
# Check GitHub stats every hour
echo "Starting GitHub stats watcher..."

while true; do
    echo "=== $(date) ===" 
    cd /home/ma/autoresearch
    gh repo view bopalvelut-prog/autoresearch --json stargazerCount,forkCount,watchers
    echo "---"
    sleep 3600
done
