#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-ranking

# copy models
# scp -r euler:/cluster/work/sachan/vilem/comet-ranking/lightning_logs/version_19759459/ lightning_logs/version_19759459/