#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-multi-cand

# copy models
# scp -r euler:/cluster/work/sachan/vilem/COMET-multi-cand/lightning_logs/version_19759459/ lightning_logs/version_19759459/

# send data
# rsync -azP data/csv/* euler:/cluster/work/sachan/vilem/COMET-multi-cand/data/csv/