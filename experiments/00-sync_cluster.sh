#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-ranking

# copy models
# scp -r euler:/cluster/work/sachan/vilem/comet-ranking/lightning_logs/version_18089134/ lightning_logs/version_18089134/
# scp -r euler:/cluster/work/sachan/vilem/comet-ranking/lightning_logs/version_18089135/ lightning_logs/version_18089135/