#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-poly

# copy models
# scp -r euler:/cluster/work/sachan/vilem/COMET-poly/lightning_logs/version_19759459/ lightning_logs/version_19759459/

# send data
# rsync -azP data/csv/* euler:/cluster/work/sachan/vilem/COMET-poly/data/csv/

# scp euler:/cluster/work/sachan/vilem/COMET-poly/logs/output_simple_0t00s.out computed/output_simple_0t00s.out
# scp euler:/cluster/work/sachan/vilem/COMET-poly/logs/output_all_0t00s.out computed/output_all_0t00s.out

# scp euler:/cluster/work/sachan/vilem/COMET-poly/data/csv/comet-poly-jun08.tar ~/Downloads/