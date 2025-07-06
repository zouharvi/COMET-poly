#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-poly

# copy models
# scp euler:/cluster/work/sachan/vilem/COMET-poly/lightning_logs/polycand_1t00s/checkpoints/epoch\=4-step\=21975-val_kendall\=0.530.ckpt checkpoints/model.ckpt

# send data
# rsync -azP data/csv/* euler:/cluster/work/sachan/vilem/COMET-poly/data/csv/

# scp euler:/cluster/work/sachan/vilem/COMET-poly/logs/output_simple_0t00s.out computed/output_simple_0t00s.out
# scp euler:/cluster/work/sachan/vilem/COMET-poly/logs/output_all_0t00s.out computed/output_all_0t00s.out

# scp euler:/cluster/work/sachan/vilem/COMET-poly/data/csv/comet-poly-jun08.tar ~/Downloads/