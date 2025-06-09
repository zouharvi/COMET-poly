# take everything from the second line
for split in "train" "dev"; do
    for kind in "same_rand" "same_sim" "retrieval_minilm_11_src"; do
    cp data/csv/${split}_${kind}.csv data/csv/${split}_mix_${kind}.csv
    tail -n +2 data/csv/${split}_${kind}.csv >> data/csv/${split}_mix_${kind}.csv
done;
done;