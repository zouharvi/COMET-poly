# for public HF models

# train on all the data
cp data/csv/train_same_rand.csv data/csv/train_same_rand_wmt25.csv
tail -n +2 data/csv/test_same_rand.csv >> data/csv/train_same_rand_wmt25.csv
cp data/csv/train_retrieval_minilm_11_src.csv data/csv/train_retrieval_minilm_11_src_wmt25.csv
tail -n +2 data/csv/test_retrieval_minilm_11_src.csv >> data/csv/train_retrieval_minilm_11_src_wmt25.csv

sbatch_gpu_bigg "train_poly_cand1" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polycand_1t00s_wmt25.yaml"
sbatch_gpu_bigg "train_poly_cand2" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polycand_2t00s_wmt25.yaml"
sbatch_gpu_bigg "train_poly_cand3" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polycand_3t00s_wmt25.yaml"
sbatch_gpu_bigg "train_poly_ic1" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polyic_1t_src_wmt25.yaml"
sbatch_gpu_bigg "train_poly_ic3" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polyic_3t_src_wmt25.yaml"
# baseline
sbatch_gpu_bigg "train_poly_cand0" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polycand_0t00s_wmt25.yaml"

# baseline without wmt24
sbatch_gpu_bigg "train_polycand_0t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_0t00s.yaml"


# continue cancelled run
sbatch_gpu_bigg "train_poly_ic3" "comet-poly-train --cfg comet_poly/configs/experimental/wmt25/model_polyic_3t_src_wmt25_cont.yaml --load_from_checkpoint lightning_logs/polyic_3t_src_wmt25/checkpoints/epoch=2-step=15477-val_kendall=0.470.ckpt"

# copy models
huggingface-cli upload zouharvi/COMET-poly-ic3-wmt25 . .