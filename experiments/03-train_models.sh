# baseline
sbatch_gpu_bigg "train_multicand_0t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_0t00s.yaml"
# one additional translation
sbatch_gpu_bigg "train_multicand_1t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t00s.yaml"
sbatch_gpu_bigg "train_multicand_1t01s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t01s.yaml"
sbatch_gpu_bigg "train_multicand_1t10s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t10s.yaml"
# two additional translations
sbatch_gpu_bigg "train_multicand_2t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t00s.yaml"
sbatch_gpu_bigg "train_multicand_2t20s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t20s.yaml"
sbatch_gpu_bigg "train_multicand_2t02s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t02s.yaml"