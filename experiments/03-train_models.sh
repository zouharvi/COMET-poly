# baseline
sbatch_gpu_bigg "train_multicand_0t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_0t00s.yaml"
# one additional translation
sbatch_gpu_bigg "train_multicand_1t00s_nograd" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t00s_nograd.yaml"
sbatch_gpu_bigg "train_multicand_1t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t00s.yaml"
sbatch_gpu_bigg "train_multicand_1t01s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t01s.yaml"
sbatch_gpu_bigg "train_multicand_1t10s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_1t10s.yaml"
# two additional translations
sbatch_gpu_bigg "train_multicand_2t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t00s.yaml"
sbatch_gpu_bigg "train_multicand_2t20s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t20s.yaml"
sbatch_gpu_bigg "train_multicand_2t02s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_2t02s.yaml"
# three additional translations
sbatch_gpu_bigg "train_multicand_3t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_3t00s.yaml"
sbatch_gpu_bigg "train_multicand_3t30s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_3t30s.yaml"
sbatch_gpu_bigg "train_multicand_3t03s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_3t03s.yaml" # failed
# four additional translations
sbatch_gpu_bigg "train_multicand_4t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_4t00s.yaml" # todo training
sbatch_gpu_bigg "train_multicand_4t40s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_4t40s.yaml" # todo training
sbatch_gpu_bigg "train_multicand_4t04s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_4t04s.yaml" # todo training
# five additional translations
sbatch_gpu_bigg "train_multicand_5t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_5t00s.yaml" # todo training
sbatch_gpu_bigg "train_multicand_5t50s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_5t50s.yaml" # todo training
sbatch_gpu_bigg "train_multicand_5t05s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_5t05s.yaml" # todo training

# reference-based
sbatch_gpu_bigg "train_multicand_ref_0t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_ref_0t00s.yaml"
sbatch_gpu_bigg "train_multicand_ref_1t00s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_ref_1t00s.yaml"
sbatch_gpu_bigg "train_multicand_ref_1t01s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_ref_1t01s.yaml"
sbatch_gpu_bigg "train_multicand_ref_1t10s" "comet-multi-cand-train --cfg comet_multi_cand/configs/experimental/model_multicand_ref_1t10s.yaml"