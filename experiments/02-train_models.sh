sbatch_gpu_bigg "train_anchor_metric" "comet-train --cfg configs/experimental/model_anchor.yaml"
sbatch_gpu_bigg "train_anchor_score_metric" "comet-train --cfg configs/experimental/model_anchor_score.yaml"
sbatch_gpu_bigg "train_anchor_baseline" "comet-train --cfg configs/experimental/model_anchor_baseline.yaml"