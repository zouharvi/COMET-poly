sbatch_gpu_big "pw_dedup" "comet-train --cfg configs/experimental/pairwise_model.yaml"

sbatch_gpu_long "secondrun_pairwise" "comet-train --cfg configs/experimental/pairwise_model.yaml"
sbatch_gpu_long "secondrun_da" "comet-train --cfg configs/experimental/referenceless_model.yaml"

sbatch_gpu_short "eval_secondrun_da" "python3 experiments/07-evaluate_da_model.py lightning_logs/version_18089135/checkpoints/epoch=2-step=35484-val_kendall=0.292.ckpt"
sbatch_gpu_short "eval_secondrun_pw" "python3 experiments/06-evaluate_pw_model.py lightning_logs/version_18089134/checkpoints/epoch=0-step=24307-val_accuracy=0.650.ckpt"

sbatch_gpu_bigg "train_anchor_metric" "comet-train --cfg configs/experimental/model_anchor.yaml"
sbatch_gpu_bigg "train_anchor_score_metric" "comet-train --cfg configs/experimental/model_anchor_score.yaml"
sbatch_gpu_bigg "train_anchor_baseline" "comet-train --cfg configs/experimental/model_anchor_baseline.yaml"