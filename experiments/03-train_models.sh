# baseline
sbatch_gpu_bigg "train_polycand_0t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_0t00s.yaml"
# one additional translation
sbatch_gpu_bigg "train_polycand_1t00s_nograd" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_1t00s_nograd.yaml"
sbatch_gpu_bigg "train_polycand_1t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_1t00s.yaml"
sbatch_gpu_bigg "train_polycand_1t01s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_1t01s.yaml"
sbatch_gpu_bigg "train_polycand_1t10s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_1t10s.yaml"
# two additional translations
sbatch_gpu_bigg "train_polycand_2t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_2t00s.yaml"
sbatch_gpu_bigg "train_polycand_2t20s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_2t20s.yaml"
sbatch_gpu_bigg "train_polycand_2t02s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_2t02s.yaml"
# three additional translations
sbatch_gpu_bigg "train_polycand_3t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_3t00s.yaml"
sbatch_gpu_bigg "train_polycand_3t30s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_3t30s.yaml"
sbatch_gpu_bigg "train_polycand_3t03s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_3t03s.yaml"
# four additional translations
sbatch_gpu_bigg "train_polycand_4t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_4t00s.yaml"
sbatch_gpu_bigg "train_polycand_4t40s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_4t40s.yaml"
sbatch_gpu_bigg "train_polycand_4t04s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_4t04s.yaml"
# five additional translations
sbatch_gpu_bigg "train_polycand_5t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_5t00s.yaml"
sbatch_gpu_bigg "train_polycand_5t50s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_5t50s.yaml"
sbatch_gpu_bigg "train_polycand_5t05s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_5t05s.yaml"

# reference-based
sbatch_gpu_bigg "train_polycand_ref_0t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_ref_0t00s.yaml"
sbatch_gpu_bigg "train_polycand_ref_1t00s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_ref_1t00s.yaml"
sbatch_gpu_bigg "train_polycand_ref_1t01s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_ref_1t01s.yaml"
sbatch_gpu_bigg "train_polycand_ref_1t10s" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_ref_1t10s.yaml"

# polyic
sbatch_gpu_bigg "train_polyic_1t_src" "comet-poly-train --cfg comet_poly/configs/experimental/model_polyic_1t_src.yaml"
sbatch_gpu_bigg "train_polyic_1t_mt"  "comet-poly-train --cfg comet_poly/configs/experimental/model_polyic_1t_mt.yaml"
sbatch_gpu_bigg "train_polyic_1t_src_ref" "comet-poly-train --cfg comet_poly/configs/experimental/model_polyic_1t_src_ref.yaml"

# for public HF

sbatch_gpu_bigg "train_poly_cand1" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_1t00s.yaml"
sbatch_gpu_bigg "train_poly_cand3" "comet-poly-train --cfg comet_poly/configs/experimental/model_polycand_3t00s.yaml"
sbatch_gpu_bigg "train_poly_ic1" "comet-poly-train --cfg comet_poly/configs/experimental/model_polyic_1t_src.yaml"
sbatch_gpu_bigg "train_poly_ic3" "comet-poly-train --cfg comet_poly/configs/experimental/model_polyic_3t_src.yaml"
