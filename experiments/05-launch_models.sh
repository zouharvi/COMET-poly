function sbatch_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}


sbatch_gpu "firstrun_pairwise" "comet-train --cfg configs/experimental/pairwise_model.yaml"
sbatch_gpu "firstrun_da" "comet-train --cfg configs/experimental/referenceless_model.yaml"

# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python