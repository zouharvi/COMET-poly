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

function sbatch_gpu_long() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=7-0 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=0-4 \
        --wrap="$JOB_WRAP";
}


sbatch_gpu_long "secondrun_pairwise" "comet-train --cfg configs/experimental/pairwise_model.yaml"
sbatch_gpu_long "secondrun_da" "comet-train --cfg configs/experimental/referenceless_model.yaml"

# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

sbatch_gpu_short "eval_firstrun_da" "python3 experiments/07-evaluate_da_model.py lightning_logs/version_18089129/checkpoints/epoch=2-step=3552-val_kendall=0.035.ckpt"
sbatch_gpu_short "eval_firstrun_pw" "python3 experiments/06-evaluate_pw_model.py lightning_logs/version_18088243/checkpoints/epoch=1-step=4298-val_accuracy=0.618.ckpt"