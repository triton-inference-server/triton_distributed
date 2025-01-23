#! /bin/bash

#SBATCH --partition=batch
#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --job-name=coreai_tritoninference_triton3-vllm:benchmark
#SBATCH --nodes=2
#SBATCH --time=4:00:00
#SBATCH --exclusive

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all SLURM job logs and files
# Does not affect directories referenced in your script
export BASE_JOB_DIR=`pwd`
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

# Logging information
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run
# In the script, Dask must connect to a cluster through the Dask scheduler
# We recommend passing the path to a Dask scheduler's file in a
# nemo_curator.utils.distributed_utils.get_client call like the examples
export DEVICE="gpu"


# Make sure to mount the directories your script references
export BASE_DIR=`pwd`
export MOUNTS="${BASE_DIR}:${BASE_DIR}"

# Network interface specific to the cluster being used
# export UCX_IB_ROCE_LOCAL_SUBNET=y
export INTERFACE=eth3
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT="14GB"

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF
export DASK_DATAFRAME__QUERY_PLANNING=False


# =================================================================
# End easy customization
# =================================================================

mkdir -p $LOGDIR
mkdir -p $PROFILESDIR

# Start the container
srun \
    --container-mounts=/lustre/share/:/lustre/share/ \
    --container-image=gitlab-master.nvidia.com/dl/triton/triton-3/triton_with_vllm:20241219 \
    /workspace/examples/vllm/scripts/launch_workers.py \
    --model-ckpt /lustre/share/coreai_dlalgo_ci/artifacts/model/llama3.1_70b_pyt/safetensors_mode-instruct/hf-08b31c0_fp8 \
    --model-name llama \
    --baseline-tp-size 4 \
    --baseline-workers 4 \
    --context-max-batch-size 999999 \
    --max-model-len 3500 \
    --log-level INFO \
    --data-plane-backend nccl \
    --enable-prefix-caching