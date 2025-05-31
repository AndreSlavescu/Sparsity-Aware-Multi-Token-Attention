#!/usr/bin/bash

# Multi-Token Attention Training Experiments
# This script trains both softmax and sparsemax MTA variants of the 340M transformer

set -e  # Exit on any error

# Configuration
NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]; then
  export MASTER_ADDR="localhost"
fi
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT="0"
fi

# GPU Detection and Validation
echo "Detecting available GPUs..."

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs."
    echo "Please ensure you're running on a system with NVIDIA drivers installed."
    exit 1
fi

# Detect number of GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    # If CUDA_VISIBLE_DEVICES is set, count the devices
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    DETECTED_GPUS=${#GPU_ARRAY[@]}
    echo "CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
    echo "Detected $DETECTED_GPUS GPU(s) from CUDA_VISIBLE_DEVICES"
else
    # Use nvidia-smi to count GPUs
    DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $DETECTED_GPUS GPU(s) using nvidia-smi"
fi

# Validate GPU count
REQUIRED_GPUS=8
if [ "$DETECTED_GPUS" -ne "$REQUIRED_GPUS" ]; then
    echo ""
    echo "ERROR: Insufficient GPUs detected!"
    echo "Required: $REQUIRED_GPUS GPUs"
    echo "Detected: $DETECTED_GPUS GPUs"
    echo ""
    echo "This script is designed to run multi-token attention experiments using exactly 8 GPUs."
    echo "Please ensure you have 8 NVIDIA GPUs available or modify the NGPU parameter."
    echo ""
    echo "Available options:"
    echo "1. Run on a system with 8 GPUs"
    echo "2. Set CUDA_VISIBLE_DEVICES to specify 8 GPU IDs"
    echo "3. Modify NGPU variable (not recommended as this may affect training stability)"
    echo ""
    exit 1
fi

echo "✓ GPU validation passed: $DETECTED_GPUS GPUs detected"

# Display GPU information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -n $REQUIRED_GPUS
echo ""

# Set up experiment directories
EXPERIMENT_BASE="exp/mta-experiments"
DATE=$(date +%Y%m%d_%H%M)

# Common training parameters
COMMON_PARAMS="
  --job.config_file flame/models/fla.toml \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1"

echo "==============================================="
echo "Starting Multi-Token Attention Experiments"
echo "Date: $DATE"
echo "Using $NGPU GPUs across $NNODE node(s)"
echo "==============================================="

# Function to run training experiment
run_experiment() {
    local config_file=$1
    local experiment_name=$2
    local attention_type=$3
    
    echo ""
    echo "================================================"
    echo "Starting $experiment_name experiment"
    echo "Config: $config_file"
    echo "Attention Type: $attention_type"
    echo "================================================"
    
    # Set experiment directory
    DUMP_FOLDER="$EXPERIMENT_BASE/$experiment_name-$DATE"
    
    # Set WANDB environment variables for this experiment
    export WANDB_PROJECT="mta-experiments"
    export WANDB_NAME="$experiment_name-$DATE"
    export WANDB_RUN_ID="$experiment_name-$DATE-$(date +%s)"
    export WANDB_RESUME="allow"
    
    # Build full command
    FULL_PARAMS="--job.dump_folder $DUMP_FOLDER --model.config $config_file $COMMON_PARAMS"
    
    echo "Starting training with command:"
    echo "PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\" torchrun --nnodes=${NNODE} --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint \"${MASTER_ADDR}:${MASTER_PORT}\" --local-ranks-filter ${LOG_RANK} --role rank --tee 3 --log-dir $DUMP_FOLDER/logs -m flame.train $FULL_PARAMS"
    
    # Run training
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    torchrun --nnodes=${NNODE} \
      --nproc_per_node=${NGPU} \
      --rdzv_backend c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      --local-ranks-filter ${LOG_RANK} \
      --role rank \
      --tee 3 \
      --log-dir $DUMP_FOLDER/logs \
      -m flame.train \
      $FULL_PARAMS
    
    echo "Training completed for $experiment_name"
    
    # Convert DCP to HF format
    echo "Converting DCP checkpoints to HF format for $experiment_name..."
    python -m flame.utils.convert_dcp_to_hf \
      --path $DUMP_FOLDER \
      --step 20480 \
      --config $config_file \
      --tokenizer fla-hub/transformer-1.3B-100B
    
    echo "$experiment_name experiment completed successfully!"
    echo "Results saved in: $DUMP_FOLDER"
}

# Check if config files exist
if [ ! -f "flame/configs/softmax_mta_transformer_340M_with_pruning.json" ]; then
    echo "Error: softmax_mta_transformer_340M_with_pruning.json not found!"
    echo "Please ensure the config files are in flame/configs/"
    exit 1
fi

if [ ! -f "flame/configs/sparsemax_mta_transformer_340M_with_pruning.json" ]; then
    echo "Error: sparsemax_mta_transformer_340M_with_pruning.json not found!"
    echo "Please ensure the config files are in flame/configs/"
    exit 1
fi

# Make sure the register_mta.py is available
if [ ! -f "flame/register_mta.py" ]; then
    echo "Error: register_mta.py not found in flame/"
    echo "Please ensure the multi-token attention registry is available"
    exit 1
fi

# Create base experiment directory
mkdir -p $EXPERIMENT_BASE

echo "Config files found. Starting experiments..."

# Run Softmax Multi-Token Attention Experiment
run_experiment \
    "flame/configs/softmax_mta_transformer_340M_with_pruning.json" \
    "softmax-mta-340M" \
    "softmax_multi_token"

# Run Sparsemax Multi-Token Attention Experiment  
run_experiment \
    "flame/configs/sparsemax_mta_transformer_340M_with_pruning.json" \
    "sparsemax-mta-340M" \
    "sparsemax_multi_token"

echo ""
echo "==============================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "==============================================="
echo ""
echo "Results Summary:"
echo "- Softmax MTA:    $EXPERIMENT_BASE/softmax-mta-340M-$DATE"
echo "- Sparsemax MTA:  $EXPERIMENT_BASE/sparsemax-mta-340M-$DATE"
echo ""
echo "Both models have been converted to HuggingFace format and are ready for evaluation."
echo "Check the respective directories for:"
echo "  - Training logs"
echo "  - Checkpoints"
echo "  - Converted HF models"
echo "  - WANDB logs (if enabled)"
echo ""
echo "You can now run evaluation or inference with these trained models."
