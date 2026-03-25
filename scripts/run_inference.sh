#!/bin/bash
# ProAR Inference Script
# Run molecular dynamics trajectory prediction on protein structures.
#
# Usage:
#   bash scripts/run_inference.sh
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Prepare data (Atlas test set example):
#        bash scripts/download_atlas_test.sh ./data/atlas_test
#        python scripts/preprocess_esm.py --data_dir ./data/atlas_test --device cuda
#   3. Set DATA_DIR below to point to your prepared data directory.
#   4. Model checkpoints are auto-downloaded from HuggingFace Hub on first run.
#      To use local checkpoints instead, set the paths below.
#
# Input data format:
#   DATA_DIR/
#     protein_A/
#       init.pdb            # Initial frame PDB file
#       esm_seq.npy         # (num_res, 1280) ESM sequence representation
#       esm_pair.npy        # (num_res, num_res, 20) ESM pair representation
#     protein_B/
#       ...

# ---- Configuration (modify as needed) ----

# Path to your data directory (REQUIRED)
DATA_DIR="/path/to/your/data"

# Number of autoregressive steps for trajectory rollout
AR_STEPS=42

# Sampling type: "naive" or "cold"
SAMPLING_TYPE="naive"

# Whether to refine intermediate predictions
REFINE=True

# Optional: local checkpoint paths (leave empty to auto-download from HuggingFace Hub)
FORECASTER_CKPT=""
INTERPOLATOR_CKPT=""
INTERPOLATOR_CONFIG=""

# ---- Run inference ----

CMD="python run.py \
    experiment=atlas \
    datamodule.data_dir=${DATA_DIR} \
    module.autoregressive_steps=${AR_STEPS} \
    diffusion.sampling_type=${SAMPLING_TYPE} \
    diffusion.refine_intermediate_predictions=${REFINE}"

# Append optional local checkpoint paths
if [ -n "$FORECASTER_CKPT" ]; then
    CMD="$CMD ckpt_path=${FORECASTER_CKPT}"
fi

if [ -n "$INTERPOLATOR_CKPT" ]; then
    CMD="$CMD diffusion.interpolator_local_checkpoint_path=${INTERPOLATOR_CKPT}"
fi

if [ -n "$INTERPOLATOR_CONFIG" ]; then
    CMD="$CMD diffusion.hydra_local_config_path=${INTERPOLATOR_CONFIG}"
fi

echo "Running: $CMD"
eval "$CMD"
