#!/bin/bash
# Evaluation script for CDD on UDLM-QM9

set -e

export HF_HOME=/scratch/liyues_root/liyues1/zyluo/hf_cache

cd /home/zyluo/code/constrained-dllm

source /home/zyluo/miniconda3/etc/profile.d/conda.sh
conda activate cdd

echo "Running CDD evaluation..."
python evaluation/udlm_qm9_cdd_eval.py