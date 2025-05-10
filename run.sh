#!/bin/bash

# Set environment variables
export PYTHONPATH=/root/unsloth_multi_gpu:$PYTHONPATH

# Run distributed training
python3 -m torch.distributed.run --nproc_per_node=NUM_GPUS --master_port=29500 unsloth_multi.py