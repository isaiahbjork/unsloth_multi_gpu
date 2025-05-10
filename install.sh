#!/bin/bash

apt-get update && apt-get install -y git python3-pip \

pip install --upgrade pip && \
pip install torch==2.5.1 transformers[torch] trl peft bitsandbytes accelerate wandb huggingface_hub datasets pillow xformers && \
pip install unsloth_zoo && \    
pip install ninja packaging wheel hf_xet

echo "Installation complete!"