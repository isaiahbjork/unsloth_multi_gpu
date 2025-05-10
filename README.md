# Unsloth Multi-GPU Finetuning

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/isaiahbjork/unsloth_multi_gpu)

Set `/path/to/unsloth_multi_gpu` in the unsloth_multi.py file.

Set `os.environ["WANDB_API_KEY"]` and `os.environ["HF_TOKEN"]` in the unsloth_multi.py file.

Set `NUM_GPUS` to the number of GPUs you want to use in the run.sh file.

Install dependencies:
```bash
chmod +x install.sh
```

```bash
./install.sh
```

Run the training script:
```bash
chmod +x run.sh
```

```bash
./run.sh
```