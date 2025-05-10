# Unsloth Multi-GPU Finetuning

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/isaiahbjork/unsloth_multi_gpu)

Set `/path/to/unsloth_multi_gpu` in the unsloth_multi.py file.

Set `os.environ["WANDB_API_KEY"]` and `os.environ["HF_TOKEN"]` in the unsloth_multi.py file.

Set `export PYTHONPATH=/path/to/unsloth_multi_gpu:$PYTHONPATH` in the run.sh file.

Set `dataset_name` to the name of the dataset you want to use in the unsloth_multi.py file.

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

Still testing and might not work or crash.


Need GPUs? Use my referral link on vast.ai:
https://cloud.vast.ai/?ref_id=126744&creator_id=126744&name=cuda%3A12.0.1-devel-ubuntu20.04