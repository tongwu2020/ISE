


## Installation

```bash
conda create -n ISE python=3.10 
conda activate ISE
pip install -r requirements.txt
pip install -e .["dev"]
```

## Download the pre-trained models

```bash
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir ./pretrained_models/llama3_1_8B_base

# you can download other models, please check the official documents of torchtune for more details
```

## Dataset

For training data, we store the data in the `data/train_data` folder, and the evaluation data is stored in the `data/eval_data` folder.


## Training

We provide the training recipes in the `recipes` folder, you can change the `recipes/configs` file to train other models.

Config files `llama3_1/8B_full_Ba_base.yaml` is the training recipe for llama3.1 8B base model without using ISE, `llama3_1/8B_full_BaT_base.yaml` is the training recipe for llama3.1 8B base model using ISE. 

`Ba` means UltraChat Baseline, `IF` mean System Follow, and `IH` means Instruct Hierarchy. Config files with `T` means using ISE.

```bash
cd recipes
tune run --nproc_per_node 4 full_finetune_distributed --config configs/llama3_1/8B_full_Ba_base.yaml batch_size=4 gradient_accumulation_steps=8 output_dir="./output/model/Llama-3_1-8B-Ba_base/" checkpointer.output_dir="./models/Llama-3_1-8B-Base-Ba/" epochs=3 

# We use 4 GPUs for training, you can change the number of GPUs by changing the --nproc_per_node argument, and change the batch size and gradient accumulation steps according to your GPU memory
```


## Evaluation

Similar to training, we provide the evaluation recipes in the `recipes` folder, you can change the `recipes/configs` file to evaluate other models.

Config files `llama3_1_eval.yaml` is the evaluation recipe for llama3.1 8B base model without using ISE, `llama3_1T_eval.yaml` is the evaluation recipe for llama3.1 8B base model using ISE.

We provide the evaluation data in the `data/eval_data` folder, and the evaluation results will be stored in the `output/local_eval` folder.




```bash
config="configs/llama3_1_eval.yaml"
model=$1 # the model name
checkpoint=$2 # the checkpoint number (0,1,2)
dataset=$3 # the dataset name (share_gpt_attack_0) 

python -u evaluate.py --config $config \
    checkpointer.checkpoint_dir="/scratch/gpfs/tw6664/IH/torchtune/models/$model/" \
    checkpointer.checkpoint_files=["meta_model_${checkpoint}.pt"] \
    device="cuda:0" \
    checkpointer.output_dir="../output/" \
    output_dir="../output/local_eval/" \
    batch_size=1 \
    dataset.source="../data/eval_data/$dataset.json" \
    enable_kv_cache=False
```



















