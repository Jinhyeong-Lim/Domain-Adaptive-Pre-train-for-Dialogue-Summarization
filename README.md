# Dialogue Summarization

## Installation

`pip install -r requirements.txt`
> Note: The current code base has been tested with 2 Nvidia 3090 GPUs.
> We also use [wandb](https://wandb.ai/site) cloud-based logger 
> which you may need to register and login first.
 
### Data generation
[Download](https://drive.google.com/file/d/1nODYkmuTYAl3hy6dLYeeZFJM7RRZbyN6/view?usp=share_link) aihub dialogue summary dataset. 

```bash
python data_generator.py --path aihub_dialogue_summary --output data 
```

### Update tokenizer for dialogue summary dataset 
P01, P02와 같은 speaker id를 하나의 token으로 처리함.
1. Download `gogamza/kobart-base-v1` tokenizer
2. Replace `<unusedX>` tokens into others

```bash
python update_tokenizer.py
```

### Post-training
```bash
torchrun --nproc_per_node=2 run_post_train.py \
        --model_path "kobart-dialogue" \
        --model_name "gogamza/kobart-base-v1" \
        --run_name "kobart-post_train" \
        --do_train \
        --report_to "none" \
        --dialogue_max_seq_length 1024 \
        --train_file data/dialogue.json \
        --output_dir "checkpoints/kobart-post_train" \
        --learning_rate 0.0005 \
        --warmup_steps 50000 \
        --per_device_train_batch_size 8 \
        --max_steps 500000 \
        --save_strategy steps \
        --dataloader_num_workers 16 \
        --save_steps 10000 \
        --save_total_limit 3 \
        --gradient_accumulation_steps 1 \
        --logging_steps 100 
```

### Fine-tuning
```bash
torchrun --nproc_per_node=1 run_summarization.py \
  --model_path "kobart-dialogue" \
  --model_name "gogamza/kobart-base-v1" \
  --run_name "kobart-dialsumm" \
  --do_train \
  --do_eval \
  --do_predict \
  --report_to "none" \
  --train_file data/train.csv \
  --valididation_file data/valid.csv \
  --predict_file data/predict.csv \
  --max_source_length 512 \
  --max_target_length 128 \
  --output_dir "checkpoints/kobart-dialsumm" \
  --learning_rate 5e-5 \
  --warmup_steps 50 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 8 \
  --max_steps 1000 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --dataloader_num_workers 10 \
  --save_steps 10 \
  --eval_steps 10 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --label_smoothing_factor 0.1 \
  --gradient_accumulation_steps 2 \
  --overwrite_cache \
  --fp16 \
  --predict_with_generate
```

### Loss Graph
![Untitled (2)](https://user-images.githubusercontent.com/64317686/230779759-9d90e558-49ea-4e58-a197-e9da54061732.png)


### Performance

| Model | R1 | R2 | RL |
| :-----------: | :------------: | :------------: |:------------: |
| Naive Fine tuning   |  31.3613  |    17.6152    |     28.2803    |
| **Post training then Fine tuning**   |   **32.591**  |    **18.5439**    |     **29.4671**    |

