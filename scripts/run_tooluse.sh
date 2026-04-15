#!/bin/bash
# Train CaOPD on the Tool-Use dataset (SDFT backbone)
# Tested on a single NVIDIA H200 GPU.

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="outputs/caopd_tooluse"

accelerate launch --num_processes 1 main.py \
    --model_name $MODEL_NAME \
    --dataset_name tooluse \
    --output_dir $OUTPUT_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --num_prompts_per_batch 32 \
    --num_generations 8 \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --ref_model_mixup_alpha 0.01 \
    --teacher_context_mode demo \
    --seed 42
