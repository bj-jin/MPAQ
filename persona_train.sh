export CUDA_VISIBLE_DEVICES=0
export DATASET='30k'
export MODEL_PARENT_DIR=meta-llama
export MODEL_DIR=Meta-Llama-3.1-8B
export MODEL_MAX_LENGTH=768

# 模型细节版本-数据集版本-prompt版本
export VERSION=persona-persona
export LEARNING_RATE=1e-4
export NUM_TRAIN_EPOCHS=12
export GRADIENT_ACCUMULATION_STEPS=2
export WARMUP_RATIO=0.0
export DROPOUT=0.1
export LORA_R=32
export LORA_ALPHA=64
export SEED=42
export OUTPUT_DIR=    # type your output dir here
export PROMPT_ROOT_PATH=data/${DATASET}/$TYPE

export MAX_NEW_TOKENS=256

### 对学习率进行调参
export LEARNING_RATE=1e-4
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python persona_train.py \
    --model_name_or_path $MODEL_PARENT_DIR/$MODEL_DIR \
    --data_path data/$DATASET \
    --output_dir $OUTPUT_DIR \
    --version $VERSION \
    --prompt_root_path $PROMPT_ROOT_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy epoch \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type constant \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --dropout $DROPOUT \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio $WARMUP_RATIO \
    --seed $SEED \
    --bf16 True \
    --max_new_tokens $MAX_NEW_TOKENS \
    --gradient_checkpointing True