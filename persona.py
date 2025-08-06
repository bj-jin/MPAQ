from dataclasses import dataclass, field
from pyexpat import model
import copy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed,
    EarlyStoppingCallback,
)
import os
# import llama_patch
import sys
from loguru import logger
from typing import List, Optional, Mapping, Union, Tuple, Dict, Any, Callable
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.trainer_pt_utils import nested_detach
from peft import (
    TaskType,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM,
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
from datasets import load_dataset, concatenate_datasets
import argparse

cnt = 0

data_args = None
model_args = None
training_args = None

# tokenize the prompt and the label
def generate_and_tokenize_prompt(data_point, split, tokenizer, prompt_rolegen: str):
    prompt_inputs = []
    prompt_labels = []
    for i in range(len(data_point["argument"])):
        prompt = prompt_rolegen
        boundary_pos = prompt.find("### Output:") + len("### Output:")
        prompt_input = tokenizer.bos_token + prompt[:boundary_pos]
        prompt_label = prompt[boundary_pos:] + tokenizer.eos_token
        prompt_input = prompt_input.format(topic=data_point['topic'][i], argument=data_point['argument'][i])
        if split.find("train") != -1:
            prompt_label = prompt_label.format(persona_list=data_point["role"][i])
        prompt_inputs.append(prompt_input)
        prompt_labels.append(prompt_label)
    
    inputs = {}
    inputs["input_ids"] = [
        tokenizer.encode(prompt_input + prompt_label if split.find("train") != -1 else prompt_input, add_special_tokens=False)
        for prompt_input, prompt_label in zip(prompt_inputs, prompt_labels)
    ]
    
    if split.find("train") != -1:
        labels = copy.deepcopy(inputs["input_ids"])
        for i in range(len(labels)):
            input_length = len(tokenizer.encode(prompt_inputs[i], add_special_tokens=False))
            labels[i][:input_length] = [-100] * input_length
    else:
        # replace the label part of input_ids with -100
        labels = copy.deepcopy(inputs["input_ids"])
        for i in range(len(labels)):
            labels[i] = [-100] * len(labels[i])
    inputs["labels"] = labels
    
    # find the max length of the batch
    max_length = max(len(input_ids) for input_ids in inputs["input_ids"])
    # left padding the input_ids and the label_ids
    input_ids_padded = []
    label_ids_padded = []
    attention_masks = []
    for input_ids, label_ids in zip(inputs["input_ids"], inputs["labels"]):
        padding_length = max_length - len(input_ids)
        # left padding
        padded_input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        # replace the label part of input with -100
        padded_label_ids = [-100] * padding_length + label_ids
        # generate attention mask
        attention_mask = [0] * padding_length + [1] * len(input_ids)
        
        input_ids_padded.append(padded_input_ids)
        label_ids_padded.append(padded_label_ids)
        attention_masks.append(attention_mask)
    inputs["input_ids"] = input_ids_padded
    inputs["labels"] = label_ids_padded
    inputs["attention_mask"] = attention_masks
    
    if max_length > training_args.model_max_length:
        global cnt
        cnt += 1
    
    return inputs

def load_and_tokenize_dataset(data_path, split, tokenizer):
    data_rolegen = load_dataset('json', data_files=os.path.join(data_path, f"{split}.json"))
    # shuffle the training set
    if split.find("train") != -1:
        data_rolegen['train'] = data_rolegen['train'].shuffle(seed=training_args.seed)
    prompt_version = model_args.version[model_args.version.rfind("-")+1:]
    with open(os.path.join(model_args.prompt_root_path, f"prompt{prompt_version}.txt"), 'r', encoding='utf-8') as f:
        prompt_rolegen = f.read()
    
    data_rolegen['train'] = data_rolegen['train'].map(lambda x: generate_and_tokenize_prompt(x, split, tokenizer, prompt_rolegen), batched=True, batch_size=training_args.per_device_train_batch_size if split.find("train") != -1 else training_args.per_device_eval_batch_size)
    logger.info(f"out of boundary count: {cnt}")
    
    return data_rolegen['train']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    dropout: Optional[float] = field(default=0.1)
    max_new_tokens: Optional[int] = field(default=512)
    version: Optional[str] = field(default="v1")
    prompt_root_path: Optional[str] = field(default="prompt")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=True)



if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    generation_config = GenerationConfig(
        max_new_tokens=model_args.max_new_tokens,
        num_beams=1,
    )
    training_args.generation_config = generation_config
    training_args.save_total_limit = 2
    training_args.do_eval = True
    training_args.do_predict = True
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.metric_for_best_model = "loss"
    training_args.greater_is_better = False
    training_args.load_best_model_at_end = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        token=""
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    data_version = model_args.version[:model_args.version.find("-")]
    # load dataset
    train_dataset = load_and_tokenize_dataset(data_args.data_path, f"train{data_version}", tokenizer)
    print("train_dataset:", len(train_dataset), train_dataset[0])
    val_dataset = load_and_tokenize_dataset(data_args.data_path, f"val{data_version}", tokenizer)
    print("val_dataset:", len(val_dataset))
    
    # load model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, token="", cache_dir=training_args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, token="", cache_dir=training_args.cache_dir)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # define trainer
    class CustomTrainer(Trainer):
        def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            return SequentialSampler(self.train_dataset)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=1)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # training
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    trainer.train()
    
    # inference validation set
    training_args.per_device_eval_batch_size = 32
    val_dataset = load_and_tokenize_dataset(data_args.data_path, "val", tokenizer)
    with open(os.path.join(data_args.data_path, f"val.json"), "r") as f:
        val_data = json.load(f)
    val_dataloader = trainer.get_test_dataloader(val_dataset)
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=model_args.max_new_tokens, num_beams=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, early_stopping=True)
        decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_text = [text[text.find("### Output:")+len("### Output:"):].strip() for text in decoded_text]
        for j, text in enumerate(decoded_text):
            val_data[i*training_args.per_device_eval_batch_size+j]["type"] = 1
            val_data[i*training_args.per_device_eval_batch_size+j]["role"] = decoded_text[j]
        if i % 10 == 1:
            with open(os.path.join(training_args.output_dir, f"valgen.json"), "w") as f:
                json.dump(val_data, f, indent=4)
    with open(os.path.join(training_args.output_dir, f"valgen.json"), "w") as f:
        json.dump(val_data, f, indent=4)
    
    # inference test set
    training_args.per_device_eval_batch_size = 32
    test_dataset = load_and_tokenize_dataset(data_args.data_path, "test", tokenizer)
    with open(os.path.join(data_args.data_path, f"test.json"), "r") as f:
        test_data = json.load(f)
    test_dataloader = trainer.get_test_dataloader(test_dataset)
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=model_args.max_new_tokens, num_beams=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, early_stopping=True)
        decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_text = [text[text.find("### Output:")+len("### Output:"):].strip() for text in decoded_text]
        for j, text in enumerate(decoded_text):
            test_data[i*training_args.per_device_eval_batch_size+j]["type"] = 1
            test_data[i*training_args.per_device_eval_batch_size+j]["role"] = decoded_text[j]
        if i % 10 == 1:
            with open(os.path.join(training_args.output_dir, f"testgen.json"), "w") as f:
                json.dump(test_data, f, indent=4)
    with open(os.path.join(training_args.output_dir, f"testgen.json"), "w") as f:
        json.dump(test_data, f, indent=4)