from dataclasses import dataclass, field
from pyexpat import model
import re
import gc
import math
import copy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
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
)
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error, root_mean_squared_log_error, ndcg_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
from datasets import load_dataset, concatenate_datasets
import argparse

cnt = 0

data_args = None
model_args = None
training_args = None

class MyModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.args = model_args
        self.llama = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, token="", cache_dir=training_args.cache_dir)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.dropout,
        )
        self.llama = get_peft_model(self.llama, peft_config)
        self.llama.print_trainable_parameters()
        
        self.regressor = torch.nn.Linear(config.hidden_size, 1)
    def generate(self, **kwargs):
        return self.llama.generate(**kwargs)
    def save_pretrained(
        self, 
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        self.llama.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization)
        torch.save(self.regressor.state_dict(), os.path.join(save_directory, "regressor.pt"))
    def forward(self,input_ids=None,attention_mask=None,labels=None,score=None,score_pos=None,score_len=None,type=None,**kwargs):
        outputs = self.llama.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
        loss = outputs.loss
        
        regressed_num = 0
        last_hidden_states = outputs.hidden_states[-1]
        score_logits = torch.empty(0, 1, device=input_ids.device)
        score_loss = torch.tensor(0.0, device=last_hidden_states.device)
        for i in range(input_ids.size(0)):
            regressed_num += 1
            score_representation = torch.mean(last_hidden_states[i, score_pos[i].item():score_pos[i].item()+score_len[i].item(), :], dim=0, keepdim=True)
            tmp_score_logits = self.regressor(score_representation)
            
            score_logits = torch.cat((score_logits, tmp_score_logits), dim=0)
            
            loss_fct = torch.nn.L1Loss()
            score_loss += loss_fct(tmp_score_logits.view(-1), score[i].view(-1))
            
        # average the loss
        if regressed_num > 0 and self.training:
            score_loss /= regressed_num
            loss += score_loss
        
        return loss, (score_logits)
    
        

def compute_metrics(pred):
    score_preds = pred.predictions.flatten()
    # average the prediction scores of every 4 samples
    score_preds = [np.mean([score_preds[i], score_preds[i+1], score_preds[i+2], score_preds[i+3]]) for i in range(0, len(score_preds), 4)]
    score_preds = np.maximum(score_preds, 0.0)
    score_labels = pred.label_ids
    # take the first sample of every 4 samples as the label
    score_labels = [score_labels[i] for i in range(0, len(score_labels), 4)]
    
    pearson = pearsonr(score_labels, score_preds)[0]
    spearman = spearmanr(score_labels, score_preds)[0]
    kendall = kendalltau(score_labels, score_preds)[0]
    mae = mean_absolute_error(score_labels, score_preds)
    ndcg10 = ndcg_score([score_labels], [score_preds], k=10)
    ndcg15 = ndcg_score([score_labels], [score_preds], k=15)
    ndcg50 = ndcg_score([score_labels], [score_preds], k=50)
    ndcg100 = ndcg_score([score_labels], [score_preds], k=100)
    ndcg200 = ndcg_score([score_labels], [score_preds], k=200)
    ndcg500 = ndcg_score([score_labels], [score_preds], k=500)
    ndcg1000 = ndcg_score([score_labels], [score_preds], k=1000)
    
    return {
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "mae": mae,
        "ndcg10": ndcg10,
        "ndcg15": ndcg15,
        "ndcg50": ndcg50,
        "ndcg100": ndcg100,
        "ndcg200": ndcg200,
        "ndcg500": ndcg500,
        "ndcg1000": ndcg1000,
    }

# tokenize the prompt and the label
def generate_and_tokenize_prompt(data_point, split, tokenizer, prompt_sw: str):
    prompt_inputs = []
    prompt_labels = []
    for i in range(len(data_point["argument"])):
        prompt = prompt_sw
        boundary_pos = prompt.find("### Output:") + len("### Output:")
        prompt_input = tokenizer.bos_token + prompt[:boundary_pos]
        prompt_label = prompt[boundary_pos:] + tokenizer.eos_token
        prompt_input = prompt_input.format(topic=data_point['topic'][i], argument=data_point['argument'][i], role=data_point['role'][i], role_profile=data_point["role_profile"][i])
        if split.find("train") != -1:
            prompt_label = prompt_label.format(strengths=data_point["strengths"][i], weaknesses=data_point["weaknesses"][i], score=min(int(data_point["score"][i]*10), 9))
        prompt_inputs.append(prompt_input)
        prompt_labels.append(prompt_label)
    
    inputs = {}
    inputs["input_ids"] = [
        tokenizer.encode(prompt_input + prompt_label if split.find("train") != -1 or split.find("val") != -1 or split.find("gen") != -1 else prompt_input, add_special_tokens=False)
        for prompt_input, prompt_label in zip(prompt_inputs, prompt_labels)
    ]
    
    if split.find("train") != -1 or split.find("val") != -1 or split.find("gen") != -1:
        labels = copy.deepcopy(inputs["input_ids"])
        for i in range(len(labels)):
            input_length = len(tokenizer.encode(prompt_inputs[i], add_special_tokens=False))
            labels[i][:input_length] = [-100] * input_length
    else:
        # set the label to -100
        labels = copy.deepcopy(inputs["input_ids"])
        for i in range(len(labels)):
            labels[i] = [-100] * len(labels[i])
    inputs["labels"] = labels
    
    # find the max length of the batch
    max_length = max(len(input_ids) for input_ids in inputs["input_ids"])
    # manually left padding
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
    
    # find the position of the score in the input_ids
    score_poses = []
    score_lens = []
    score_token_start_ids = [14711, 9442, 512]
    score_token_end_ids = [933, 58, 26197, 82, 5787]
    score_token_start_ids_len = len(score_token_start_ids)
    score_token_end_ids_len = len(score_token_end_ids)
    if split.find("train") != -1 or split.find("val") != -1 or split.find("gen") != -1:
        for i in range(len(inputs["input_ids"])):
            score_start_pos = -1
            score_end_pos = -1
            for j in range(max_length - score_token_start_ids_len, 0, -1):
                if inputs["input_ids"][i][j:j+score_token_start_ids_len] == score_token_start_ids:
                    score_start_pos = j + score_token_start_ids_len
                    break
            assert score_start_pos >= 0
            score_poses.append(score_start_pos-1)
            score_lens.append(1)
    else:
        score_poses = [max_length-2] * len(inputs["input_ids"])
        score_lens = [1] * len(inputs["input_ids"])
    inputs["score_pos"] = score_poses
    inputs["score_len"] = score_lens
    
    return inputs

def load_and_tokenize_dataset(data_path, split, tokenizer):
    data = load_dataset('json', data_files=os.path.join(data_path, f"{split}.json"))
    # shuffle the training set
    if split.find("train") != -1:
        data['train'] = data['train'].shuffle(seed=training_args.seed)
    prompt_version = model_args.version[model_args.version.rfind("-")+1:]
    with open(os.path.join(model_args.prompt_root_path, f"prompt{prompt_version}.txt"), 'r', encoding='utf-8') as f:
        prompt_sw = f.read()
    
    data['train'] = data['train'].map(lambda x: generate_and_tokenize_prompt(x, split, tokenizer, prompt_sw), batched=True, batch_size=training_args.per_device_train_batch_size if split.find("train") != -1 else training_args.per_device_eval_batch_size)
    logger.info(f"out of boundary count: {cnt}")
    
    return data['train']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B")
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=128)
    dropout: Optional[float] = field(default=0.1)
    max_new_tokens: Optional[int] = field(default=512)
    version: Optional[str] = field(default="qualityassessment-qualityassessment")
    prompt_root_path: Optional[str] = field(default="data/30k")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
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
    training_args.do_eval = True
    training_args.do_predict = True
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.metric_for_best_model = "pearson"
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True
    training_args.label_names = ["score"]

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
    
    # load the dataset
    version_list = model_args.version.split("-")
    data_version = version_list[0]
    train_dataset = load_and_tokenize_dataset(data_args.data_path, f"train{data_version}", tokenizer)
    print("train_dataset:", len(train_dataset), train_dataset[0])
    val_dataset = load_and_tokenize_dataset(data_args.data_path, f"val{data_version}", tokenizer)
    print("val_dataset:", len(val_dataset))
    
    # load the model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, token="", cache_dir=training_args.cache_dir)
    model = MyModel(config)
    model.llama.enable_input_require_grads()
    
    embedding_size = model.llama.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.llama.resize_token_embeddings(len(tokenizer))

    # define the trainer
    class CustomTrainer(Trainer):
        def _load_best_model(self):
            self.model.llama.load_adapter(self.state.best_model_checkpoint, self.model.llama.active_adapter)
            self.model.regressor.load_state_dict(torch.load(os.path.join(self.state.best_model_checkpoint, "regressor.pt"), weights_only=True))
            logger.info(f"Best adapter and regressors loaded successfully from {self.state.best_model_checkpoint}!")
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            self.model.llama.load_adapter(resume_from_checkpoint, self.model.llama.active_adapter)
            self.model.regressor.load_state_dict(torch.load(os.path.join(resume_from_checkpoint, "regressor.pt"), weights_only=True))
            logger.info(f"Last Adapter and regressors loaded successfully from {resume_from_checkpoint}!")
        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)
        
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    # train the model
    trainer.train()
    
    # infer the rationale
    training_args.per_device_eval_batch_size = 8
    test_dataset = load_and_tokenize_dataset(data_args.data_path, f"test{data_version}", tokenizer)
    with open(os.path.join(data_args.data_path, f"test{data_version}.json"), "r") as f:
        test_data = json.load(f)
    test_dataloader = trainer.get_test_dataloader(test_dataset)
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=model_args.max_new_tokens, num_beams=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, early_stopping=True)
        decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_text = [text[text.find("### Output:")+len("### Output:"):].strip() for text in decoded_text]
        for j, text in enumerate(decoded_text):
            test_data[i*training_args.per_device_eval_batch_size+j]["answers"] = decoded_text[j]
        if i % 10 == 4:
            with open(os.path.join(training_args.output_dir, f"testgen.json"), "w") as f:
                json.dump(test_data, f, indent=4)
    with open(os.path.join(training_args.output_dir, f"testgen.json"), "w") as f:
        json.dump(test_data, f, indent=4)
    
    # test the prediction scores
    training_args.per_device_eval_batch_size = 8
    with open(os.path.join(training_args.output_dir, f"testgen.json"), "r") as f:
        test_data_gen = json.load(f)
    test_dataset_gen = load_and_tokenize_dataset(training_args.output_dir, f"testgen", tokenizer)
    test_results = trainer.predict(test_dataset_gen)
    test_metrics = test_results.metrics
    test_preds = test_results.predictions.flatten()
    test_texts = [str(test_preds[i]) + "\t" + str(test_preds[i+1]) + "\t" + str(test_preds[i+2]) + "\t" + str(test_preds[i+3]) for i in range(0, len(test_preds), 4)]
    logger.info(f"test_metrics: {test_metrics}")
    output_test_metric_file = os.path.join(training_args.output_dir, f"test_metrics.txt")
    output_test_text_file = os.path.join(training_args.output_dir, f"test_text.txt")
    output_test_score_file = os.path.join(training_args.output_dir, f"test.json")
    with open(output_test_metric_file, "w") as f:
        f.write(json.dumps(test_metrics, indent=4))
    logger.info(f"test_metrics saved to {output_test_metric_file}")
    with open(output_test_text_file, "w") as f:
        # test_texts is a 2D array, each row is the 4 predictions of a sample
        for text in test_texts:
            f.write(text + "\n")
    logger.info(f"test_texts saved to {output_test_text_file}")
    with open(os.path.join(data_args.data_path, "test.json"), "r") as f:
        test_data = json.load(f)
        for i, item in enumerate(test_data):
            test_data[i]["role1"] = test_data_gen[i*4]["role"]
            test_data[i]["role_profile1"] = test_data_gen[i*4]["role_profile"]
            test_data[i]["infer_answers_role1"] = test_data_gen[i*4]["answers"] if "answers" in test_data_gen[i*4] else ""
            test_data[i]["infer_score_role1"] = float(test_preds[i*4])
            test_data[i]["role2"] = test_data_gen[i*4+1]["role"]
            test_data[i]["role_profile2"] = test_data_gen[i*4+1]["role_profile"]
            test_data[i]["infer_answers_role2"] = test_data_gen[i*4+1]["answers"] if "answers" in test_data_gen[i*4+1] else ""
            test_data[i]["infer_score_role2"] = float(test_preds[i*4+1])
            test_data[i]["role3"] = test_data_gen[i*4+2]["role"]
            test_data[i]["role_profile3"] = test_data_gen[i*4+2]["role_profile"]
            test_data[i]["infer_answers_role3"] = test_data_gen[i*4+2]["answers"] if "answers" in test_data_gen[i*4+2] else ""
            test_data[i]["infer_score_role3"] = float(test_preds[i*4+2])
            test_data[i]["role4"] = test_data_gen[i*4+3]["role"]
            test_data[i]["role_profile4"] = test_data_gen[i*4+3]["role_profile"]
            test_data[i]["infer_answers_role4"] = test_data_gen[i*4+3]["answers"] if "answers" in test_data_gen[i*4+3] else ""
            test_data[i]["infer_score_role4"] = float(test_preds[i*4+3])
            test_data[i]["infer_score"] = float(np.mean([test_preds[i*4], test_preds[i*4+1], test_preds[i*4+2], test_preds[i*4+3]]))
        with open(output_test_score_file, "w") as f:
            json.dump(test_data, f, indent=4)
    logger.info(f"test_scores saved to {output_test_score_file}")