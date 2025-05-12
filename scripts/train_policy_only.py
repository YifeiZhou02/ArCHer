import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch.nn as nn
import numpy as np

class PolicyOnlyAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2",
                 cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                 do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = '\n'):
        super(PolicyOnlyAgent, self).__init__()
        self.device = device
        self.accelerator = accelerator
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                             torch_dtype = torch.bfloat16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA")
            self.model.print_trainable_parameters()
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str

    def prepare(self):
        self.model = self.accelerator.prepare(self.model)

    def get_action(self, observation):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        context_len = obs_ids['attention_mask'].size(1)
        outputs = self.accelerator.unwrap_model(self.model).generate(**obs_ids,
                                                                     max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,
                                                                     pad_token_id = self.tokenizer.eos_token_id).cpu()
        outputs = outputs[:, context_len:]
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        if self.eos_str is not None:
            return [raw_a.split(self.eos_str)[0] for raw_a in raw_action]
        else:
            return raw_action

    def get_log_prob(self, observation, action):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],
                                    dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],
                                                        action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        if values is not None:
            return values[:, obs_ids["attention_mask"].size(1)-1:-1], torch.log(selected_prediction_probs)*action_ids["attention_mask"], action_ids["attention_mask"]
        else:
            return torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)

    # Các hàm liên quan đến critic đã bị loại bỏ hoàn toàn