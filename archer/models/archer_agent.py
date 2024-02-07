import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
from typing import Tuple
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from LLM_rep_RL.models.critic import DoubleCritic

class ArcherAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2", cache_dir = '~/.cache', dropout = 0.5,\
                  do_sample = True, temperature = 1.0, max_new_tokens = 32):
        super(ArcherAgent, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        self.policy_lm = policy_lm
        self.critic = DoubleCritic(device, accelerator, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.target_critic = DoubleCritic(device, accelerator, cache_dir = cache_dir, in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)
        self.tokenizer =AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
    
    def prepare(self):
        # self.model = self.accelerator.prepare(self.model)
        # self.critic.prepare()
        # self.target_critic.prepare()
        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)

    def get_action(self, observation):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # obs_embeds = self.accelerator.unwrap_model(self.model).get_input_embeddings()(obs_ids["input_ids"])
        # print(inputs_embeds.shape)
        # outputs = self.model.generate(inputs_embeds=obs_embeds, attention_mask=obs_ids['attention_mask'],\
        #                                max_new_tokens=32, do_sample=self.do_sample, temperature = self.temperature,\
        #                                pad_token_id = self.tokenizer.eos_token_id).cpu()
        context_len = obs_ids['attention_mask'].size(1)
        outputs = self.accelerator.unwrap_model(self.model).generate(**obs_ids,\
                                    max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,\
                                    pad_token_id = self.tokenizer.eos_token_id).cpu()
        outputs = outputs[:, context_len:]
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        return [raw_a.split('\n')[0] + '\n' for raw_a in raw_action]

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model = detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model = detach_model)
    
    def get_target_q(self, observation, action, detach_model=False):
        return self.target_critic.get_q(observation, action, detach_model = detach_model)

    def get_log_prob(self, observation, action):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
                                dim = 1)
        outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        if values is not None:
            return values[:, obs_ids["attention_mask"].size(1)-1:-1], torch.log(selected_prediction_probs)*action_ids["attention_mask"], action_ids["attention_mask"]
        else:
            return torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)

    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )


