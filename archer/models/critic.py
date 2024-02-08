import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
class DoubleCritic(torch.nn.Module):
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        super(DoubleCritic, self).__init__()
        self.device = device
        self.accelerator = accelerator
        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer.truncation_side = 'left'
        self.critic1 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.critic2 = nn.Sequential(nn.Linear(in_dim*2, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.v_critic1 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
        self.v_critic2 = nn.Sequential(nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, in_dim),\
                                    nn.ReLU(),\
                                    nn.Linear(in_dim, out_dim)).to(device)
    
    # def prepare(self):
    #     self.base_lm, self.critic1, self.critic2, self.v_critic1, self.v_critic2 = \
    #         self.accelerator.prepare(self.base_lm, self.critic1, self.critic2, self.v_critic1, self.v_critic2)

    def forward(self, observation, action, detach_model=False):
        state_actions = [o + a for o,a in zip(observation, action)]
        obs_ids = self.base_tokenizer(observation, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        # breakpoint()
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
        action_ids = self.base_tokenizer(action, padding = True, return_tensors='pt', max_length=512, truncation = True).to(self.device)
        # breakpoint()
        if detach_model:
            with torch.no_grad():
                action_states = self.base_lm(**action_ids).pooler_output
        else:
            action_states = self.base_lm(**action_ids).pooler_output
        q_states = torch.cat([lm_states, action_states], dim = 1)
        # print(action.size())
        return self.critic1(q_states), self.critic2(q_states), self.v_critic1(lm_states), self.v_critic2(lm_states)