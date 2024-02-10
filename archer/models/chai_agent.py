import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from archer.models.critic import DoubleCritic
class CHAIAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2", critic_lm="roberta-base",
                cache_dir = '~/.cache', max_new_tokens=32,
                do_sample = True, temperature = 1.0, eos_str = '\n'):
        super(CHAIAgent, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(policy_lm, cache_dir = cache_dir).to(device)
        self.critic = DoubleCritic(device, accelerator=accelerator, 
                                    critic_lm=critic_lm,cache_dir = cache_dir, 
                                    in_dim = 768, out_dim = 1)  
        self.target_critic = DoubleCritic(device, accelerator=accelerator, 
                                    critic_lm=critic_lm,cache_dir = cache_dir, 
                                    in_dim = 768, out_dim = 1) 
        self.soft_update_target_critic(1)
        self.tokenizer =AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str

    def prepare(self):
        # self.model = self.accelerator.prepare(self.model)
        # self.critic.prepare()
        # self.target_critic.prepare()
        self.model, self.critic, self.target_critic = self.accelerator.prepare(self.model, self.critic, self.target_critic)


    def _get_action(self, observation):
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding= True, max_length=512, truncation = True).to(self.device)
        # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"])
        context_len = obs_ids['attention_mask'].size(1)
        # print(inputs_embeds.shape)
        # try:
        outputs = self.accelerator.unwrap_model(self.model).generate(**obs_ids,\
                                    max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,\
                                    pad_token_id = self.tokenizer.eos_token_id).cpu()
        outputs = outputs[:, context_len:]
        # except:
        #     import IPython; IPython.embed()

        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        #remove begining \n
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        return [raw_a.split(self.eos_str)[0] + self.eos_str for raw_a in raw_action]

    def get_action(self, observation):
        batch_actions = []
        batch_qs = []
        for _ in range(5):
            actions = self._get_action(observation)
            q1, q2, _, _ = self.target_critic(observation, actions, detach_model = True)
            qs = torch.minimum(q1, q2)
            batch_actions.append(actions)
            batch_qs.append(qs.reshape(1, -1))
        # import IPython; IPython.embed()
        batch_qs = torch.cat(batch_qs, dim = 0)
        selected_ids = torch.argmax(batch_qs, dim = 0).cpu().numpy()
        selected_actions = []
        for i, idx in enumerate(selected_ids):
            selected_actions.append(batch_actions[idx][i])
        return selected_actions

    # def get_q(self, observation, action, detach_model=False):
    #     return self.critic.get_q(observation, action, detach_model = detach_model)

    # def get_v(self, inputs, detach_model=False):
    #     return self.critic.get_v(inputs, detach_model = detach_model)
    
    # def get_target_q(self, observation, action, detach_model=False):
    #     return self.target_critic.get_q(observation, action, detach_model = detach_model)

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
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)
        return logsum_probs
    
    def soft_update_target_critic(self, tau):
        # for target_critic, critic in zip(self.target_critics, self.critics):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
