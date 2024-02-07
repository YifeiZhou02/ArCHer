import torch
import transformers
from tqdm import tqdm
from LLM_rep_RL.environment import ContextualTextNavEnv
from LLM_rep_RL.models import DecisionModel
from LLM_rep_RL.data import DummyDataset
from LLM_rep_RL.algorithms.bc import train_loop
from LLM_rep_RL.utils import colorful_print
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
import json
import os
import hydra
import random

CONFIG_NAME = "reinforce"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    from huggingface_hub import login
    login(token=config.huggingface_token)
    os.environ['TRANSFORMERS_CACHE'] = config.cache_dir

    # default to using cuda:0 and cuda:1
    device = 'cuda:0'
    # load reference llama model and llama tokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    

    # construct dataloader
    with open("/nfs/kun2/users/yifei/LLM_rep_RL/LLM_rep_RL/data/twenty_questions.json", "r") as fb:
        raw_data = json.load(fb)
    data = []
    for d in raw_data:
        lines = d["lines"]
        len_lines = len(lines)
        for i in range(len_lines):
            obs = "Questions:\n" + "\n".join(lines[:i])
            action = lines[i].split("?")[0]+'?'
            data.append({'observation': obs, 'action': action+'\n'})
    random.shuffle(data)
    dataloader = DataLoader(DummyDataset(data[:]), batch_size = 8)

    train_loop(model = model,\
                tokenizer=tokenizer,\
                bc_dataloader = dataloader,\
                optimizer = optimizer,\
                iterations  = 1,\
                grad_accum_steps = 4)


if __name__ == "__main__":
    main()
