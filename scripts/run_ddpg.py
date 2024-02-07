import torch
import transformers
from tqdm import tqdm
from LLM_rep_RL.environment import ContextualTextNavEnv
from LLM_rep_RL.models import DecisionAgent
from LLM_rep_RL.algorithms.ddpg import train_loop
from LLM_rep_RL.utils import colorful_print
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra

CONFIG_NAME = "ddpg"
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
    llama_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True).to(device)
    llama_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # load environment
    env = ContextualTextNavEnv()
    env.load(base_path = config.env_load_path, offset=config.env_offset, cutoff=config.env_cutoff)

    # load decision model
    agent = DecisionAgent(device="cuda:1")
    state_dict = torch.load(config.checkpoint_path)['model_state_dict']
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k[6:]] = v
    # print(torch.load(config.checkpoint_path)['model_state_dict'])
    agent.model.load_state_dict(new_state_dict)

    if config.use_wandb:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    train_loop(env = env,\
                agent = agent,\
                llama_model = llama_model,\
                llama_tokenizer = llama_tokenizer,\
                **config)


if __name__ == "__main__":
    main()
