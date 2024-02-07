import torch
import transformers
from tqdm import tqdm
from LLM_rep_RL.environment import ContextualTextNavEnv, \
    TwentyQuestionsEnv, BatchedTwentyQuestionsEnv,\
    BatchedAdventureEnv, BatchedGuessMyCityEnv
from LLM_rep_RL.models import DQNAgent
from LLM_rep_RL.algorithms.online_filteredbc import filteredbc_train_loop
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
from accelerate import Accelerator
transformers.logging.set_verbosity_error()

CONFIG_NAME = "filteredbc_spellbrkr"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    from huggingface_hub import login
    login(token=config.huggingface_token)
    os.environ['TRANSFORMERS_CACHE'] = config.cache_dir

    # default to using cuda:0 and cuda:1
    accelerator = Accelerator()
    device = accelerator.device
    # load reference llama model and llama tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # load environment
    if config.env_name == "twenty_questions":
        env = BatchedTwentyQuestionsEnv()
    elif config.env_name == "adventure":
        env = BatchedAdventureEnv()
    elif config.env_name == "guess_my_city":
        env = BatchedGuessMyCityEnv()
    else:
        env = ContextualTextNavEnv()
        env.load(base_path = config.env_load_path, offset=config.env_offset, cutoff=config.env_cutoff)
    eval_env = env
    # load decision model
    agent = DQNAgent(device=device, accelerator=accelerator, temperature=config.temperature, do_sample=config.do_sample, policy_lm=config.policy_lm, cache_dir=config.cache_dir)
    state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
    tokenizer = agent.tokenizer
    # new_state_dict = {}
    # for k,v in state_dict.items():
    #     new_state_dict[k[6:]] = v
    # print(torch.load(config.checkpoint_path)['model_state_dict'])
    agent.model.load_state_dict(state_dict)
    agent.prepare()
    

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    filteredbc_train_loop(env = env,\
                agent = agent,\
                tokenizer = tokenizer,\
                accelerator=accelerator,
                eval_env=eval_env,\
                **config)


if __name__ == "__main__":
    main()
