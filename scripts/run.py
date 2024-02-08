import torch
import transformers
from tqdm import tqdm
from archer.environment import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv,\
    BatchedAdventureEnv, BatchedGuessMyCityEnv, BatchedWebShopEnv
from archer.models import ArcherAgent, CHAIAgent
from archer.algorithms import offpolicy_train_loop
from archer.utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator

from accelerate import DistributedDataParallelKwargs
transformers.logging.set_verbosity_error()

CONFIG_NAME = "dqn_20q"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    from huggingface_hub import login
    login(token=config.huggingface_token)

    accelerator = Accelerator()
    device = accelerator.device

    # load environment
    if config.env_name == "twenty_questions":
        env = BatchedTwentyQuestionsEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir)
        eval_env = env
    elif config.env_name == "adventure":
        env = BatchedAdventureEnv(env_load_path = config.env_load_path,
                                    max_steps=50)
        eval_env = env
    elif config.env_name == "guess_my_city":
        env = BatchedGuessMyCityEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir)
        eval_env = env
    elif config.env_name == "webshop":
        env = BatchedWebShopEnv(lower=config.webshop_lower,
                                upper=config.webshop_upper,
                                env_load_path=config.env_load_path)
        eval_env = env
    else:
        raise NotImplementedError("Environment not implemented.")
    
    # load decision model
    if config.agent_type.lower() == "chai":
        print(">>> Using CHAI agent")
        agent = CHAIAgent(device=device, accelerator=accelerator, 
                        temperature=config.temperature, 
                        do_sample=config.do_sample, policy_lm=config.policy_lm, 
                        critic_lm=config.critic_lm, cache_dir=config.cache_dir,
                        max_new_tokens=config.max_new_tokens)
        #if use chai, do not update the actor
        config.warmup_iter = config.iterations
    elif config.agent_type.lower() == "archer":
        print(">>> Using CHAI agent")
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens)
    else:
        raise NotImplementedError("Agent not implemented.")
    state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
    tokenizer = agent.tokenizer
    # new_state_dict = {}
    # for k,v in state_dict.items():
    #     new_state_dict[k[6:]] = v
    # print(torch.load(config.checkpoint_path)['model_state_dict'])
    agent.model.load_state_dict(state_dict)
    # agent = accelerator.prepare(agent)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train_loop(env = env,\
                agent = agent,\
                tokenizer = tokenizer,\
                eval_env = eval_env,\
                accelerator = accelerator,\
                **config)


if __name__ == "__main__":
    main()
