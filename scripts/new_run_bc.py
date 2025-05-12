import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
import json
import os
import hydra
import random
from archer.data.utils import DummyDataset
from archer.algorithms.bc.core import plain_bc_loss
from archer.utils import colorful_print

CONFIG_NAME = "reinforce"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    from huggingface_hub import login
    login(token=config.huggingface_token)
    os.environ['TRANSFORMERS_CACHE'] = config.cache_dir

    # default to using cuda:0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # load GPT-2 model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # construct dataloader
    with open(config.data_path, "r") as fb:
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
    dataloader = DataLoader(DummyDataset(data[:]), batch_size=config.batch_size)

    # Training loop
    print_step = 0
    grad_step = 0
    for i in range(config.iterations):
        for batch in tqdm(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss = plain_bc_loss(model, tokenizer, **batch)
            grad_step += 1
            loss.backward()
            
            if grad_step % config.grad_accum_steps == 0:
                print_step += 1
                optimizer.step()
                optimizer.zero_grad()
                
                if print_step % 100 == 0:
                    print(f"Iteration {i}, Step {print_step}, Loss: {loss.item():.4f}")
                    if config.use_wandb:
                        wandb.log({
                            "loss": loss.item(),
                            "iteration": i,
                            "step": print_step
                        })
                    print_step = 0

    # Save checkpoint
    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, config.checkpoint_path)
    print(f"Saved checkpoint to {config.checkpoint_path}")

if __name__ == "__main__":
    main() 