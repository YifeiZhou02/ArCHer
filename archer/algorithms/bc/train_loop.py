from archer.data import DummyDataset
from archer.algorithms.bc import plain_bc_loss
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import torch
def train_loop(model,\
                bc_dataloader,\
                tokenizer,\
                iterations: int = 10,\
                grad_accum_steps: int = 1,\
                **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    print_step = 0
    grad_step = 0
    for i in range(iterations):
            for batch in tqdm(bc_dataloader):
                loss = plain_bc_loss(model, tokenizer, **batch)
                grad_step += 1
                loss.backward()
                if grad_step % grad_accum_steps == 0:
                    print_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    if print_step % 100 == 0:
                         print(loss.item())
                         print_step = 0
    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'../twenty_questions_gpt2_model0_full.pt')
    # return model