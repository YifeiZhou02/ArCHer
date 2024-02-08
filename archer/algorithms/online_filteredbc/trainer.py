import torch
import transformers
from tqdm import tqdm
from archer.algorithms.bc import plain_bc_loss
import copy
import random
from torch.utils.data import DataLoader
from archer.data import DummyDataset
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
class BCTrainer():
    def __init__(self, agent,\
                    tokenizer,\
                    accelerator,\
                    lm_lr: float = 1e-5,\
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    grad_accum_steps: int = 8):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.step = 0
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.agent, self.lm_optimizer = self.accelerator.prepare(self.agent, self.lm_optimizer)

    def actor_loss(self, observation, action, **kwargs):
        loss = plain_bc_loss(self.accelerator.unwrap_model(self.agent).model, self.tokenizer, observation, action)
        self.accelerator.backward(loss)
        return {"bc.loss": loss.detach().cpu().item()}


    def update(self, replay_buffer, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        #update actor
        if  not no_update_actor:
            action_bsize = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer.batch_size
            for _ in range(self.epochs):
                self.lm_optimizer.zero_grad()
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                grad_index = 0
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                dataloader = self.accelerator.prepare(dataloader)
                for batch in dataloader:
                    info_list.append(self.actor_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        return info

    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                    'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
                    'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent
