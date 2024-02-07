from archer.environment import plain_interact_environment, \
    batch_plain_interact_environment, BatchedTwentyQuestionsEnv
from archer.data import DummyDataset, get_bc_dataloader, ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .trainer import BCTrainer
import wandb
def filteredbc_train_loop(env,\
                eval_env,\
                agent,\
                tokenizer,\
                accelerator,
                post_f = lambda x:x,\
                bc_data_cutoff: int = 50,
                bc_num_trajectories: int = 1000,
                beta: float = 1.0,\
                warmup_size: int = 200,
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                env_idx:int = None,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                lm_lr: float = 1e-5,\
                use_wandb: bool = False,
                env_load_path: str = '../LLM_rep_RL/simple_game/',
                max_grad_norm: float = 0.01,
                top_p: float = 0.1,
                **kwargs):
    trainer = BCTrainer(agent=agent,\
                            tokenizer=tokenizer,\
                            accelerator=accelerator,
                            lm_lr = lm_lr,\
                            epochs = epochs,\
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    all_trajectories = []
    # bc_dataloader = get_bc_dataloader(bc_data_cutoff, bc_num_trajectories, env_base_path=env_load_path)
    bc_dataloader = None
    # warmup rollout
    print(">>>collecting warmup rollouts")
    trajectories = batch_plain_interact_environment(agent = agent,\
                                            tokenizer = tokenizer,
                                            env = env,\
                                            num_trajectories= warmup_size,\
                                            post_f = post_f,\
                                            env_idx = env_idx,
                                            temperature=temperature,
                                            do_sample=do_sample)
    all_trajectories += trajectories
    # data = list(filter(lambda x: x["reward"] > 0, data))
    # breakpoint()
    #main training loop
    for i in tqdm(range(iterations)):
        # print(">>>Interacting with Environment")
        trajectories = batch_plain_interact_environment(agent = agent,\
                                        tokenizer= tokenizer,\
                                        env = env,\
                                        num_trajectories= rollout_size,\
                                        post_f = post_f,\
                                        env_idx = env_idx,
                                        use_tqdm=False,
                                        temperature=temperature,
                                        do_sample=do_sample)
        # else:
        #     trajectories = plain_interact_environment(agent = agent,\
        #                                     tokenizer= tokenizer,\
        #                                     env = env,\
        #                                     num_trajectories= rollout_size,\
        #                                     post_f = post_f,\
        #                                     env_idx = env_idx,
        #                                     use_tqdm=False,
        #                                     temperature=temperature,
        #                                     do_sample=do_sample)
        old_sample = agent.do_sample
        agent.sample = False
        #use 100 trajectories to eval
        #TODO: set to 1 for dummy eval
        eval_trajectories =  batch_plain_interact_environment(agent = agent,\
                                            tokenizer= tokenizer,\
                                            env = eval_env,\
                                            num_trajectories=  32,\
                                            post_f = post_f,\
                                            env_idx = env_idx,
                                            use_tqdm=False,
                                            temperature=temperature,
                                            do_sample=do_sample)
        # import IPython; IPython.embed()
        all_trajectories += trajectories
        replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
        episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
        cutoff = np.quantile(episode_rewards, 1 - top_p)
        print("Episode Reward Cutoff: ", cutoff)
        filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
        data = sum(filtered_trajectories, [])
        assert len(data) > 0
        for d in data:
            replay_buffer.insert(**d)
        agent.do_sample = old_sample
        # print("Trajectory Mean for Data Collection")
        # print(np.mean([d[0]["trajectory_reward"] for d in trajectories]))
        info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),\
                "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),\
                "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories]),
                "eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),}
        info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                "rollout.reward.max": np.max([d["reward"] for d in data]),\
                "rollout.reward.min": np.min([d["reward"] for d in data])})
        # data = list(filter(lambda x: x["reward"] >0, data))
        print("Training")
        info.update(trainer.update(replay_buffer, bc_dataloader, no_update_actor = (i < warmup_iter)))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
    # return model