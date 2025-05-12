import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Thêm đường dẫn này vào sys.path nếu nó chưa có
if project_root not in sys.path:
    sys.path.append(project_root)



import torch
import transformers
from tqdm import tqdm
from archer.environment import BatchedTwentyQuestionsEnv, batch_interact_environment
from archer.models import ArcherAgent
from archer.prompts import MISTRAL_TWENTY_QUESTIONS_TEMPLATE, mistral_twenty_questions_decode_actions
from archer.utils import colorful_print
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
import numpy as np
from pathlib import Path
import json

# (Optional) WandB
import wandb

transformers.logging.set_verbosity_error()

CONFIG_NAME = "archer_20q"

@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: " + CONFIG_NAME + " <<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # === Thêm các cấu hình mặc định ngay trong mã nguồn ===
    use_wandb = True
    wandb_project = "llm_rl_20qsubset"
    log_dir = "logs"

    # WandB init nếu cần
    if use_wandb:
        wandb.init(
    project=wandb_project,
    config={
        "temperature": config.temperature,
        "do_sample": config.do_sample,
        "policy_lm": config.policy_lm,
        "critic_lm": config.critic_lm,
        "max_new_tokens": config.max_new_tokens,
        "eval_size": config.eval_size,
        "agent_type": config.agent_type,
        "env_name": config.env_name,
    }
)


    # Load environment
    if config.env_name == "twenty_questions":
        env = BatchedTwentyQuestionsEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir)
        eval_env = env
    else:
        raise NotImplementedError("Environment not implemented.")
    
    decode_f = lambda x: x
    
    if config.agent_type.lower() == "archer":
        print(">>> Using ArCHer agent")
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            eos_str='\n')
    else:
        raise NotImplementedError("Agent not implemented.")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if use_wandb:
        wandb.log({
            "model/num_parameters": count_parameters(agent.model),
            "critic/num_parameters": count_parameters(agent.critic)
        })

    
    tokenizer = agent.tokenizer
    
    if config.checkpoint_path is not None:
        try:
            state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
            model_state_dict = agent.model.state_dict()

            for key in state_dict.keys():
                if key in model_state_dict:
                    print(f"Checkpoint key: {key}, Checkpoint shape: {state_dict[key].shape}, Model shape: {model_state_dict[key].shape}")
                else:
                    print(f"Checkpoint key: {key} not found in model state dict")

            agent.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Model architecture or configuration mismatch. Please check the model configuration.")
            return
    
    print(">>> Evaluating the model")
    print(f"Traject: {config.eval_size}")
    eval_trajectories = batch_interact_environment(agent=agent,
                                                   tokenizer=tokenizer,
                                                   env=eval_env,
                                                   num_trajectories=config.eval_size*10,
                                                   env_idx=None,
                                                   use_tqdm=True,
                                                   decode_f=decode_f)
    
    # Đánh giá reward
    eval_rewards = [episode[-1]["trajectory_reward"] for episode in eval_trajectories]
    episode_lengths = [len(episode) for episode in eval_trajectories]

    mean_reward = np.mean(eval_rewards)
    variance_reward = np.var(eval_rewards)
    std_reward = np.std(eval_rewards)
    max_reward = np.max(eval_rewards)
    min_reward = np.min(eval_rewards)
    mean_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)

    print(f"Evaluation results:")
    print(f"- Mean Reward: {mean_reward:.4f}")
    print(f"- Variance: {variance_reward:.4f}")
    print(f"- Standard Deviation: {std_reward:.4f}")
    print(f"- Max Reward: {max_reward:.4f}")
    print(f"- Min Reward: {min_reward:.4f}")
    print(f"- Mean Episode Length: {mean_episode_length:.2f}")
    print(f"- Std Episode Length: {std_episode_length:.2f}")

    # >>> Đánh giá Critic
    print("\n>>> Đánh giá hiệu quả Critic (Q-value so với Reward thực tế)")

    critic = agent.critic
    num_samples = min(20, len(eval_trajectories))
    q_errors = []

    # Tạo thư mục log
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    critic_log_path = log_dir_path / "critic_eval_log.txt"
    summary_log_path = log_dir_path / "critic_summary.json"

    with open(critic_log_path, "w", encoding="utf-8") as f_log:
        for traj in eval_trajectories[:num_samples]:
            if not traj:
                continue
            for step in traj:
                obs = step.get("observation")
                act = step.get("action")
                r = step.get("reward")
                if obs is None or act is None or r is None:
                    continue
                q1, q2, _, _ = critic([obs], [act], detach_model=True)
                q_avg = ((q1 + q2) / 2.0).squeeze().item()
                q_errors.append(abs(q_avg - r))
                f_log.write(f"[Critic] Obs: {obs[:50]}..., Act: {act}, Q: {q_avg:.4f}, Reward: {r:.4f}, |Q - R| = {abs(q_avg - r):.4f}\n")

    if q_errors:
        summary = {
            "mean_abs_error": float(np.mean(q_errors)),
            "std_abs_error": float(np.std(q_errors)),
            "max_abs_error": float(np.max(q_errors)),
            "num_eval_steps": len(q_errors)
        }

        with open(summary_log_path, "w", encoding="utf-8") as f_summary:
            json.dump(summary, f_summary, indent=4)

        print(f">>> Critic evaluation saved to {critic_log_path}")
        print(f">>> Summary saved to {summary_log_path}")

        if use_wandb:
            wandb.log({
                "critic/mean_abs_error": summary["mean_abs_error"],
                "critic/std_abs_error": summary["std_abs_error"],
                "critic/max_abs_error": summary["max_abs_error"],
                "critic/num_eval_steps": summary["num_eval_steps"]
            })


    
if __name__ == "__main__":
    main()
