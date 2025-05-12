import torch
from torch.optim import Adam
from accelerate import Accelerator
from tqdm import tqdm
from archer.models.policy_only_agent import PolicyOnlyAgent
from archer.environment import BatchedTwentyQuestionsEnv  # Đảm bảo đường dẫn đúng

# Cấu hình huấn luyện
class TrainingConfig:
    policy_lm = "gpt2"
    learning_rate = 1e-6
    gamma = 0.99
    num_episodes = 5000  
    log_interval = 100
    eval_interval = 500
    eval_episodes = 100
    max_conversation_length = 20
    batch_size = 8
    env_load_path = '/home/biggod/archer/archer_public_data/20q_t5_oracle.pt'
    cache_dir = "~/.cache"
    device = "cuda" if torch.cuda.is_available() else "cpu" # Có thể không cần nếu dùng accelerate
    use_wandb = True
    project_name = 'llm_rl_baselines'
    run_name = 'policy-only-test'

# Khởi tạo cấu hình và accelerator
config = TrainingConfig()
accelerator = Accelerator()
device = accelerator.device

# Khởi tạo WandB (nếu được cấu hình)
if config.use_wandb:
    import wandb
    wandb.init(project=config.project_name, name=config.run_name)
    wandb.config.update(config)

# Khởi tạo môi trường
env = BatchedTwentyQuestionsEnv(
    env_load_path=config.env_load_path,
    cache_dir=config.cache_dir,
    device=device,
    max_conversation_length=config.max_conversation_length,
    bsize=config.batch_size
)

# Khởi tạo agent
policy_agent = PolicyOnlyAgent(
    device=device,
    accelerator=accelerator,
    policy_lm=config.policy_lm,
    cache_dir=config.cache_dir,
    max_new_tokens=32 # Điều chỉnh nếu cần
)
policy_agent.prepare()

# Khởi tạo optimizer
policy_optimizer = Adam(policy_agent.model.parameters(), lr=config.learning_rate)

# Hàm đánh giá (tương tự như phần đánh giá agent ArCHer của bạn)
def evaluate_agent(agent, env, num_episodes):
    total_reward = 0
    total_success = 0
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        observations = env.reset()
        done = [False] * config.batch_size
        episode_reward = [0] * config.batch_size
        steps = 0
        while not all(done) and steps < config.max_conversation_length:
            actions = agent.get_action(observations)
            results = env.step(actions)
            for i, (next_obs, reward, d) in enumerate(results):
                if not done[i]:
                    observations[i] = next_obs
                    episode_reward[i] += reward
                    done[i] = d
                    if d and reward == 0:
                        total_success += 1
            steps += 1
        total_reward += sum(episode_reward)
    avg_reward = total_reward / (num_episodes * config.batch_size)
    success_rate = total_success / (num_episodes * config.batch_size)
    return avg_reward, success_rate

# Vòng lặp huấn luyện
for episode in range(config.num_episodes):
    observations = env.reset()
    episode_rewards = [[] for _ in range(config.batch_size)]
    episode_log_probs = [[] for _ in range(config.batch_size)]
    done = [False] * config.batch_size
    steps = 0

    while not all(done) and steps < config.max_conversation_length:
        actions = policy_agent.get_action(observations)
        log_probs = policy_agent.get_log_prob(observations, actions)
        results = env.step(actions)

        for i, (next_obs, reward, d) in enumerate(results):
            if not done[i]:
                episode_rewards[i].append(reward)
                episode_log_probs[i].append(log_probs[i])
                observations[i] = next_obs
                done[i] = d

        steps += 1

    # Tính toán discounted rewards và policy loss cho mỗi episode trong batch
    policy_loss = torch.tensor([0.0] * config.batch_size, device=device)
    for i in range(config.batch_size):
        rewards = episode_rewards[i]
        log_probs = episode_log_probs[i]
        discounted_rewards = [0] * len(rewards)
        cumulative_reward = 0
        for j in reversed(range(len(rewards))):
            cumulative_reward = rewards[j] + config.gamma * cumulative_reward
            discounted_rewards[j] = cumulative_reward

        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        log_probs_tensor = torch.stack(log_probs)
        policy_loss[i] = (-log_probs_tensor * discounted_rewards_tensor).mean()

    mean_policy_loss = policy_loss.mean()

    # Cập nhật policy
    policy_optimizer.zero_grad()
    accelerator.backward(mean_policy_loss)
    policy_optimizer.step()

    # Ghi log
    if (episode + 1) % config.log_interval == 0:
        print(f"Episode {episode + 1}, Policy Loss: {mean_policy_loss.item():.4f}")
        if config.use_wandb:
            wandb.log({"policy_loss": mean_policy_loss.item()}, step=episode + 1)

    # Đánh giá
    if (episode + 1) % config.eval_interval == 0:
        avg_reward, success_rate = evaluate_agent(policy_agent.accelerator.unwrap_model(policy_agent), env, config.eval_episodes)
        print(f"Evaluation at Episode {episode + 1}: Avg Reward = {avg_reward:.4f}, Success Rate = {success_rate:.4f}")
        if config.use_wandb:
            wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate}, step=episode + 1)

if config.use_wandb:
    wandb.finish()
print("Huấn luyện hoàn tất!")