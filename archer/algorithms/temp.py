from archer.environment import batch_interact_environment
from archer.data import ReplayBuffer
import numpy as np
import os
import torch
import time
from tqdm import tqdm

def test_model_loop(env, agent, tokenizer, num_questions=5, env_idx=None, decode_f=lambda x: x):
    """
    Chạy thử mô hình với một số câu hỏi thay vì huấn luyện.
    """
    agent.prepare()
    print(">>> Bắt đầu chạy thử mô hình")
    
    # Thực hiện một số lượt tương tác với môi trường
    for i in tqdm(range(num_questions)):
        print(f"\n>>> Lượt tương tác {i+1}/{num_questions}")
        
        batch_obs = env.reset(idx=env_idx)
        print("Giá trị khởi tạo của batch_obs:", batch_obs)
        
        trajectories = batch_interact_environment(
            agent=agent,
            tokenizer=tokenizer,
            env=env,
            num_trajectories=1,  # Chỉ chạy thử một câu hỏi mỗi lượt
            env_idx=env_idx,
            use_tqdm=False,
            decode_f=decode_f
        )
        
        print("Hành động của agent:", agent.get_action(batch_obs))
        
        if not trajectories or not any(trajectories):
            print("Cảnh báo: trajectories rỗng! Kiểm tra lại quá trình tương tác với môi trường.")
            continue
        
        for traj in trajectories:
            if not traj:
                print("Cảnh báo: Một trajectory rỗng!")
                continue
            
            for step in traj:
                if 'query' in step and 'response' in step and 'reward' in step:
                    print(f"Câu hỏi: {step['query']}")
                    print(f"Phản hồi: {step['response']}")
                    print(f"Điểm thưởng: {step['reward']}")
                else:
                    print("Cảnh báo: Bước trong trajectory không chứa đủ thông tin!")
        
        time.sleep(2)  # Tạo độ trễ nhỏ để quan sát kết quả rõ hơn

    print("\n>>> Hoàn thành chạy thử mô hình!")
