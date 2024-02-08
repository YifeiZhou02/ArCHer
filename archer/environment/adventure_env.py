import jericho
import random
import concurrent.futures
from tqdm import tqdm

class AdventureEnv():
    def __init__(self, env_load_path, max_steps = 60,):
        self.env = jericho.FrotzEnv(env_load_path)
        self.max_time = max_steps
        self.steps = 0
        self.done = False
        self.reset_obs = None
        self.reset_state = None

    def get_surroudings(self):
        bkp = self.env.get_state()
        obs, _, _, _ = self.env.step('look')
        self.env.set_state(bkp)
        return obs
    
    def get_inventory(self):
        bkp = self.env.get_state()
        obs, _, _, _ = self.env.step('inventory')
        self.env.set_state(bkp)
        return obs
    
    def get_observation(self, obs):
        surrounding = self.get_surroudings()
        inventory = self.get_inventory()
        available_actions = str(self.env.get_valid_actions())
        return surrounding + inventory + obs + available_actions
    
    def reset(self, reset_state = None, reset_obs = None):
        if reset_state is not None and reset_obs is not None:
            self.reset_state = reset_state
            self.reset_obs = reset_obs
            self.env.set_state(self.reset_state)
        if self.reset_obs is None:
            obs, _ = self.env.reset()
            self.reset_obs = obs
            self.reset_state = self.env.get_state()
        
        self.steps = 0
        self.done = False
        return self.get_observation(self.reset_obs)
    
    def step(self, action):
        if self.done:
            return None
        action = action.split("\n")[0]
        self.steps += 1
        obs, reward, done, _ = self.env.step(action)
        done = done or self.steps >= self.max_time
        self.done = done
        return self.get_observation(obs), reward, done
    
    def random_action(self):
        return random.choice(self.env.get_valid_actions())
    
class BatchedAdventureEnv():
    def __init__(self, env_load_path, bsize = 32, max_steps = 60):
        #there is a resetting issue, so cannot set bsize too large
        bsize = 4
        self.env_list = [AdventureEnv(env_load_path, max_steps) for _ in range(bsize)]
        self.bsize = bsize
        self.reset_obs = None
        self.reset_state = None

    def reset(self, idx =None):
        results = []
        # print("resetting")
        if self.reset_obs is None:
            self.env_list[0].reset()
            self.reset_obs = self.env_list[0].reset_obs
            self.reset_state = self.env_list[0].reset_state
        # import IPython; IPython.embed()
        for env in tqdm(self.env_list, disable=True):
            results.append(env.reset(reset_state=self.reset_state, reset_obs=self.reset_obs))
        return results
        # with concurrent.futures.ThreadPoolExecutor() as executor: 
        #     jobs = [executor.submit(env.reset) for env in self.env_list]
        #     results = [job.result() for job in jobs]
        # return results
    
    def step(self, action_list):
        # print("stepping")
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            jobs = [executor.submit(env.step, action) for env, action in zip(self.env_list, action_list)]
            results = [job.result() for job in jobs]
        return results
