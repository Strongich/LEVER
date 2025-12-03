from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import time

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.cumulative_reward = 0
        self.total_time = 0
        self.elapsed_time = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Access the reward from the parent class
        reward = self.locals["rewards"][0]
        
        # Update the cumulative reward
        self.cumulative_reward += reward
        
        elapsed_time = time.time() - self.start_time
        self.total_time += elapsed_time
        
        # Log the cumulative reward using wandb
        wandb.log({'cumulative_reward': self.cumulative_reward})
        
        return True


class Agent:
    def __init__(self, env, algorithm='A2C'):
        self.algorithm = algorithm
        
        if self.algorithm == 'DQN':
            self.model = DQN("MlpPolicy", env, verbose=1)
        elif self.algorithm == 'A2C':
            self.model = A2C("MlpPolicy", env, verbose=1)
        elif self.algorithm == 'PPO':
            self.model = PPO("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("Invalid algorithm. Choose either 'DQN' or 'A2C' or 'PPO'")

    def learn(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    
    def save(self, path):
        self.model.save(path)

    def load(self, path, grid_world):
        if self.algorithm == 'DQN':
            self.model = DQN.load(path, grid_world)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path, grid_world)
        elif self.algorithm == 'PPO':
            self.model = PPO.load(path, grid_world)
