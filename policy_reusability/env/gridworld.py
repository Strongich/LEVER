import gym
from gym import spaces
import numpy as np
import copy

from env.Random_Policies_Generation import generate_random_policies

class GridWorld(gym.Env):
    
    def __init__(self, grid_width, grid_length, reward_system, agent_position, target_position, cell_low_value, cell_high_value, 
        start_position_value, target_position_value, block_position_value, agent_position_value, gold_position_value, block_reward, target_reward, gold_k=0, gold_positions=None, block_positions=None, n = 0, action_size=2, parameterized = False, alpha_beta = (1, 1)):
        
        super(GridWorld, self).__init__()

        self.grid_width = grid_width
        self.grid_length = grid_length
        self.agent_position = agent_position # e.g., [0, 0]
        self.start_position = agent_position # e.g., [0, 0]
        self.target_position = target_position # e.g., [4, 4]
        self.start_position_value = start_position_value
        self.target_position_value = target_position_value
        self.agent_position_value = agent_position_value
        self.reward_system = reward_system
        self.gold_positions = gold_positions
        self.block_positions = block_positions
        self.block_reward = block_reward
        self.target_reward = target_reward
        self.block_position_value = block_position_value
        self.gold_position_value = gold_position_value
        self.gold_k = gold_k
        self.parameterized = parameterized
        self.num_synthetic_policies = n
        self.alpha_beta = alpha_beta
        self.reward_dict = generate_random_policies(self.grid_width, self.grid_length, self.num_synthetic_policies, 0, 1)

        # action space in case we want to avoid cycles
        self.action_space = spaces.Discrete(action_size)
        self.action_count = action_size
        #self.action_space = spaces.Discrete(4)
        self.state_count = self.grid_length * self.grid_width
        self.observation_space = spaces.Box(low = cell_low_value,
            high= cell_high_value, shape=(self.grid_width, self.grid_length))

        # Initialize the grid
        self.grid = np.zeros((self.grid_width, self.grid_length))
        self.visited_count_states = np.zeros((self.grid_width, self.grid_length))
        self.visited_count_transitions = np.zeros((self.grid_width, self.grid_length, self.action_count))
        self.grid = np.zeros((self.grid_width, self.grid_length), dtype=np.int8)
        self.grid[self.start_position[0]][self.start_position[1]] = start_position_value # e.g., 5
        self.grid[self.target_position[0]][self.target_position[1]] = target_position_value # e.g., 10

        # Position the golds
        if gold_positions is not None:
            for pos in gold_positions:
                self.grid[pos[0]][pos[1]] = 1

        # Position the blocks
        if block_positions is not None:
            for pos in block_positions:
                self.grid[pos[0]][pos[1]] = -1

        # position the agent
        self.grid[self.start_position[0]][self.start_position[1]] = self.agent_position_value

    def reset(self, new_start_position=None):
       #self.visited_count_states = np.zeros((self.grid_width, self.grid_length))
        #self.visited_count_transitions = np.zeros((self.grid_width, self.grid_length, self.action_count))
        self.grid[self.agent_position[0]][self.agent_position[1]] = 0
        if new_start_position != None:
            self.start_position = new_start_position
        self.agent_position = copy.copy(self.start_position) # e.g., [0, 0]
        self.grid[self.target_position[0]][self.target_position[1]] = self.target_position_value
        for gold in self.gold_positions:
            self.grid[gold[0]][gold[1]] = 1
        for block in self.block_positions:
            self.grid[block[0]][block[1]] = -1
        self.grid[self.start_position[0]][self.start_position[1]] = self.agent_position_value
        return self.grid
    
    # this function is just to convert a position on the grid to an index
    def state_to_index(self, state):
        next_state_index = np.ravel_multi_index(tuple(state), dims=self.grid.shape)
        return next_state_index

    # this function is only used for grid world environment
    # and is to convert a state index to its position on the grid
    @staticmethod
    def index_to_state(index, grid_length):
        result = int(index / grid_length), int(index % grid_length)
        return result    
    
    def obtain_action(self, state_1, state_2):
        # down
        if (state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1]):
            return 1
        # right
        elif (state_2[0] == state_1[0] and state_2[1] == state_1[1] + 1):
            return 0
        # down*2
        if (state_2[0] == state_1[0] + 2 and state_2[1] == state_1[1]):
            return 3
        # right*2
        elif (state_2[0] == state_1[0] and state_2[1] == state_1[1] + 2):
            return 2
        #diagonal
        elif (state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1] + 1):
            return 4
        else:
            return None
            print("Action could not be obtained")
    
    def check_boundry_constraint(self):
        if (0 <= self.agent_position[0] < self.grid_width) and (0 <= self.agent_position[1] < self.grid_length):
            return True
        return False

    def step(self, action):
        prev_agent_position = [self.agent_position[0], self.agent_position[1]]
        self.visited_count_transitions[prev_agent_position[0], prev_agent_position[1], action] += 1
        #NOTE: actions in case we want to avoid cycle
        if action == 0: # right
            self.agent_position[1] += 1
        elif action == 1: # down
            self.agent_position[0] += 1
        elif action == 2: # right*2
            self.agent_position[1] += 2
        elif action == 3: #down*2
            self.agent_position[0] += 2
        elif action == 4: #diagonal
            self.agent_position[0] += 1
            self.agent_position[1] += 1
        else:
            print(f"Action {action} not defined!")

        
        # check boundary constraint of the grid world
        if not self.check_boundry_constraint():
            self.agent_position = prev_agent_position
            #return self.grid, 0, False, {False}

        reward = self._get_reward(prev_agent_position)

        # update observation space
        self.grid[prev_agent_position[0]][prev_agent_position[1]] = 0
        self.grid[self.agent_position[0]][self.agent_position[1]] = self.agent_position_value

        done = np.array_equal(self.agent_position, self.target_position)

        return self.grid, reward, done, {}
    
    def _get_reward(self, prev_agent_position):

        # gold collection task
        if self.reward_system == "gold":
            return self.get_reward_gold()

        # shortest path task
        elif self.reward_system == "path":
            return self.get_reward_path(prev_agent_position)
        
        elif self.reward_system == "combined":
            if not self.parameterized:
                return self.get_reward_gold() + self.get_reward_path(prev_agent_position)
            else:
                alpha, beta = self.alpha_beta[0], self.alpha_beta[1]
                return  alpha * self.get_reward_gold() + beta * self.get_reward_path(prev_agent_position)
        # get synthetic policies reward

        action = self.obtain_action(prev_agent_position, self.agent_position)
        total = 0

        for i in range(self.num_synthetic_policies):
            if self.reward_system == f"R{i}":
                return self.get_reward_synthetic(prev_agent_position, i, action)
            elif self.reward_system == "combined_synthetic":
                total += self.get_reward_synthetic(prev_agent_position, i, action)
        return total

        return 0

    def render(self, mode='human'):
        print(self.grid)

    def get_reward_synthetic(self, prev_agent_position, i, action):

        if action == None:
            return self.block_reward

        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:  # block
            return self.block_reward
        if current_cell_value == self.target_position_value: # target
            return self.target_reward

        return self.reward_dict[i][tuple(prev_agent_position)][action]

    def get_reward_path(self, prev_agent_position):
        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:  # block
            return self.block_reward
        if current_cell_value == self.target_position_value: # target
            return self.target_reward
        d1 = np.sum(np.abs(np.array(prev_agent_position) - np.array(self.target_position)))
        d2 = np.sum(np.abs(np.array(self.agent_position) - np.array(self.target_position)))
        r = d1 - d2
        return r    
        
    def get_reward_gold(self):
        reward = 0
        candidates = []
        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:  # block
            return self.block_reward
        if current_cell_value == self.target_position_value: # target
            return self.target_reward
        for i in range(-self.gold_k, self.gold_k + 1):
            for j in range(-self.gold_k, self.gold_k + 1):
                new_candidate = [self.agent_position[0] + i, self.agent_position[1] + j]
                new_candidate = np.clip(new_candidate, (0, 0), (self.grid_width - 1, self.grid_length - 1)).tolist()
                if new_candidate not in candidates:
                    candidates.append(new_candidate)
        for candidate in candidates:
            cell_value = self.grid[candidate[0], candidate[1]]
            if cell_value == self.gold_position_value:
                reward += 1
        if current_cell_value == self.gold_position_value:  # gold
            self.grid[self.agent_position[0]][self.agent_position[1]] = 0

        return reward
