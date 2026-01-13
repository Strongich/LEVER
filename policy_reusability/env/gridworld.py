import copy

import gym
import numpy as np
from gym import spaces

from policy_reusability.env.Random_Policies_Generation import generate_random_policies


class GridWorld(gym.Env):
    def __init__(
        self,
        grid_width,
        grid_length,
        reward_system,
        agent_position,
        target_position,
        cell_low_value,
        cell_high_value,
        start_position_value,
        target_position_value,
        block_position_value,
        agent_position_value,
        gold_position_value,
        block_reward,
        target_reward,
        gold_k=0,
        gold_positions=None,
        block_positions=None,
        hazard_positions=None,
        lever_position=None,
        n=0,
        action_size=4,
        parameterized=False,
        alpha_beta=(1, 1),
        step_penalty=0.0,
        hazard_position_value=-2,
        lever_position_value=2,
        hazard_penalty=0.0,
        exit_without_lever_penalty=0.0,
        legacy_gold_exit_penalty: bool = False,
    ):
        super(GridWorld, self).__init__()

        self.grid_width = grid_width
        self.grid_length = grid_length
        self.agent_position = agent_position  # e.g., [0, 0]
        self.start_position = agent_position  # e.g., [0, 0]
        self.target_position = target_position  # e.g., [4, 4]
        self.start_position_value = start_position_value
        self.target_position_value = target_position_value
        self.agent_position_value = agent_position_value
        self.reward_system = reward_system
        self.gold_positions = gold_positions
        self.block_positions = block_positions
        self.hazard_positions = hazard_positions or []
        self.lever_position = lever_position
        self.total_gold_count = len(gold_positions) if gold_positions else 0
        self.gold_remaining = self.total_gold_count
        self.lever_collected = False
        self.hazard_steps = 0
        self.block_reward = block_reward
        self.target_reward = target_reward
        self.step_penalty = step_penalty  # flat per-step cost to discourage long paths
        self.block_position_value = block_position_value
        self.gold_position_value = gold_position_value
        self.hazard_position_value = hazard_position_value
        self.lever_position_value = lever_position_value
        self.hazard_penalty = hazard_penalty
        self.exit_without_lever_penalty = exit_without_lever_penalty
        self.legacy_gold_exit_penalty = legacy_gold_exit_penalty
        self.gold_k = gold_k
        self.parameterized = parameterized
        self.num_synthetic_policies = n
        self.alpha_beta = alpha_beta
        self.reward_dict = generate_random_policies(
            self.grid_width, self.grid_length, self.num_synthetic_policies, 0, 1
        )

        # action space in case we want to avoid cycles
        self.action_space = spaces.Discrete(action_size)
        self.action_count = action_size
        # self.action_space = spaces.Discrete(4)
        self.state_count = self.grid_length * self.grid_width
        self.observation_space = spaces.Box(
            low=cell_low_value,
            high=cell_high_value,
            shape=(self.grid_width, self.grid_length),
        )

        # Initialize the grid
        self.grid = np.zeros((self.grid_width, self.grid_length))
        self.visited_count_states = np.zeros((self.grid_width, self.grid_length))
        self.visited_count_transitions = np.zeros(
            (self.grid_width, self.grid_length, self.action_count)
        )
        self.grid = np.zeros((self.grid_width, self.grid_length), dtype=np.int8)
        self.grid[self.start_position[0]][self.start_position[1]] = (
            start_position_value  # e.g., 5
        )
        self.grid[self.target_position[0]][self.target_position[1]] = (
            target_position_value  # e.g., 10
        )

        # Position the golds
        if gold_positions is not None:
            for pos in gold_positions:
                self.grid[pos[0]][pos[1]] = 1

        # Position the blocks
        if block_positions is not None:
            for pos in block_positions:
                self.grid[pos[0]][pos[1]] = -1

        # Position hazards
        for pos in self.hazard_positions:
            self.grid[pos[0]][pos[1]] = self.hazard_position_value

        # Position the lever (single cell)
        if self.lever_position is not None:
            self.grid[self.lever_position[0]][self.lever_position[1]] = (
                self.lever_position_value
            )

        # position the agent
        self.grid[self.start_position[0]][self.start_position[1]] = (
            self.agent_position_value
        )

    def reset(self, new_start_position=None):
        # self.visited_count_states = np.zeros((self.grid_width, self.grid_length))
        # self.visited_count_transitions = np.zeros((self.grid_width, self.grid_length, self.action_count))
        self.grid[self.agent_position[0]][self.agent_position[1]] = 0
        if new_start_position != None:
            self.start_position = new_start_position
        self.agent_position = copy.copy(self.start_position)  # e.g., [0, 0]
        self.gold_remaining = self.total_gold_count
        self.lever_collected = False
        self.hazard_steps = 0
        self.grid[self.target_position[0]][self.target_position[1]] = (
            self.target_position_value
        )
        for gold in self.gold_positions:
            self.grid[gold[0]][gold[1]] = 1
        for block in self.block_positions:
            self.grid[block[0]][block[1]] = -1
        for hazard in self.hazard_positions:
            self.grid[hazard[0]][hazard[1]] = self.hazard_position_value
        if self.lever_position is not None:
            self.grid[self.lever_position[0]][self.lever_position[1]] = (
                self.lever_position_value
            )
        self.grid[self.start_position[0]][self.start_position[1]] = (
            self.agent_position_value
        )
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
        # right
        if state_2[0] == state_1[0] and state_2[1] == state_1[1] + 1:
            return 0
        # down
        elif state_2[0] == state_1[0] + 1 and state_2[1] == state_1[1]:
            return 1
        # right x2
        elif state_2[0] == state_1[0] and state_2[1] == state_1[1] + 2:
            return 2
        # down x2
        elif state_2[0] == state_1[0] + 2 and state_2[1] == state_1[1]:
            return 3
        else:
            return None
            print("Action could not be obtained")

    def check_boundry_constraint(self):
        if (0 <= self.agent_position[0] < self.grid_width) and (
            0 <= self.agent_position[1] < self.grid_length
        ):
            return True
        return False

    def step(self, action):
        prev_agent_position = [self.agent_position[0], self.agent_position[1]]
        self.visited_count_transitions[
            prev_agent_position[0], prev_agent_position[1], action
        ] += 1
        # NOTE: actions in case we want to avoid cycle
        if action == 0:  # right
            self.agent_position[1] += 1
        elif action == 1:  # down
            self.agent_position[0] += 1
        elif action == 2:  # right x2
            self.agent_position[1] += 2
        elif action == 3:  # down x2
            self.agent_position[0] += 2
        else:
            print(f"Action {action} not defined!")

        # check boundary constraint of the grid world
        if not self.check_boundry_constraint():
            self.agent_position = prev_agent_position
            return self.grid, -1.0, False, {}

        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:
            self.agent_position = prev_agent_position
            return self.grid, -1.0, False, {}
        if current_cell_value == self.hazard_position_value:
            self.hazard_steps += 1
            # update observation space
            self.grid[prev_agent_position[0]][prev_agent_position[1]] = 0
            self.grid[self.agent_position[0]][self.agent_position[1]] = (
                self.agent_position_value
            )
            return self.grid, -100.0, True, {}

        reward = self._get_reward(prev_agent_position)

        # update observation space
        self.grid[prev_agent_position[0]][prev_agent_position[1]] = 0
        self.grid[self.agent_position[0]][self.agent_position[1]] = (
            self.agent_position_value
        )

        done = np.array_equal(self.agent_position, self.target_position)

        return self.grid, reward, done, {}

    def _get_reward(self, prev_agent_position):
        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]

        rs = self.reward_system
        if rs.startswith("R") or "synthetic" in rs:
            action = self.obtain_action(prev_agent_position, self.agent_position)
            total = 0
            for i in range(self.num_synthetic_policies):
                if self.reward_system == f"R{i}":
                    return self.get_reward_synthetic(prev_agent_position, i, action)
                elif self.reward_system == "combined_synthetic":
                    total += self.get_reward_synthetic(prev_agent_position, i, action)
            return total

        if rs == "combined":
            components = {"path", "gold"}
        else:
            components = {token for token in rs.split("-") if token}

        use_path = "path" in components
        use_gold = "gold" in components
        use_hazard = "hazard" in components
        use_lever = "lever" in components

        # Block overrides all other rewards
        if current_cell_value == self.block_position_value:
            return self.block_reward

        reward_total = 0.0

        # Lever handling (one-time reward if applicable)
        if (
            self.lever_position is not None
            and current_cell_value == self.lever_position_value
            and not self.lever_collected
        ):
            self.lever_collected = True
            self.grid[self.agent_position[0]][self.agent_position[1]] = 0
            if use_lever:
                reward_total += 20

        # Gold reward (only for gold objective)
        if use_gold:
            reward_total += self.get_reward_gold(
                current_cell_value=current_cell_value,
                reward_per_gold=5,
            )

        # Hazard proximity penalty (4-neighborhood)
        if use_hazard:
            x, y = self.agent_position
            hazard_neighbors = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_length:
                    if self.grid[nx][ny] == self.hazard_position_value:
                        hazard_neighbors += 1
            reward_total -= 2.0 * hazard_neighbors

        # Exit reward
        if current_cell_value == self.target_position_value:
            if use_path:
                reward_total += 100
            if use_hazard:
                reward_total += 100
            if use_gold:
                if self.gold_remaining == 0:
                    reward_total += 100
                else:
                    if self.legacy_gold_exit_penalty:
                        reward_total -= 20
                    else:
                        total_gold = max(1, self.total_gold_count)
                        missing_frac = self.gold_remaining / total_gold
                        reward_total -= 10 * missing_frac
            if use_lever:
                reward_total += 80 if self.lever_collected else 20

        # Step costs (sum of active objectives)
        step_cost = 0.0
        if use_path:
            step_cost += 1.0
        if use_gold:
            step_cost += 0.01
        if use_hazard:
            step_cost += 0.1
        if use_lever:
            step_cost += 0.1

        reward_total -= step_cost
        return reward_total

    def render(self, mode="human"):
        print(self.grid)

    def get_reward_synthetic(self, prev_agent_position, i, action):
        if action == None:
            return self.block_reward

        current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:  # block
            return self.block_reward
        if current_cell_value == self.target_position_value:  # target
            return self.target_reward

        return self.reward_dict[i][tuple(prev_agent_position)][action]

    def get_reward_path(self, prev_agent_position):
        d1 = np.sum(
            np.abs(np.array(prev_agent_position) - np.array(self.target_position))
        )
        d2 = np.sum(
            np.abs(np.array(self.agent_position) - np.array(self.target_position))
        )
        r = d1 - d2
        return r

    def get_reward_gold(
        self,
        current_cell_value=None,
        reward_per_gold=1,
    ):
        if current_cell_value is None:
            current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        reward = 0
        candidates = []
        for i in range(-self.gold_k, self.gold_k + 1):
            for j in range(-self.gold_k, self.gold_k + 1):
                new_candidate = [self.agent_position[0] + i, self.agent_position[1] + j]
                new_candidate = np.clip(
                    new_candidate, (0, 0), (self.grid_width - 1, self.grid_length - 1)
                ).tolist()
                if new_candidate not in candidates:
                    candidates.append(new_candidate)
        for candidate in candidates:
            cell_value = self.grid[candidate[0], candidate[1]]
            if cell_value == self.gold_position_value:
                reward += reward_per_gold
        if current_cell_value == self.gold_position_value:  # gold
            self.grid[self.agent_position[0]][self.agent_position[1]] = 0
            if self.gold_remaining > 0:
                self.gold_remaining -= 1

        return reward

    def get_reward_lever_path(
        self,
        prev_agent_position,
        current_cell_value=None,
        apply_step_penalty=True,
    ):
        if current_cell_value is None:
            current_cell_value = self.grid[self.agent_position[0]][self.agent_position[1]]
        if current_cell_value == self.block_position_value:
            return self.block_reward
        if current_cell_value == self.lever_position_value:
            self.lever_collected = True
        if current_cell_value == self.target_position_value:
            if self.lever_collected or self.lever_position is None:
                return self.target_reward
            return 0

        d1 = np.sum(
            np.abs(np.array(prev_agent_position) - np.array(self.target_position))
        )
        d2 = np.sum(
            np.abs(np.array(self.agent_position) - np.array(self.target_position))
        )
        r = d1 - d2
        if apply_step_penalty and self.step_penalty:
            r -= self.step_penalty

        return r
