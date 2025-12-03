import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate_decay = 0.99, min_learning_rate = 0.01, learning_rate=0.1, discount_factor=0.99, exploration_rate_decay=0.99, min_exploration_rate=0.01, exploration_rate=1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay

        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.n_actions)  # Explore action space
        else:
            action = np.argmax(self.q_table[state])  # Exploit learned values
        return action
       
    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value

class SarsaAgent:
    def __init__(self, n_states, n_actions, learning_rate_decay = 0.9, min_learning_rate = 0.01, learning_rate=0.1, discount_factor=0.99, exploration_rate_decay=0.9, min_exploration_rate=0.01, exploration_rate=1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay

        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.n_actions)  # Explore action space
        else:
            action = np.argmax(self.q_table[state])  # Exploit learned values
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_value)
        self.q_table[state, action] = new_value
