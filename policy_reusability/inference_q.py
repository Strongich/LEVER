import numpy as np
from env.gridworld import GridWorld
from env.init_gridworld import init_gridworld_1
import wandb
import time


# inference method
# the q_table.npy file must be in the same directory as this file

def inference_q(grid_world, q_table_path):

    # Load the Q-table
    q_table = np.load(q_table_path)

    #run = wandb.init(project="Inference_Q")
    total_time = 0
    total_reward = 0

    # Reset the environment to its initial state
    grid_world.reset().flatten()
    state_index = grid_world.state_to_index(grid_world.agent_position)


    # Maximum number of steps for inference
    max_steps_inference = 100
    path = []

    for step in range(max_steps_inference):
        # turn on stopwatch
        start_time = time.time()

        # greedy action selection (inference)
        action = np.argmax(q_table[state_index, :])
        path.append(action)

        # step
        grid, reward, done, _ = grid_world.step(action)
        total_reward += reward
        next_state_index = grid_world.state_to_index(grid_world.agent_position)

        # upadate state index
        state_index = next_state_index

        # check if the agent reached the target or the maximum number of steps is reached
        if done:
            if reward > 0:
                print("Agent reached the target!")
            else:
                print("Agent failed to reach the target!")
            break

        # turn of stopwatch
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        #wandb.log({"Total Inference Time": total_time}, step=step)
    
    #run.finish()
    return total_time, total_reward, path
