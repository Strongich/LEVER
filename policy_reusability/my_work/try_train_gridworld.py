from inference_q import inference_q
from train_q_policy import train_q_policy

from policy_reusability.env.gridworld import (
    GridWorld,  # assumes your GridWorld class is here
)


def main():
    # ======== Environment setup (manual, same logic as init_gridworld_4) ========
    reward_system = "gold"
    width_size = 8
    length_size = 8

    # Define positions
    gold_positions = []
    for i in range(1, min(width_size, length_size) - 1):
        if i == 3 or i == 4 or i == 5:
            continue
        gold_positions.append([i, i])  # e.g., [1,1], [2,2]

    block_positions = []
    agent_initial_position = [0, 0]
    target_position = [width_size - 1, length_size - 1]

    # Define cell and reward values
    cell_low_value = -1
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -10
    target_reward = +100

    # Create GridWorld instance directly
    grid_world = GridWorld(
        grid_width=width_size,
        grid_length=length_size,
        gold_positions=gold_positions,
        block_positions=block_positions,
        reward_system=reward_system,
        agent_position=agent_initial_position,
        target_position=target_position,
        cell_high_value=cell_high_value,
        cell_low_value=cell_low_value,
        start_position_value=start_position_value,
        target_position_value=target_position_value,
        block_position_value=block_position_value,
        gold_position_value=gold_position_value,
        agent_position_value=agent_position_value,
        block_reward=block_reward,
        target_reward=target_reward,
        gold_k=0,
        n=0,
        action_size=5,
        parameterized=False,
        alpha_beta=(1, 1),
    )

    print("GridWorld initialized!")
    print(f"Size: {width_size}x{length_size}")
    print(f"Gold positions: {gold_positions}")
    print(f"Target position: {target_position}")

    # ======== Training parameters ========
    agent_type = "Sarsa"  # or "Q-learning"
    n_episodes = 10000
    max_steps_per_episode = 20
    learning_rate = 0.1
    discount_factor = 0.0
    result_step_size = 10
    q_table_output_path = "q_table_example.npy"

    # ======== Train one Q-learning/SARSA agent ========
    total_time, dag, _, _ = train_q_policy(
        grid_world,
        n_episodes,
        max_steps_per_episode,
        agent_type,
        q_table_output_path,
        result_step_size=result_step_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )

    print(f"\nTraining finished in {total_time:.2f}s using {agent_type}")
    print(f"Q-table saved to: {q_table_output_path}")

    # ======== Evaluate trained agent ========
    best_path, cumulative_reward, path = inference_q(
        grid_world=grid_world, q_table_path=q_table_output_path
    )

    print("\nEvaluation results:")
    print(f"Cumulative reward: {cumulative_reward}")
    print(f"Best path found: {path}")


if __name__ == "__main__":
    main()
