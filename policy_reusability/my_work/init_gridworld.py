import random

from policy_reusability.env.gridworld import GridWorld


def init_gridworld_rand(reward_system, seed=42, grid_size=64):
    random.seed(seed)

    # ======== Environment setup ========
    # reward_system = "gold"
    width_size = grid_size
    length_size = grid_size

    # Define positions
    agent_initial_position = (0, 0)
    target_position = (width_size - 1, length_size - 1)

    # Parameters controlling density
    # Scale based on grid size: ~20% density
    if grid_size == 16:
        num_golds = 50
        num_blocks = 50
    elif grid_size == 64:
        num_golds = 800
        num_blocks = 800
    else:
        # Default: scale proportionally
        density = 0.2
        total_cells = grid_size * grid_size - 2  # Exclude agent and target
        num_golds = int(total_cells * density)
        num_blocks = int(total_cells * density)

    # Generate all possible coordinates except agent and target
    all_positions = [
        (x, y)
        for x in range(width_size)
        for y in range(length_size)
        if (x, y) not in [agent_initial_position, target_position]
    ]

    # Randomly sample gold and block positions without overlap
    random.shuffle(all_positions)
    gold_positions = all_positions[:num_golds]
    block_positions = all_positions[num_golds : num_golds + num_blocks]

    # Convert to lists for GridWorld
    gold_positions_list = [list(pos) for pos in gold_positions]
    block_positions_list = [list(pos) for pos in block_positions]

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

    # Create GridWorld instance
    grid_world = GridWorld(
        grid_width=width_size,
        grid_length=length_size,
        gold_positions=gold_positions_list,
        block_positions=block_positions_list,
        reward_system=reward_system,
        agent_position=list(agent_initial_position),
        target_position=list(target_position),
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

    return grid_world
