import random

from policy_reusability.env.gridworld import GridWorld


def init_gridworld_rand(
    reward_system, seed=42, grid_size=64, legacy_gold_exit_penalty: bool = False
):
    random.seed(seed)
    width_size = grid_size
    length_size = grid_size

    agent_initial_position = (0, 0)
    target_position = (width_size - 1, length_size - 1)

    # Parameters controlling density
    total_cells = grid_size * grid_size - 2  # Exclude agent and target
    num_golds = 50 if grid_size == 16 else int(total_cells * 0.2)
    num_blocks = 25 if grid_size == 16 else int(total_cells * 0.2)
    if grid_size == 16:
        hazard_count = 30
    else:
        hazard_count = max(1, int(total_cells * 0.05))

    def sample_layout():
        all_positions = [
            (x, y)
            for x in range(width_size)
            for y in range(length_size)
            if (x, y) not in [agent_initial_position, target_position]
        ]
        random.shuffle(all_positions)
        hazards = all_positions[:hazard_count]
        gold_start_idx = hazard_count
        golds = all_positions[gold_start_idx : gold_start_idx + num_golds]
        block_start_idx = gold_start_idx + num_golds
        blocks = all_positions[block_start_idx : block_start_idx + num_blocks]
        remaining = all_positions[block_start_idx + num_blocks :]
        lever = remaining[0] if remaining else None
        return hazards, golds, blocks, lever

    def has_hazard_free_path(start, goal, hazards, blocks):
        blocked = set(hazards) | set(blocks)
        if start in blocked or goal in blocked:
            return False
        queue = [start]
        visited = set(queue)
        while queue:
            x, y = queue.pop(0)
            if (x, y) == goal:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width_size
                    and 0 <= ny < length_size
                    and (nx, ny) not in blocked
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def layout_is_valid(hazards, blocks, lever):
        if not has_hazard_free_path(
            agent_initial_position, target_position, hazards, blocks
        ):
            return False
        if "lever" in reward_system:
            if lever is None:
                return False
            if not has_hazard_free_path(
                agent_initial_position, lever, hazards, blocks
            ):
                return False
            if not has_hazard_free_path(lever, target_position, hazards, blocks):
                return False
        return True

    max_layout_attempts = 200
    hazard_positions = []
    gold_positions = []
    block_positions = []
    lever_position = None
    for _ in range(max_layout_attempts):
        hazards, golds, blocks, lever = sample_layout()
        if layout_is_valid(hazards, blocks, lever):
            hazard_positions, gold_positions, block_positions, lever_position = (
                hazards,
                golds,
                blocks,
                lever,
            )
            break
    else:
        # As a last resort, drop hazards to guarantee a path.
        hazard_count = 0
        hazard_positions, gold_positions, block_positions, lever_position = sample_layout()

    gold_positions_list = [list(pos) for pos in gold_positions]
    block_positions_list = [list(pos) for pos in block_positions]
    hazard_positions_list = [list(pos) for pos in hazard_positions]

    # Define cell and reward values
    hazard_position_value = -2
    lever_position_value = 2
    cell_low_value = min(-1, hazard_position_value)
    cell_high_value = 10
    start_position_value = 5
    target_position_value = 10
    block_position_value = -1
    gold_position_value = +1
    agent_position_value = 7
    block_reward = -1
    target_reward = +100
    hazard_penalty = 0.0
    step_penalty = 0.0
    exit_without_lever_penalty = 0.0

    # Create GridWorld instance
    grid_world = GridWorld(
        grid_width=width_size,
        grid_length=length_size,
        gold_positions=gold_positions_list,
        block_positions=block_positions_list,
        hazard_positions=hazard_positions_list,
        lever_position=list(lever_position) if lever_position else None,
        reward_system=reward_system,
        agent_position=list(agent_initial_position),
        target_position=list(target_position),
        cell_high_value=cell_high_value,
        cell_low_value=cell_low_value,
        start_position_value=start_position_value,
        target_position_value=target_position_value,
        block_position_value=block_position_value,
        gold_position_value=gold_position_value,
        hazard_position_value=hazard_position_value,
        lever_position_value=lever_position_value,
        agent_position_value=agent_position_value,
        block_reward=block_reward,
        target_reward=target_reward,
        gold_k=0,
        n=0,
        action_size=4,
        parameterized=False,
        alpha_beta=(1, 1),
        step_penalty=step_penalty,
        hazard_penalty=hazard_penalty,
        exit_without_lever_penalty=exit_without_lever_penalty,
        legacy_gold_exit_penalty=legacy_gold_exit_penalty,
    )

    return grid_world
