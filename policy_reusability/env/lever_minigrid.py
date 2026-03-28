"""
Custom MiniGrid environment replicating the LEVER gridworld.

Grid specifications (balanced for PPO learning):
- 16x16 grid
- 25 balls (gold) - collection gives reward
- 8 blocks (walls)
- 8 hazards (lava)
- 1 key - collecting it before exit gives bonus reward
- 1 exit (goal) - reaching it ends episode

Actions (native MiniGrid):
- 0: Turn left
- 1: Turn right
- 2: Move forward
- 3: Pick up object
- 4: Drop object
- 5: Toggle (interact with object)
- 6: Done (unused)

Cardinal action mode (optional):
- 0: Move right
- 1: Move down
- 2: Move left
- 3: Move up
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


class LeverGridEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with collectible balls, lava hazards,
    walls, and a key that provides bonus reward when exiting.

    The agent must navigate the grid, optionally collecting balls for
    intermediate rewards, and reach the exit. Collecting the key before
    exiting yields a larger reward than exiting without it.
    """

    def __init__(
        self,
        size: int = 16,
        num_balls: int = 25,
        num_walls: int = 8,
        num_lava: int = 8,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int | None = None,
        max_steps: int | None = None,
        # Reward configuration
        reward_system: str = "path-gold-hazard-lever",
        ball_reward: float = 5.0,
        key_reward: float = 20.0,
        exit_reward: float = 50.0,
        exit_with_key_reward: float = 100.0,
        lava_penalty: float = -1.0,
        step_penalty: float = 0.0,
        path_progress_scale: float = 0.5,
        action_mode: str = "minigrid",
        **kwargs,
    ):
        """
        Initialize the LeverGrid environment.

        Args:
            size: Grid size (size x size)
            num_balls: Number of collectible balls (gold)
            num_walls: Number of wall blocks
            num_lava: Number of lava/hazard cells
            agent_start_pos: Fixed agent start position, or None for random
            agent_start_dir: Fixed agent start direction (0-3), or None for random
            max_steps: Maximum steps per episode (default: 2 * size^2)
            reward_system: Reward components (path, gold, hazard, lever)
            ball_reward: Reward for collecting a ball (gold objective)
            key_reward: Reward for collecting the key (lever objective)
            exit_reward: Reward for reaching exit (path objective)
            exit_with_key_reward: Reward for exiting with key (lever objective)
                when path is also active, this acts as a bonus above exit_reward
            lava_penalty: Penalty for stepping on lava
            step_penalty: Small penalty per step to encourage efficiency
            path_progress_scale: Scale for path proximity shaping
            action_mode: "minigrid" for 7 actions, "cardinal" for 4 actions
        """
        self.size = size
        self.num_balls = num_balls
        self.num_walls = num_walls
        self.num_lava = num_lava
        self._agent_start_pos = agent_start_pos
        self._agent_start_dir = agent_start_dir

        # Reward config
        self.reward_system = reward_system
        self.ball_reward = ball_reward
        self.key_reward = key_reward
        self.exit_reward = exit_reward
        self.exit_with_key_reward = exit_with_key_reward
        self.lava_penalty = lava_penalty
        self.step_penalty = step_penalty
        self.path_progress_scale = path_progress_scale
        self.action_mode = action_mode

        self.reward_components = self._parse_reward_system(reward_system)

        # Track collected items
        self.has_key = False
        self.balls_collected = 0
        self.goal_pos: tuple[int, int] | None = None
        self.goal_distances: dict[tuple[int, int], int] = {}
        self.key_distances: dict[tuple[int, int], int] = {}
        self.start_goal_dist: int | None = None
        self.key_pos: tuple[int, int] | None = None
        self.ball_positions: set[tuple[int, int]] = set()
        self.ball_distances: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        self.lava_positions: set[tuple[int, int]] = set()

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 2 * size * size

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

        if self.action_mode == "cardinal":
            self.action_space = spaces.Discrete(4)
        elif self.action_mode != "minigrid":
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

    @staticmethod
    def _gen_mission() -> str:
        return "Collect the key and reach the exit. Collect balls for bonus rewards."

    @staticmethod
    def _parse_reward_system(reward_system: str) -> set[str]:
        if reward_system in (None, "", "combined", "all"):
            return {"path", "gold", "hazard", "lever"}
        return {token for token in reward_system.split("-") if token}

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate the grid layout with all objects."""
        max_layout_attempts = 200
        for _ in range(max_layout_attempts):
            # Create empty grid
            self.grid = Grid(width, height)

            # Surround with walls
            self.grid.wall_rect(0, 0, width, height)

            # Reset state
            self.has_key = False
            self.balls_collected = 0

            # Place internal walls (blocks)
            for _ in range(self.num_walls):
                self.place_obj(Wall(), max_tries=100)

            # Place lava (hazards)
            for _ in range(self.num_lava):
                self.place_obj(Lava(), max_tries=100)

            # Place balls (gold)
            for _ in range(self.num_balls):
                self.place_obj(Ball(color="yellow"), max_tries=100)

            # Place the key
            self.place_obj(Key(color="blue"), max_tries=100)

            # Place the exit (goal)
            self.place_obj(Goal(), max_tries=100)
            self.goal_pos = self._find_goal_pos(width, height)
            self.key_pos = self._find_key_pos(width, height)
            self.ball_positions = set(
                self._find_positions_by_type(width, height, "ball")
            )
            self.lava_positions = set(
                self._find_positions_by_type(width, height, "lava")
            )

            # Place agent
            if self._agent_start_pos is not None:
                self.agent_pos = self._agent_start_pos
                self.agent_dir = (
                    self._agent_start_dir if self._agent_start_dir is not None else 0
                )
            else:
                self.place_agent()

            if self._layout_is_valid():
                break

        self.mission = self._gen_mission()
        self.goal_distances = self._compute_distances_to(self.goal_pos)
        self.key_distances = self._compute_distances_to(self.key_pos)
        self.ball_distances = {
            bp: self._compute_distances_to(bp) for bp in self.ball_positions
        }
        self.start_goal_dist = self.goal_distances.get(
            (int(self.agent_pos[0]), int(self.agent_pos[1]))
        )

    def step(self, action: int):
        """
        Execute one step in the environment.

        Overrides parent to add custom reward logic for:
        - Collecting balls
        - Collecting key
        - Reaching exit (with/without key)
        - Stepping on lava
        """
        if self.action_mode == "cardinal":
            if action not in (0, 1, 2, 3):
                raise ValueError(f"Invalid cardinal action: {action}")
            # MiniGrid directions: 0=right, 1=down, 2=left, 3=up
            self.agent_dir = int(action)
            action = self.actions.forward

        # Get the cell in front of the agent before the step
        fwd_pos = self.front_pos
        fwd_pos_tuple = (int(fwd_pos[0]), int(fwd_pos[1]))
        fwd_cell = self.grid.get(*fwd_pos)
        prev_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))

        # Pre-clear balls and keys so agent can walk onto them (they block by default)
        pending_pickup = None
        if action == self.actions.forward and fwd_cell is not None:
            if fwd_cell.type in ("ball", "key"):
                pending_pickup = fwd_cell.type
                self.grid.set(*fwd_pos_tuple, None)

        # Execute the action
        obs, _, terminated, truncated, info = super().step(action)
        curr_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))

        # Use custom reward only
        reward = 0.0

        use_path = "path" in self.reward_components
        use_gold = "gold" in self.reward_components
        use_hazard = "hazard" in self.reward_components
        use_lever = "lever" in self.reward_components

        # Add step penalty only when path is an objective.
        if use_path:
            reward += self.step_penalty

        # Check what happened based on action and cell
        if action == self.actions.forward:
            # Handle pre-cleared pickups (ball/key were removed before step so agent could move)
            if pending_pickup == "ball":
                self.balls_collected += 1
                if use_gold:
                    reward += self.ball_reward
                info["balls_collected"] = self.balls_collected
                if fwd_pos_tuple in self.ball_positions:
                    self.ball_positions.remove(fwd_pos_tuple)

            elif pending_pickup == "key":
                self.has_key = True
                if use_lever:
                    reward += self.key_reward
                info["has_key"] = True
                self.key_pos = None

            elif fwd_cell is not None:
                # Stepped on lava
                if fwd_cell.type == "lava":
                    reward += self.lava_penalty
                    terminated = True
                    info["death"] = "lava"

                # Reached goal
                elif fwd_cell.type == "goal":
                    if use_path:
                        reward += self.exit_reward
                    if use_gold:
                        reward += self.exit_reward
                    if use_hazard:
                        reward += self.exit_reward
                    if use_lever and self.has_key:
                        if use_path:
                            bonus = max(
                                0.0, self.exit_with_key_reward - self.exit_reward
                            )
                            reward += bonus
                        else:
                            reward += self.exit_with_key_reward
                        info["exit_bonus"] = True
                    elif use_lever:
                        if not (use_path or use_gold or use_hazard):
                            reward += self.exit_reward
                        info["exit_bonus"] = False
                    terminated = True
                    info["success"] = True

        elif action == self.actions.pickup:
            # Check if we picked up something
            if self.carrying is not None:
                if self.carrying.type == "ball":
                    self.balls_collected += 1
                    if use_gold:
                        reward += self.ball_reward
                    info["balls_collected"] = self.balls_collected
                    if fwd_pos_tuple in self.ball_positions:
                        self.ball_positions.remove(fwd_pos_tuple)
                    # Drop the ball immediately (we just collect it)
                    self.carrying = None

                elif self.carrying.type == "key":
                    self.has_key = True
                    if use_lever:
                        reward += self.key_reward
                    info["has_key"] = True
                    self.key_pos = None
                    # Keep carrying the key (or drop it, your choice)
                    # For simplicity, let's say picking up key marks it as collected
                    self.carrying = None

        info["has_key"] = self.has_key
        info["balls_collected"] = self.balls_collected

        # Path shaping: pure delta shaping toward the exit (BFS distance).
        # Use max_steps as proxy for "unreachable" positions (isolated pockets).
        if use_path and self.goal_pos is not None:
            prev_dist = self.goal_distances.get(prev_pos, self.max_steps)
            curr_dist = self.goal_distances.get(
                (int(self.agent_pos[0]), int(self.agent_pos[1])), self.max_steps
            )
            reward += self.path_progress_scale * (prev_dist - curr_dist)

        # Gold shaping: BFS delta toward nearest remaining ball, then to exit.
        if use_gold:
            agent_pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))
            if self.ball_positions:
                prev_dist = min(
                    self.ball_distances[bp].get(prev_pos, self.max_steps)
                    for bp in self.ball_positions
                )
                curr_dist = min(
                    self.ball_distances[bp].get(agent_pos_tuple, self.max_steps)
                    for bp in self.ball_positions
                )
                reward += self.path_progress_scale * (prev_dist - curr_dist)
            elif self.goal_pos is not None:
                prev_dist = self.goal_distances.get(prev_pos, self.max_steps)
                curr_dist = self.goal_distances.get(agent_pos_tuple, self.max_steps)
                reward += self.path_progress_scale * (prev_dist - curr_dist)

        # Lever shaping: move to key, then to exit (BFS delta).
        # Use max_steps as proxy for "unreachable" positions.
        if use_lever:
            agent_pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))
            if not self.has_key and self.key_pos is not None:
                prev_dist = self.key_distances.get(prev_pos, self.max_steps)
                curr_dist = self.key_distances.get(agent_pos_tuple, self.max_steps)
                reward += self.path_progress_scale * (prev_dist - curr_dist)
            elif self.goal_pos is not None:
                prev_dist = self.goal_distances.get(prev_pos, self.max_steps)
                curr_dist = self.goal_distances.get(agent_pos_tuple, self.max_steps)
                reward += self.path_progress_scale * (prev_dist - curr_dist)

        # Hazard shaping: move toward exit while avoiding lava proximity.
        if use_hazard and self.goal_pos is not None:
            prev_dist = abs(prev_pos[0] - self.goal_pos[0]) + abs(
                prev_pos[1] - self.goal_pos[1]
            )
            curr_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
                self.agent_pos[1] - self.goal_pos[1]
            )
            reward += prev_dist - curr_dist
            if self.start_goal_dist is not None and self.start_goal_dist > 0:
                progress = (self.start_goal_dist - curr_dist) / self.start_goal_dist
                reward += self.path_progress_scale * progress
            prev_hazard = self._hazard_neighbor_count(prev_pos)
            curr_hazard = self._hazard_neighbor_count(self.agent_pos)
            reward += 2.0 * (prev_hazard - curr_hazard)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.has_key = False
        self.balls_collected = 0
        return super().reset(**kwargs)

    def _find_goal_pos(self, width: int, height: int) -> tuple[int, int] | None:
        for x in range(width):
            for y in range(height):
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)
        return None

    def _find_key_pos(self, width: int, height: int) -> tuple[int, int] | None:
        for x in range(width):
            for y in range(height):
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == "key":
                    return (x, y)
        return None

    def _find_positions_by_type(
        self, width: int, height: int, cell_type: str
    ) -> list[tuple[int, int]]:
        positions = []
        for x in range(width):
            for y in range(height):
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == cell_type:
                    positions.append((x, y))
        return positions

    def _compute_distances_to(
        self, target_pos: tuple[int, int] | None
    ) -> dict[tuple[int, int], int]:
        if target_pos is None:
            return {}

        distances: dict[tuple[int, int], int] = {target_pos: 0}
        queue = deque([target_pos])

        while queue:
            x, y = queue.popleft()
            base_dist = distances[(x, y)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in distances:
                    continue
                if not self._is_passable(nx, ny):
                    continue
                distances[(nx, ny)] = base_dist + 1
                queue.append((nx, ny))

        return distances

    @staticmethod
    def _min_manhattan(
        pos: tuple[int, int], targets: set[tuple[int, int]]
    ) -> int | None:
        if not targets:
            return None
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    @staticmethod
    def _manhattan(
        pos: tuple[int, int] | list[int] | None,
        target: tuple[int, int] | None,
    ) -> int | None:
        if pos is None or target is None:
            return None
        return abs(int(pos[0]) - target[0]) + abs(int(pos[1]) - target[1])

    def _hazard_neighbor_count(self, pos: tuple[int, int]) -> int:
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (pos[0] + dx, pos[1] + dy)
            if neighbor in self.lava_positions:
                count += 1
        return count

    def _layout_is_valid(self) -> bool:
        if self.goal_pos is None or self.key_pos is None:
            return False
        start = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        if not self._is_reachable(start, self.goal_pos):
            return False
        if not self._is_reachable(start, self.key_pos):
            return False
        if not self._is_reachable(self.key_pos, self.goal_pos):
            return False
        return True

    def _is_reachable(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> bool:
        if start == goal:
            return True
        queue = [start]
        visited = {start}
        while queue:
            x, y = queue.pop(0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not self._is_passable(nx, ny):
                    continue
                if (nx, ny) == goal:
                    return True
                visited.add((nx, ny))
                queue.append((nx, ny))
        return False

    def _is_passable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.grid.width or y >= self.grid.height:
            return False
        cell = self.grid.get(x, y)
        if cell is None:
            return True
        return cell.type not in ("wall", "lava")


class LocalObsWrapper(gym.ObservationWrapper):
    """Direction-independent local observation: a top-down patch centered on the agent.

    Returns a *channel-first* ``(3, size, size)`` uint8 array extracted from
    the full grid encoding.  Out-of-bounds cells are padded with the wall
    encoding.  Channel-first layout avoids the ambiguous-shape problem with
    ``VecTransposeImage`` when ``size`` happens to equal 3.
    """

    def __init__(self, env: gym.Env, size: int = 3) -> None:
        super().__init__(env)
        self.patch_size = size
        # Channel-first: (C=3, H=size, W=size)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, size, size), dtype=np.uint8
        )

    def observation(self, obs):
        base = self.env.unwrapped
        full_grid = base.grid.encode()  # (width, height, 3)
        ax, ay = base.agent_pos
        half = self.patch_size // 2
        wall = np.array([OBJECT_TO_IDX["wall"], 0, 0], dtype=np.uint8)

        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        for dx in range(self.patch_size):
            for dy in range(self.patch_size):
                gx = int(ax) - half + dx
                gy = int(ay) - half + dy
                if 0 <= gx < full_grid.shape[0] and 0 <= gy < full_grid.shape[1]:
                    patch[dx, dy] = full_grid[gx, gy]
                else:
                    patch[dx, dy] = wall
        # Transpose to channel-first (3, H, W) for CNN consumption
        return patch.transpose(2, 0, 1)


class ScaleObsWrapper(gym.ObservationWrapper):
    """Rescale MiniGrid encoded observations to use the full [0, 255] range.

    MiniGrid encodes cells as (object_type, color, state) with max values of
    roughly (10, 5, 2).  SB3's CnnPolicy divides uint8 by 255, so the raw
    encodings end up squished into [0, 0.04] — too small for a CNN to
    discriminate.  This wrapper multiplies each channel by a per-channel
    scale factor so the values span [0, 255].
    """

    # Scale factors: 255 / max_value_per_channel (rounded down).
    # object_type max=10 → 25, color max=5 → 51, state max=2 → 127
    SCALES = np.array([25, 51, 127], dtype=np.uint8)

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # observation_space shape & dtype stay the same (uint8, 0-255)

    def observation(self, obs):
        # obs can be (H, W, 3) or (3, H, W); SCALES broadcast on last dim
        if obs.shape[-1] == 3:
            return (obs.astype(np.uint16) * self.SCALES).clip(0, 255).astype(np.uint8)
        else:
            # channel-first: (3, H, W)
            scales = self.SCALES[:, None, None]
            return (obs.astype(np.uint16) * scales).clip(0, 255).astype(np.uint8)


# Register the environment with Gymnasium
register(
    id="LeverGrid-16x16-v0",
    entry_point="policy_reusability.env.lever_minigrid:LeverGridEnv",
    kwargs={
        "size": 16,
        "num_balls": 25,
        "num_walls": 8,
        "num_lava": 8,
    },
)

register(
    id="LeverGrid-32x32-v0",
    entry_point="policy_reusability.env.lever_minigrid:LeverGridEnv",
    kwargs={
        "size": 32,
        "num_balls": 25,
        "num_walls": 8,
        "num_lava": 8,
    },
)


# Quick test when run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create environment
    env = LeverGridEnv(size=16, num_balls=25, num_walls=15, num_lava=15)
    obs, info = env.reset()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Mission: {env.mission}")

    # Render the grid
    img = env.render()
    if img is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title("LeverGrid Environment")
        plt.axis("off")
        plt.savefig("lever_grid_preview.png", dpi=150, bbox_inches="tight")
        plt.show()

    env.close()
