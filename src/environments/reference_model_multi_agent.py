"""Multi-agent reference grid environment."""
import logging

logger = logging.getLogger(__name__)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.environments import get_grid
from src.environments.actions import DOWN, LEFT, NO_OP, RIGHT, UP


class ReferenceModel(MultiAgentEnv):
    """
    Multi-agent grid world with flat per-agent observations.

    Flat observation layout:
    `local_obs` (flattened) + `goal_delta` + optional `goal_distance` + `action_mask`.
    """

    EMPTY_CELL = 0
    OBSTACLE_CELL = 1
    OTHER_AGENT_CELL = 2
    OWN_GOAL_CELL = 3
    OTHER_GOAL_CELL = 4
    TRAVERSABLE_LOCAL_VALUES = (EMPTY_CELL, OWN_GOAL_CELL, OTHER_GOAL_CELL)
    UNASSIGNED_OWNER = -1

    def __init__(self, env_config):
        super().__init__()

        self.step_count = 0
        self.steps_per_episode = env_config.get("steps_per_episode", 100)
        self._num_agents = int(env_config.get("num_agents", 2))
        self.sensor_range = env_config.get("sensor_range", 1)
        self.deterministic = env_config.get("deterministic", False)
        self.normalize_goal_delta = env_config.get("normalize_goal_delta", True)
        self.include_goal_distance = env_config.get("include_goal_distance", False)
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()
        self.render_env = env_config.get("render_env", False)
        self.info_mode = str(env_config.get("info_mode", "lite")).lower()
        self.lifelong_mapf = bool(env_config.get("lifelong_mapf", False))
        if self.info_mode not in {"lite", "full"}:
            msg = f"Unsupported info_mode '{self.info_mode}'. Expected 'lite' or 'full'."
            raise ValueError(msg)
        self.deadlock_window_steps = max(1, int(env_config.get("deadlock_window_steps", 8)))
        self.livelock_window_steps = max(1, int(env_config.get("livelock_window_steps", 16)))
        self.lock_nearby_manhattan = max(1, int(env_config.get("lock_nearby_manhattan", 2)))
        self.lock_progress_epsilon = float(env_config.get("lock_progress_epsilon", 1))
        self.lock_min_neighbors = max(1, int(env_config.get("lock_min_neighbors", 1)))
        self.enable_lock_metrics = bool(env_config.get("enable_lock_metrics", True))
        self.goal_reached_once = dict.fromkeys(self.agents, False)
        self._episode_blocking_count = 0.0
        self._episode_deadlock_events = 0.0
        self._episode_livelock_events = 0.0
        self._episode_deadlock_steps = 0.0
        self._episode_livelock_steps = 0.0
        self._deadlock_state_prev = False
        self._livelock_state_prev = False
        self._agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agents)}
        self._coord_dtype = np.int16

        # Initialize a random number generator with the provided seed
        self.seed = env_config.get("seed", None)
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

        # TODO: Implement the environment initialization depending on the env_config
        # 0 - empty cell, 1 - obstacle,
        # coords are [y-1, x-1] from upper left, so [0, 4] is the aisle
        self.grid = get_grid.get_grid(env_config["env_name"])

        self._free_positions = np.argwhere(self.grid == self.EMPTY_CELL).astype(self._coord_dtype, copy=False)
        self._starts_arr = np.zeros((self._num_agents, 2), dtype=self._coord_dtype)
        self._positions_arr = np.zeros((self._num_agents, 2), dtype=self._coord_dtype)
        self._goals_arr = np.zeros((self._num_agents, 2), dtype=self._coord_dtype)
        self._reached_arr = np.zeros(self._num_agents, dtype=np.bool_)
        self._completed_once_arr = np.zeros(self._num_agents, dtype=np.bool_)
        self._episode_goals_reached_total = 0.0
        self._scratch_prev_positions = np.empty_like(self._positions_arr)
        self._scratch_intended_next = np.empty_like(self._positions_arr)
        self._scratch_reached_goal = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_goal_reached_step_flags = np.zeros(self._num_agents, dtype=np.float32)
        self._scratch_blocking_flags = np.zeros(self._num_agents, dtype=np.float32)
        self._scratch_actions = np.zeros(self._num_agents, dtype=np.int8)
        self._scratch_moved_flags = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_failed_move_flags = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_goal_progress_flags = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_prev_on_goal = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_current_on_goal = np.zeros(self._num_agents, dtype=np.bool_)
        self._scratch_distance_to_goal = np.zeros(self._num_agents, dtype=np.int16)
        self._occupancy_owner = np.full(self.grid.shape, self.UNASSIGNED_OWNER, dtype=np.int16)
        self._goal_owner = np.full(self.grid.shape, self.UNASSIGNED_OWNER, dtype=np.int16)
        self._action_deltas = np.array(
            [
                [0, 0],  # no-op
                [-1, 0],  # up
                [0, 1],  # right
                [1, 0],  # down
                [0, -1],  # left
            ],
            dtype=self._coord_dtype,
        )
        self._lock_history_size = max(self.deadlock_window_steps, self.livelock_window_steps)
        self._lock_hist_goal_progress = np.zeros((self._lock_history_size, self._num_agents), dtype=np.uint8)
        self._lock_hist_moved = np.zeros((self._lock_history_size, self._num_agents), dtype=np.uint8)
        self._lock_hist_failed_move = np.zeros((self._lock_history_size, self._num_agents), dtype=np.uint8)
        self._lock_hist_distance = np.zeros((self._lock_history_size, self._num_agents), dtype=np.int16)
        self._lock_hist_count = 0
        self._lock_hist_head = 0

        self._bind_public_state_views()

        if self.deterministic:
            starts = get_grid.get_start_positions(env_config["env_name"], self._num_agents)
            goals = get_grid.get_goal_positions(env_config["env_name"], self._num_agents)
            for agent_id, idx in self._agent_index.items():
                self._starts_arr[idx] = np.asarray(starts[agent_id], dtype=self._coord_dtype)
                self._goals_arr[idx] = np.asarray(goals[agent_id], dtype=self._coord_dtype)
            np.copyto(self._positions_arr, self._starts_arr)
            self._rebuild_goal_owner()
            self._rebuild_occupancy_owner()
        else:
            self.generate_starts_goals()

        # POMPD, small grid around the agent
        self._view_side = self.sensor_range * 2 + 1
        self._local_obs_space = gym.spaces.Box(
            low=0,
            high=self.OTHER_GOAL_CELL,
            shape=(self._view_side, self._view_side),
            dtype=np.uint8,
        )
        goal_delta_low = np.array(
            [-(self.grid.shape[0] - 1), -(self.grid.shape[1] - 1)],
            dtype=np.float32,
        )
        goal_delta_high = np.array(
            [self.grid.shape[0] - 1, self.grid.shape[1] - 1],
            dtype=np.float32,
        )
        self._goal_delta_denominator = np.array(
            [max(self.grid.shape[0] - 1, 1), max(self.grid.shape[1] - 1, 1)],
            dtype=np.float32,
        )
        if self.normalize_goal_delta:
            goal_delta_low = goal_delta_low / self._goal_delta_denominator
            goal_delta_high = goal_delta_high / self._goal_delta_denominator
        self._goal_delta_space = gym.spaces.Box(
            low=goal_delta_low,
            high=goal_delta_high,
            shape=(2,),
            dtype=np.float32,
        )
        # Assuming all agents have the same action space
        self._single_act_space = gym.spaces.Discrete(5)
        self._action_mask_space = gym.spaces.MultiBinary(self._single_act_space.n)

        self._single_obs_space, self._obs_slices = self._build_obs_layout()
        self._single_obs_len = int(np.prod(self._single_obs_space.shape))

        self.observation_spaces = dict.fromkeys(self.possible_agents, self._single_obs_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, self._single_act_space)

        self.observation_space = self._single_obs_space
        self.action_space = self._single_act_space

        if self.render_env:
            # Initialize rendering attributes
            self.agent_patches = {}  # Initialize agent_patches as an empty dictionary
            self.goal_patches = {}  # Initialize goal_patches as an empty dictionary
            self.sensor_patches = {}
            self.fig = None
            self.ax = None

    def _bind_public_state_views(self):
        """Expose dict-based state views for compatibility with existing callers."""
        self.starts = {agent_id: self._starts_arr[idx] for agent_id, idx in self._agent_index.items()}
        self.positions = {agent_id: self._positions_arr[idx] for agent_id, idx in self._agent_index.items()}
        self.goals = {agent_id: self._goals_arr[idx] for agent_id, idx in self._agent_index.items()}

    def _rebuild_occupancy_owner(self):
        """Rebuild map from grid cell to occupying agent index."""
        self._occupancy_owner.fill(self.UNASSIGNED_OWNER)
        for idx in range(self._num_agents):
            x, y = self._positions_arr[idx]
            self._occupancy_owner[x, y] = idx

    def _rebuild_goal_owner(self):
        """Rebuild map from grid cell to goal owner agent index."""
        self._goal_owner.fill(self.UNASSIGNED_OWNER)
        for idx in range(self._num_agents):
            x, y = self._goals_arr[idx]
            self._goal_owner[x, y] = idx

    def _build_obs_layout(self):
        """Build flat observation space and component slices."""
        component_order = ["local_obs", "goal_delta"]
        component_spaces = {
            "local_obs": self._local_obs_space,
            "goal_delta": self._goal_delta_space,
        }
        if self.include_goal_distance:
            component_order.append("goal_distance")
            max_manhattan = float(np.abs(self._goal_delta_space.high).sum())
            component_spaces["goal_distance"] = gym.spaces.Box(
                low=np.zeros(1, dtype=np.float32),
                high=np.asarray([max_manhattan], dtype=np.float32),
                dtype=np.float32,
            )
        component_order.append("action_mask")
        component_spaces["action_mask"] = self._action_mask_space

        low_parts = []
        high_parts = []
        obs_slices = {}
        start = 0
        for name in component_order:
            space = component_spaces[name]
            if isinstance(space, gym.spaces.MultiBinary):
                size = int(np.prod(space.shape))
                low = np.zeros(size, dtype=np.float32)
                high = np.ones(size, dtype=np.float32)
            else:
                low = space.low.astype(np.float32).reshape(-1)
                high = space.high.astype(np.float32).reshape(-1)
            end = start + low.size
            obs_slices[name] = slice(start, end)
            low_parts.append(low)
            high_parts.append(high)
            start = end

        obs_space = gym.spaces.Box(
            low=np.concatenate(low_parts),
            high=np.concatenate(high_parts),
            dtype=np.float32,
        )
        return obs_space, obs_slices

    def generate_starts_goals(self):
        """Generate unique random start and goal positions for all agents."""
        required_positions = self._num_agents * 2
        if self._free_positions.shape[0] < required_positions:
            msg = (
                f"Environment has only {self._free_positions.shape[0]} free cells, "
                f"but {required_positions} are required for starts and goals."
            )
            raise ValueError(msg)

        sampled_indices = self.rng.choice(self._free_positions.shape[0], size=required_positions, replace=False)
        np.copyto(self._starts_arr, self._free_positions[sampled_indices[: self._num_agents]])
        np.copyto(self._positions_arr, self._starts_arr)
        np.copyto(self._goals_arr, self._free_positions[sampled_indices[self._num_agents :]])
        self._rebuild_goal_owner()
        self._rebuild_occupancy_owner()

    def _assign_new_goal(self, agent_idx: int) -> np.ndarray:
        """Assign a new unique, currently unoccupied goal to one agent."""
        old_goal_y = int(self._goals_arr[agent_idx, 0])
        old_goal_x = int(self._goals_arr[agent_idx, 1])
        self._goal_owner[old_goal_y, old_goal_x] = self.UNASSIGNED_OWNER

        free_y = self._free_positions[:, 0]
        free_x = self._free_positions[:, 1]
        occupied = self._occupancy_owner[free_y, free_x] != self.UNASSIGNED_OWNER
        already_goal = self._goal_owner[free_y, free_x] != self.UNASSIGNED_OWNER
        candidate_mask = (~occupied) & (~already_goal)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            msg = "No valid cell available for lifelong goal reassignment."
            raise RuntimeError(msg)

        selected_idx = int(candidate_indices[int(self.rng.integers(candidate_indices.size))])
        new_goal = self._free_positions[selected_idx]
        self._goals_arr[agent_idx, :] = new_goal
        self._goal_owner[int(new_goal[0]), int(new_goal[1])] = agent_idx
        return new_goal

    def _flatten_observation(self, agent_id: str, local_obs=None, action_mask=None):
        """Pack ordered observation components into one flat float32 vector."""
        if local_obs is None:
            local_obs = self.get_obs(agent_id)
        if action_mask is None:
            action_mask = self.get_action_mask(local_obs)

        goal_delta = self._get_goal_delta(agent_id)
        flat_obs = np.empty(self._single_obs_len, dtype=np.float32)
        flat_obs[self._obs_slices["local_obs"]] = local_obs.reshape(-1)
        flat_obs[self._obs_slices["goal_delta"]] = goal_delta
        if self.include_goal_distance:
            flat_obs[self._obs_slices["goal_distance"]] = np.abs(goal_delta).sum()
        flat_obs[self._obs_slices["action_mask"]] = action_mask.reshape(-1)
        return flat_obs

    def _get_goal_delta(self, agent_id: str) -> np.ndarray:
        agent_idx = self._agent_index[agent_id]
        goal_delta = (self._goals_arr[agent_idx] - self._positions_arr[agent_idx]).astype(np.float32)
        if self.normalize_goal_delta:
            goal_delta = goal_delta / self._goal_delta_denominator
        return goal_delta

    def _build_full_info(self, agent_id: str, local_obs: np.ndarray, action_mask: np.ndarray) -> dict:
        """Build detailed per-agent info payload (debug/evaluation mode)."""
        return {
            "position": np.asarray(self.positions[agent_id]),
            "goal": np.asarray(self.goals[agent_id]),
            "goal_delta": self._get_goal_delta(agent_id),
            "action_mask": action_mask,
            "local_obs": local_obs,
        }

    def _reset_lock_tracking(self):
        self._lock_hist_goal_progress.fill(0)
        self._lock_hist_moved.fill(0)
        self._lock_hist_failed_move.fill(0)
        self._lock_hist_distance.fill(0)
        self._lock_hist_count = 0
        self._lock_hist_head = 0
        self._episode_deadlock_events = 0.0
        self._episode_livelock_events = 0.0
        self._episode_deadlock_steps = 0.0
        self._episode_livelock_steps = 0.0
        self._deadlock_state_prev = False
        self._livelock_state_prev = False

    def _append_lock_history_step(
        self,
        goal_progress_flags: np.ndarray,
        moved_flags: np.ndarray,
        failed_move_flags: np.ndarray,
        distance_to_goal: np.ndarray,
    ) -> None:
        row = self._lock_hist_head
        self._lock_hist_goal_progress[row, :] = goal_progress_flags.astype(np.uint8, copy=False)
        self._lock_hist_moved[row, :] = moved_flags.astype(np.uint8, copy=False)
        self._lock_hist_failed_move[row, :] = failed_move_flags.astype(np.uint8, copy=False)
        self._lock_hist_distance[row, :] = distance_to_goal.astype(np.int16, copy=False)
        self._lock_hist_head = (self._lock_hist_head + 1) % self._lock_history_size
        self._lock_hist_count = min(self._lock_hist_count + 1, self._lock_history_size)

    def _get_focal_participants(self, current_off_goal: np.ndarray) -> list[np.ndarray]:
        participants = []
        for focal_idx in np.flatnonzero(current_off_goal):
            focal_pos = self._positions_arr[focal_idx]
            dists = np.abs(self._positions_arr - focal_pos).sum(axis=1)
            neighbor_indices = np.flatnonzero((dists <= self.lock_nearby_manhattan) & (dists > 0))
            if neighbor_indices.size < self.lock_min_neighbors:
                continue
            participants.append(np.concatenate(([focal_idx], neighbor_indices)))
        return participants

    def _detect_lock_step(self, current_off_goal: np.ndarray) -> tuple[bool, bool]:
        if not bool(np.any(current_off_goal)):
            return False, False

        participant_sets = self._get_focal_participants(current_off_goal)
        if not participant_sets:
            return False, False

        if self._lock_hist_count >= self.deadlock_window_steps:
            idxs = (self._lock_hist_head - np.arange(self.deadlock_window_steps, 0, -1, dtype=np.int32)) % (
                self._lock_history_size
            )
            goal_progress_window = self._lock_hist_goal_progress[idxs]
            moved_window = self._lock_hist_moved[idxs]
            failed_move_window = self._lock_hist_failed_move[idxs]
            for participants in participant_sets:
                goal_progress_sum = float(goal_progress_window[:, participants].sum())
                moved_sum = float(moved_window[:, participants].sum())
                failed_move_sum = float(failed_move_window[:, participants].sum())
                if goal_progress_sum <= 0.0 and moved_sum <= 0.0 and failed_move_sum > 0.0:
                    return True, False

        if self._lock_hist_count >= self.livelock_window_steps:
            idxs = (self._lock_hist_head - np.arange(self.livelock_window_steps, 0, -1, dtype=np.int32)) % (
                self._lock_history_size
            )
            goal_progress_window = self._lock_hist_goal_progress[idxs]
            moved_window = self._lock_hist_moved[idxs]
            distance_window = self._lock_hist_distance[idxs]
            for participants in participant_sets:
                goal_progress_sum = float(goal_progress_window[:, participants].sum())
                moved_sum = float(moved_window[:, participants].sum())
                distance_start = float(distance_window[0, participants].sum())
                distance_end = float(distance_window[-1, participants].sum())
                distance_reduction = distance_start - distance_end
                if goal_progress_sum <= 0.0 and moved_sum > 0.0 and distance_reduction <= self.lock_progress_epsilon:
                    return False, True

        return False, False

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self._episode_blocking_count = 0.0
        self._episode_goals_reached_total = 0.0
        self._reset_lock_tracking()
        infos = {aid: {} for aid in self.agents}
        obs = {}
        self._reached_arr[:] = False
        self._completed_once_arr[:] = False
        self.goal_reached_once = dict.fromkeys(self.agents, False)

        if self.deterministic:
            np.copyto(self._positions_arr, self._starts_arr)
            self._rebuild_goal_owner()
            self._rebuild_occupancy_owner()
        else:
            self.generate_starts_goals()

        for agent_id in self.agents:
            local_obs = self.get_obs(agent_id)
            action_mask = self.get_action_mask(local_obs)
            obs[agent_id] = self._flatten_observation(agent_id, local_obs, action_mask)
            if self.info_mode == "full":
                infos[agent_id] = self._build_full_info(agent_id, local_obs, action_mask)
        if self.render_env:
            self.render()

        return obs, infos

    def step(self, action_dict):
        self.step_count += 1
        rewards = dict.fromkeys(self.agents, 0.0)
        info = {aid: {} for aid in self.agents}
        obs = {}
        goal_reassigned = False
        reached_goal = self._scratch_reached_goal
        reached_goal[:] = False
        intended_next = self._scratch_intended_next
        prev_positions = self._scratch_prev_positions
        np.copyto(prev_positions, self._positions_arr)
        goal_reached_step_flags = self._scratch_goal_reached_step_flags
        goal_reached_step_flags.fill(0.0)
        blocking_flags = self._scratch_blocking_flags
        blocking_flags.fill(0.0)
        actions_taken = self._scratch_actions
        actions_taken.fill(NO_OP)
        moved_flags = self._scratch_moved_flags
        failed_move_flags = self._scratch_failed_move_flags
        goal_progress_flags = self._scratch_goal_progress_flags
        prev_on_goal = self._scratch_prev_on_goal
        current_on_goal = self._scratch_current_on_goal
        distance_to_goal = self._scratch_distance_to_goal

        if not action_dict or any(aid not in action_dict for aid in self.agents):
            action_dict = dict.fromkeys(self.agents, NO_OP)
            logger.warning("No actions provided or missing agent actions. Defaulting to no-op actions: %s", action_dict)

        for agent_idx, agent_id in enumerate(self.agents):
            action = int(action_dict[agent_id])
            if action < NO_OP or action > LEFT:
                msg = f"Invalid action {action} for {agent_id}"
                raise ValueError(msg)
            actions_taken[agent_idx] = action

            pos_y = int(self._positions_arr[agent_idx, 0])
            pos_x = int(self._positions_arr[agent_idx, 1])
            delta = self._action_deltas[action]
            next_y = pos_y + int(delta[0])
            next_x = pos_x + int(delta[1])
            intended_next[agent_idx, 0] = next_y
            intended_next[agent_idx, 1] = next_x
            valid_move = (
                0 <= next_y < self.grid.shape[0]
                and 0 <= next_x < self.grid.shape[1]
                and self.grid[next_y, next_x] == self.EMPTY_CELL
                and (self._occupancy_owner[next_y, next_x] in (self.UNASSIGNED_OWNER, agent_idx))
            )
            if valid_move and (next_y != pos_y or next_x != pos_x):
                self._occupancy_owner[pos_y, pos_x] = self.UNASSIGNED_OWNER
                self._positions_arr[agent_idx, 0] = next_y
                self._positions_arr[agent_idx, 1] = next_x
                self._occupancy_owner[next_y, next_x] = agent_idx

            local_obs = self.get_obs(agent_id)
            action_mask = self.get_action_mask(local_obs)
            obs[agent_id] = self._flatten_observation(agent_id, local_obs, action_mask)
            if self.info_mode == "full":
                info[agent_id] = self._build_full_info(agent_id, local_obs, action_mask)

            current_y = int(self._positions_arr[agent_idx, 0])
            current_x = int(self._positions_arr[agent_idx, 1])
            goal_y = int(self._goals_arr[agent_idx, 0])
            goal_x = int(self._goals_arr[agent_idx, 1])
            is_on_goal = current_y == goal_y and current_x == goal_x
            reached_goal[agent_idx] = is_on_goal
            if not is_on_goal:
                continue

            if self.lifelong_mapf:
                rewards[agent_id] += 0.5
                goal_reached_step_flags[agent_idx] = 1.0
                self._episode_goals_reached_total += 1.0
                self._completed_once_arr[agent_idx] = True
                self.goal_reached_once[agent_id] = True
                self._reached_arr[agent_idx] = False
                self._assign_new_goal(agent_idx)
                reached_goal[agent_idx] = False
                goal_reassigned = True
            elif not self._reached_arr[agent_idx]:
                self._reached_arr[agent_idx] = True
                self._completed_once_arr[agent_idx] = True
                self.goal_reached_once[agent_id] = True
                rewards[agent_id] += 0.5
                goal_reached_step_flags[agent_idx] = 1.0
                self._episode_goals_reached_total += 1.0

        if goal_reassigned:
            for agent_id in self.agents:
                local_obs = self.get_obs(agent_id)
                action_mask = self.get_action_mask(local_obs)
                obs[agent_id] = self._flatten_observation(agent_id, local_obs, action_mask)
                if self.info_mode == "full":
                    info[agent_id] = self._build_full_info(agent_id, local_obs, action_mask)

        deadlock_step = False
        livelock_step = False
        deadlock_event_step = 0.0
        livelock_event_step = 0.0
        if self.enable_lock_metrics:
            moved_flags[:] = np.any(self._positions_arr != prev_positions, axis=1)
            failed_move_flags[:] = (actions_taken != NO_OP) & (~moved_flags)
            if self.lifelong_mapf:
                prev_on_goal[:] = False
            else:
                prev_on_goal[:] = np.all(prev_positions == self._goals_arr, axis=1)
            current_on_goal[:] = np.all(self._positions_arr == self._goals_arr, axis=1)
            if self.lifelong_mapf:
                goal_progress_flags[:] = goal_reached_step_flags > 0.0
            else:
                goal_progress_flags[:] = (~prev_on_goal) & current_on_goal
            distance_to_goal[:] = np.abs(self._goals_arr - self._positions_arr).sum(axis=1)
            self._append_lock_history_step(goal_progress_flags, moved_flags, failed_move_flags, distance_to_goal)
            current_off_goal = ~current_on_goal
            deadlock_step, livelock_step = self._detect_lock_step(current_off_goal)
            if deadlock_step:
                livelock_step = False
            deadlock_event_step = float(deadlock_step and not self._deadlock_state_prev)
            livelock_event_step = float(livelock_step and not self._livelock_state_prev)
            self._deadlock_state_prev = bool(deadlock_step)
            self._livelock_state_prev = bool(livelock_step)
            self._episode_deadlock_steps += float(deadlock_step)
            self._episode_livelock_steps += float(livelock_step)
            self._episode_deadlock_events += deadlock_event_step
            self._episode_livelock_events += livelock_event_step

        # Track intent-based local blocking events.
        for blocker_idx, blocker_id in enumerate(self.agents):
            if not self._reached_arr[blocker_idx]:
                continue
            blocker_y = int(self._positions_arr[blocker_idx, 0])
            blocker_x = int(self._positions_arr[blocker_idx, 1])
            prev_blocker_y = int(prev_positions[blocker_idx, 0])
            prev_blocker_x = int(prev_positions[blocker_idx, 1])
            if blocker_y != prev_blocker_y or blocker_x != prev_blocker_x:
                continue
            for other_idx, other_id in enumerate(self.agents):
                if other_id == blocker_id or self._reached_arr[other_idx]:
                    continue
                if int(intended_next[other_idx, 0]) == blocker_y and int(intended_next[other_idx, 1]) == blocker_x:
                    blocking_flags[blocker_idx] = 1.0
                    break
        self._episode_blocking_count += float(blocking_flags.sum())

        for agent_idx, agent_id in enumerate(self.agents):
            info[agent_id]["blocking"] = float(blocking_flags[agent_idx])
            info[agent_id]["goal_reached_step"] = float(goal_reached_step_flags[agent_idx])
        if self.lifelong_mapf:
            goals_reached_total = float(self._episode_goals_reached_total)
        else:
            goals_reached_total = float(np.sum(self._reached_arr))
        blocking_count_total = float(self._episode_blocking_count)
        for agent_id in self.agents:
            info[agent_id]["goals_reached_total"] = goals_reached_total
            info[agent_id]["blocking_count_total"] = blocking_count_total
        completion_ratio = float(np.mean(self._completed_once_arr))
        info_all = {
            "goals_reached_step": float(goal_reached_step_flags.sum()),
            "goals_reached_total": goals_reached_total,
            "blocking_count_step": float(blocking_flags.sum()),
            "blocking_count_total": blocking_count_total,
            "deadlock_step": float(deadlock_step),
            "livelock_step": float(livelock_step),
            "deadlock_event_step": deadlock_event_step,
            "livelock_event_step": livelock_event_step,
            "deadlock_events_total": float(self._episode_deadlock_events),
            "livelock_events_total": float(self._episode_livelock_events),
            "deadlock_steps_total": float(self._episode_deadlock_steps),
            "livelock_steps_total": float(self._episode_livelock_steps),
        }
        if self.lifelong_mapf:
            info_all["completion_ratio"] = completion_ratio
            info_all["throughput"] = goals_reached_total / float(max(self.step_count, 1))
        info["__all__"] = info_all

        # Collision penalty (shouldn't happen, but keep it)
        for i, a in enumerate(self.agents):
            pos_a_y = int(self._positions_arr[i, 0])
            pos_a_x = int(self._positions_arr[i, 1])
            for j, b in enumerate(self.agents[i + 1 :], start=i + 1):
                if pos_a_y == int(self._positions_arr[j, 0]) and pos_a_x == int(self._positions_arr[j, 1]):
                    rewards[a] -= 1
                    rewards[b] -= 1
                    logger.warning("Agents %s and %s occupy the same position %s", a, b, self.positions[a])

        terminated = dict.fromkeys(self.agents, False)
        truncated = dict.fromkeys(self.agents, False)
        # If all agents have reached their goals, end the episode (not truncated)
        if not self.lifelong_mapf and bool(np.all(reached_goal)):
            for aid in self.agents:
                rewards[aid] += 1
                terminated[aid] = True
            terminated["__all__"] = True
            truncated["__all__"] = False
        elif (
            self.step_count >= self.steps_per_episode
        ):  # If the step limit is reached, end the episode and mark it as truncated
            # minus reward at truncation for agents not on their goal
            for agent_idx, aid in enumerate(self.agents):
                if not self.lifelong_mapf and not reached_goal[agent_idx]:
                    rewards[aid] -= 1
                terminated[aid] = True
                truncated[aid] = True
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        if self.render_env:
            self.render()

        return obs, rewards, terminated, truncated, info

    def get_next_position(self, action: int, pos):
        """Return next grid position for action [no-op, up, right, down, left]."""
        action = int(action)
        if action < NO_OP or action > LEFT:
            msg = "Invalid action"
            raise ValueError(msg)

        pos_arr = np.asarray(pos, dtype=self._coord_dtype)
        return pos_arr + self._action_deltas[action]

    def get_obs(self, agent_id: str):
        """Return local observation grid for one agent using encoded cell constants."""
        agent_idx = self._agent_index[agent_id]
        pos = self._positions_arr[agent_idx]
        obs = np.full(
            self._local_obs_space.shape,
            fill_value=self.OBSTACLE_CELL,
            dtype=np.uint8,
        )
        base_x = int(pos[0]) - self.sensor_range
        base_y = int(pos[1]) - self.sensor_range
        grid_rows, grid_cols = self.grid.shape
        occupancy_owner = self._occupancy_owner
        goal_owner = self._goal_owner

        for i in range(self._local_obs_space.shape[0]):
            x = base_x + i
            if x < 0 or x >= grid_rows:
                continue
            for j in range(self._local_obs_space.shape[1]):
                y = base_y + j
                if y < 0 or y >= grid_cols:
                    continue
                if self.grid[x, y] == self.OBSTACLE_CELL:
                    obs[i, j] = self.OBSTACLE_CELL
                    continue

                occ = occupancy_owner[x, y]
                if occ not in (self.UNASSIGNED_OWNER, agent_idx):
                    obs[i, j] = self.OTHER_AGENT_CELL
                    continue

                goal_idx = goal_owner[x, y]
                if goal_idx == agent_idx:
                    obs[i, j] = self.OWN_GOAL_CELL
                elif goal_idx != self.UNASSIGNED_OWNER:
                    obs[i, j] = self.OTHER_GOAL_CELL
                else:
                    obs[i, j] = self.EMPTY_CELL

        return obs

    def get_action_mask(self, obs):
        """Return binary mask for valid local actions in [no-op, up, right, down, left]."""
        action_mask = np.zeros(
            self._action_mask_space.shape,
            dtype=self._action_mask_space.dtype,
        )

        action_mask[NO_OP] = 1  # No-op action is always possible

        x = self.sensor_range
        y = self.sensor_range

        if x > 0 and obs[x - 1, y] in self.TRAVERSABLE_LOCAL_VALUES:
            action_mask[UP] = 1  # Move up

        if y < obs.shape[1] - 1 and obs[x, y + 1] in self.TRAVERSABLE_LOCAL_VALUES:
            action_mask[RIGHT] = 1  # Move right

        if x < obs.shape[0] - 1 and obs[x + 1, y] in self.TRAVERSABLE_LOCAL_VALUES:
            action_mask[DOWN] = 1  # Move down

        if y > 0 and obs[x, y - 1] in self.TRAVERSABLE_LOCAL_VALUES:
            action_mask[LEFT] = 1  # Move left

        return action_mask

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): "human" updates the Matplotlib figure, "rgb_array" returns an RGB frame.

        """
        if mode not in {"human", "rgb_array"}:
            msg = f"Unsupported render mode {mode}. Expected 'human' or 'rgb_array'."
            raise ValueError(msg)

        if not hasattr(self, "fig") or self.fig is None:
            # Initialize the rendering environment if it hasn't been done yet
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(self.grid.shape[1] / 3, self.grid.shape[0] / 3))

            # Draw the grid
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] == 1:  # Obstacle cell
                        self.ax.add_patch(patches.Rectangle((j, i), 1, 1, color="black"))
                    else:  # Empty cell
                        self.ax.add_patch(patches.Rectangle((j, i), 1, 1, edgecolor="gray", fill=False))

            # Initialize goal patches
            self.goal_patches = {}
            colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "yellow",
                "brown",
                "pink",
                "olive",
                "teal",
                "navy",
                "gold",
                "lime",
                "gray",
            ]
            for i, agent_id in enumerate(self.agents):
                goal = self.goals[agent_id]
                goal_patch = patches.Polygon(
                    [
                        [goal[1] + 0.5, goal[0]],
                        [goal[1], goal[0] + 0.5],
                        [goal[1] + 0.5, goal[0] + 1],
                        [goal[1] + 1, goal[0] + 0.5],
                    ],
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    label=f"Goal {i}",
                )
                self.ax.add_patch(goal_patch)
                self.goal_patches[agent_id] = goal_patch

            # Initialize agent patches
            self.agent_patches = {}
            for i, agent_id in enumerate(self.agents):
                pos = self.positions[agent_id]
                agent_patch = patches.Circle(
                    (pos[1] + 0.5, pos[0] + 0.5),
                    0.3,
                    color=colors[i % len(colors)],
                    label=f"Agent {i}",
                )
                sensor_range_size = 2 * self.sensor_range + 1
                sensor_patch = patches.Rectangle(
                    (
                        pos[1] - self.sensor_range,
                        pos[0] - self.sensor_range,
                    ),
                    sensor_range_size,
                    sensor_range_size,
                    edgecolor=colors[i % len(colors)],
                    facecolor=colors[i % len(colors)],
                    alpha=0.2,
                    fill=True,
                )
                self.ax.add_patch(agent_patch)
                self.ax.add_patch(sensor_patch)
                self.agent_patches[agent_id] = agent_patch
                self.sensor_patches[agent_id] = sensor_patch

            # Set the limits and aspect
            self.ax.set_xlim(0, self.grid.shape[1])
            self.ax.set_ylim(0, self.grid.shape[0])
            self.ax.set_aspect("equal")

            # Add grid lines for clarity
            self.ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
            self.ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
            self.ax.grid(visible=True, which="both", color="gray", linestyle="-", linewidth=0.5)

            # Reverse the y-axis to match typical grid layout (0,0 at top-left)
            self.ax.invert_yaxis()

        else:
            # Update agent positions
            for agent_id, agent_patch in self.agent_patches.items():
                pos = self.positions[agent_id]
                agent_patch.set_center((pos[1] + 0.5, pos[0] + 0.5))
                sensor_patch = self.sensor_patches[agent_id]
                sensor_patch.set_xy((pos[1] - self.sensor_range, pos[0] - self.sensor_range))

            # Update goal positions
            for agent_id, goal_patch in self.goal_patches.items():
                goal = self.goals[agent_id]
                goal_vertices = [
                    [goal[1] + 0.5, goal[0]],
                    [goal[1], goal[0] + 0.5],
                    [goal[1] + 0.5, goal[0] + 1],
                    [goal[1] + 1, goal[0] + 0.5],
                ]
                goal_patch.set_xy(goal_vertices)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if mode == "human":
            return True

        width, height = self.fig.canvas.get_width_height()

        # Prefer RGBA buffer if available (backend-dependent).
        if hasattr(self.fig.canvas, "buffer_rgba"):
            frame = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape((height, width, 4))
            return frame[:, :, :3].copy()

        # Fallback for backends that only expose ARGB.
        frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape((height, width, 4))
        return frame[:, :, 1:4].copy()

    def close(self):
        plt.close()
