from __future__ import annotations

from minigrid.minigrid_env import MiniGridEnv
from typing import Any, SupportsFloat
from gymnasium.core import ActType, ObsType

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class CustomDoorKey(MiniGridEnv):
    def __init__(self, size=8, max_steps: int | None = None, intermediate_reward = True, randomize_goal = False, key_discount = 0.1, door_discount = 0.5, **kwargs):
        if max_steps is None:
            max_steps = 1000

        self.randomize_goal = randomize_goal
        self.intermediate_reward = intermediate_reward

        self.opened_door = False
        self.obtained_key = False
        
        # Discount for intermediate rewards
        self.key_discount = key_discount
        self.door_discount = door_discount

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

        vector_pos = spaces.Box(
            low=1,
            high=size-1,
            shape=(2,),
            dtype="uint8",
        )

        vector_info = spaces.Box(
            low=0,
            high=1,
            shape=(6,),
            dtype="uint8"
        )

        self.observation_space = spaces.Dict(
            {
                **self.observation_space,
                "vector_pos": vector_pos,
                "vector_info": vector_info
            }
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        # Place anywhere in last column if randomize_goal is on
        if self.randomize_goal:
            new_height = self._rand_int(1, width - 1)
            self.put_obj(Goal(), width - 2, new_height)
        else:
            self.put_obj(Goal(), width - 2, width-2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

    def _generate_obs_dict(self, observation):
        vector_pos = []
        vector_info = []
        
        # agent position
        x, y = self.agent_pos
        vector_pos += [x,y]
        
        # one hot encoding of agent direction
        agent_dir = [0]*4
        agent_dir[self.agent_dir] = 1
        vector_info += agent_dir

        # carrying
        vector_info += [int(self.obtained_key)]

        # door opened
        vector_info += [int(self.opened_door)]

        res = {
            **observation,
            "vector_pos": vector_pos,
            "vector_info": vector_info
        }
        
        return res
        
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        obs, _ = super().reset(seed=seed)

        self.opened_door = False
        self.obtained_key = False
        
        res = self._generate_obs_dict(obs)
        return res, {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    if isinstance(fwd_cell, Key) and not self.obtained_key and self.intermediate_reward:
                        self.obtained_key = True  # Flag for key pickup
                        reward = self.key_discount * self._reward()  # Reward for picking up the key

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell and isinstance(fwd_cell, Door) and self.carrying and isinstance(self.carrying, Key):
                fwd_cell.toggle(self, fwd_pos)
                if not self.opened_door and self.intermediate_reward:
                    self.opened_door = True  # Flag for door opening
                    reward = self.door_discount * self._reward()  # Reward for opening the door

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        res = self._generate_obs_dict(obs)
        return res, reward, terminated, truncated, {}
        