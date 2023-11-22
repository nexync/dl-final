#%%
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import random


#%%
class myenv(MiniGridEnv):
    def __init__(
        self,
        size=5, 
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,):
        
        self.grid_size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set to False for a more realistic environment
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Find the key to open the door and reach the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomly place the door
        door_color = COLOR_NAMES[0]
        door_pos = self._random_position(exclude=[(1, 1)])  # Exclude agent's start position
        self.grid.set(*door_pos, Door(door_color, is_locked=True))

        # Randomly place the key
        key_pos = self._random_position(exclude=[(1, 1), door_pos])  # Exclude agent's start position and door position
        self.grid.set(*key_pos, Key(door_color))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Find the key to open the door and reach the goal"
    
    def _random_position(self, exclude=[]):
        while True:
            pos = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
            if pos not in exclude and self.grid.get(*pos) is None:
                return pos

def main():
    env = myenv(render_mode="human")

    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    # obs = env.reset()
    # for _ in range(10):  # Just run a few steps for testing
    #     action = env.action_space.sample()  # Random action
    #     obs, reward, done, info = env.step(action)
    #     env.render()  # Make sure to render the environment
    #     if done:
    #         break

if __name__ == "__main__":
    main()
# %%
