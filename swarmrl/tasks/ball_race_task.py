import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class BallRacingTask(Task):
    """This class implements the task of the ball racing experiment. The goal is to make the rafts and the ball move in a circle.
    The reward is given by the difference between the current tile and the previous tile. The tiles are numbered from 0 to 59, counting clockwise. Not moving the raft is penalized.

    Args:
        Task (Task): Parent class
    """

    def __init__(self):
        super().__init__(particle_type=0)
        self.previous_tile_com = None        
        self.previous_tile_ball = None
        self.moved_back = False
        self.time_on_current_tile = 0.0
        self.visited_tiles = set()
        self.previous_com = np.array([0, 0])

    def current_tile(self, position: np.ndarray) -> int:
        """Discretizes the position of the ball into a tile number between 0 and 59, counting clockwise.

        Args:
            position (np.ndarray): position to be discretized

        Returns:
            int: Number of the tile
        """
        x = position[0]
        y = position[1]
        
        central_point = jnp.array([126.5, 126.5])  
        vector = position - central_point
        angle = jnp.arctan2(vector[1], vector[0]) * (180 / jnp.pi)  
        if angle < 0:
            angle += 360 
        current_tile = int(angle/6)
        return current_tile

    def __call__(self, positions: np.ndarray) -> float:
        position_com = positions[:, -4, :]
        position_com = jnp.squeeze(position_com)
        current_tile_com = self.current_tile(position_com)
        current_tile_ball = self.current_tile(positions[:, -3, :].squeeze())
        self.visited_tiles.add(current_tile_com)
        
        if self.previous_tile_com is None:
            self.previous_tile_com = current_tile_com
            self.previous_tile_ball = current_tile_ball
            return 0.0
        reward = current_tile_com - self.previous_tile_com
        distance_ball = current_tile_ball - self.previous_tile_ball
        reward += distance_ball if distance_ball < 0 else distance_ball*1.1
        if reward > 30:
            reward = -1
        
        if reward <= -50 and len(self.visited_tiles) > 40:
            reward = 100 if not self.moved_back else 10
            self.moved_back = False
            self.visited_tiles = set()
        elif reward <= -50:
            reward = 1
            self.moved_back = False
            self.visited_tiles = set()
        self.moved_back = reward < 0 if not self.moved_back else self.moved_back # if true once it keeps true
        reward -= 0.1
        if self.previous_tile_com == current_tile_com:
            self.time_on_current_tile += 1
            if self.time_on_current_tile > 10:
                reward -= 10
                self.time_on_current_tile = 0
        else:
            self.time_on_current_tile = 0

        logger.info(f"Current tile: {current_tile_ball}, Previous tile: {self.previous_tile_ball}, Reward: {reward}, time on current tile: {self.time_on_current_tile}")
        logger.info(f"Current tile COM: {current_tile_com}, Previous tile COM: {self.previous_tile_com}"), 
        self.previous_tile_com = current_tile_com
        self.previous_tile_ball = current_tile_ball
        return reward