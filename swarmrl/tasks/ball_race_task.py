import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class BallRacingTask(Task):

    def __init__(self):
        super().__init__(particle_type=0)
        self.previous_tile_com = None        
        self.previous_tile_ball = None
        self.moved_back = False
        self.time_on_current_tile = 0.0

    def current_tile(self, position_ball: np.ndarray) -> int:
        x = position_ball[0]
        y = position_ball[1]
        
        central_point = jnp.array([126.5, 126.5])  
        vector = position_ball - central_point
        angle = jnp.arctan2(vector[1], vector[0]) * (180 / jnp.pi)  
        if angle < 0:
            angle += 360 
        current_tile = int(angle/6)
        return current_tile

    def __call__(self, positions: np.ndarray) -> float:
        # position_ball = positions[:, -1, :]
        position_com = jnp.mean(positions[:,:-2,:], axis=1)
        position_com = jnp.squeeze(position_com)
        current_tile_com = self.current_tile(position_com)
        current_tile_ball = self.current_tile(positions[:, -2, :].squeeze())
        if self.previous_tile_com is None:
            self.previous_tile_com = current_tile_com
            self.previous_tile_ball = current_tile_ball
            return 0.0
        reward = current_tile_com - self.previous_tile_com
        if reward <= -50:
            reward = 100 if not self.moved_back else 1
            self.moved_back = False
        self.moved_back = reward < 0
        if self.previous_tile_com == current_tile_com or self.previous_tile_ball == current_tile_ball:
            self.time_on_current_tile += 1
            if self.time_on_current_tile > 10:
                reward -= 10
                self.time_on_current_tile = 0
        else:
            self.time_on_current_tile = 0
        # reward += (current_tile_ball - self.previous_tile_ball)*2
        logger.info(f"Current tile: {current_tile_ball}, Previous tile: {self.previous_tile_ball}, Reward: {reward}, time on current tile: {self.time_on_current_tile}")
        self.previous_tile_com = current_tile_com
        self.previous_tile_ball = current_tile_ball
        return reward