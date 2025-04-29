import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class BallRacingTask(Task):

    def __init__(self):
        super().__init__(particle_type=0)
        self.previous_tile = None        
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
        position_ball = jnp.mean(positions[:,:-2,:], axis=1)
        position_ball = jnp.squeeze(position_ball)
        current_tile = self.current_tile(position_ball)
        if self.previous_tile is None:
            self.previous_tile = current_tile
            return 0.0
        reward = current_tile - self.previous_tile
        if reward <= -60:
            reward = 100 if not self.moved_back else 1
            self.moved_back = False
        self.moved_back = reward < 0
        if self.previous_tile == current_tile:
            self.time_on_current_tile += 1
            if self.time_on_current_tile > 10:
                reward = -10
                self.time_on_current_tile = 0
        else:
            self.time_on_current_tile = 0
        logger.info(f"Current tile: {current_tile}, Previous tile: {self.previous_tile}, Reward: {reward}, time on current tile: {self.time_on_current_tile}")
        self.previous_tile = current_tile
        return reward