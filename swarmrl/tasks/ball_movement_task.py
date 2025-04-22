import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class ExperimentBallMovingTask(Task):

    def __init__(self):
        super().__init__(particle_type=0)
        self.power = 2
        self.normalization = jnp.linalg.norm(np.array([253/2, 253/2]), self.power)
        self.old_distance = 0


    def distance_reward(self, position_ball: np.ndarray) -> float:
        return jnp.linalg.norm(position_ball, self.power)


    def __call__(self, positions: np.ndarray) -> float:
        position_ball = positions[:, -1, :]
        position_ball = jnp.squeeze(position_ball)
        distance_reward = np.mean(self.distance_reward(position_ball))/self.normalization
        if self.old_distance == 0:
            self.old_distance = distance_reward
            return 0.0
        reward = self.old_distance - distance_reward
        self.old_distance = distance_reward
        reward = - distance_reward
        return reward
