import jax.numpy as np
from jax import random
import logging

from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)
class DummyTask(Task):
    
    def __init__(self, box_length: np.ndarray, target: np.ndarray):
        super().__init__(particle_type=0)
        self.box_length = box_length
        self.target = target
        key = random.PRNGKey(0)
        random_pos = random.uniform(key, (10000,3), minval=0, maxval=box_length)
        self.normalization_distance = np.linalg.norm(random_pos - self.target.reshape(1,3),axis=-1).mean()
        logger.debug(f"{self.normalization_distance=}")
        self.old_distance = None
        
    def get_normalization(self):
        return self.normalization_distance
    
    def distance_reward(self, colloids: list) -> float:
        """     

        Args:
            colloids (list): _description_

        Returns:
            float: Reward, minimum expected value is -1, max is 0 
        """
        distance_target = np.linalg.norm(np.array([raft.pos[:2] - self.target[:2] for raft in colloids]),axis=-1)
        if self.old_distance is None:
            self.old_distance = distance_target.mean()
            logger.debug(f"old_distance initialized to {self.old_distance}")
        reward = self.old_distance - distance_target.mean()
        logger.debug(f"{reward=}, old_distance={self.old_distance}, new_distance={distance_target.mean()}")
        self.old_distance = distance_target.mean()
        return reward
    
    def spinning_reward(self, colloids: list) -> float:
        """     
        Args:s
            colloids (list): _description_

        Returns:
            float: Reward, minimum expected value is -1, max is 0 
        """
        mean_squared_spin = np.sqrt(np.mean(np.array([raft.rotational_velocity**2 for raft in colloids])))**2

        return mean_squared_spin/(2*np.pi)
    
    def __call__(self, colloids: list) -> float:
        return self.distance_reward(colloids) 