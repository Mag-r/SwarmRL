import jax.numpy as np
import logging

from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class ChainTask(Task):

    def __init__(self, box_length: np.ndarray):
        super().__init__(particle_type=0)
        self.box_length = box_length
        self.old_residual = None


    def orthogonal_regression_svd(self, x, y):
        data = np.column_stack((x, y))

        mean = np.mean(data, axis=0)
        centered_data = data - mean

        _, _, Vt = np.linalg.svd(centered_data)

        direction = Vt[0]  # Erste Zeile von Vt entspricht der Hauptachse

        if np.abs(direction[0]) < 1e-10:
            slope = np.inf
            intercept = np.nan  
        else:
            slope = direction[1] / direction[0]
            intercept = mean[1] - slope * mean[0]

        distances = np.abs(
            direction[1] * (x - mean[0]) - direction[0] * (y - mean[1])
        ) / np.linalg.norm(direction)

        rmse = np.sqrt(np.mean(distances**2))

        return slope, intercept, rmse

    def regression_error(self, colloids: list) -> float:
        """
        Performs regression on position of colloids and returns the error.

        Args:
            colloids (list): Particle in the system

        Returns:
            float: MSE of regression
        """
        positions = np.array([colloid.pos for colloid in colloids])
        x = positions[:, 0]
        y = positions[:, 1]
        *_, residual = self.orthogonal_regression_svd(x, y)
        return residual


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
        regression_error = self.regression_error(colloids)
        if self.old_residual is None:
            self.old_residual = regression_error
            return 0
        reward  = regression_error - self.old_residual
        self.old_residual = regression_error
        return -reward  #- self.spinning_reward(colloids)
