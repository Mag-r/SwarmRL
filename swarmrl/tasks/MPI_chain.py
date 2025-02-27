import jax.numpy as np
import numpy as onp
import logging
import scipy as sc
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class ChainTask(Task):

    def __init__(self, box_length=10000):
        super().__init__(particle_type=0)
        self.old_residual = None
        self.box_length = box_length
        self.angle_normalization = None
        self.regression_normalization = None
        self.old_angle_error = None

    def orthogonal_regression_svd(self, x, y):
        data = np.column_stack((x, y))

        mean = np.mean(data, axis=0)
        centered_data = data - mean

        _, _, Vt = np.linalg.svd(centered_data)

        direction = Vt[0]  

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

    def spinning_reward(self, colloids: list) -> float:
        """
        Args:s
            colloids (list): _description_

        Returns:
            float: Reward, minimum expected value is -1, max is 0
        """
        mean_squared_spin = (
            np.sqrt(
                np.mean(np.array([raft.rotational_velocity**2 for raft in colloids]))
            )
            ** 2
        )
        return mean_squared_spin / (2 * np.pi)

    def angle_between_particles(self, x, y) -> float:
        """
        Calculate the angle between two particles and the x-axis and takes the std of all pairs.
        Minimize this.

        Args:
            colloids (list): Particle in the system

        Returns:
            float: Angle between the particles
        """
        idx = np.triu_indices(len(x), k=1)
        dx = x[idx[0]] - x[idx[1]]
        dy = y[idx[0]] - y[idx[1]]
        angles = np.arctan2(dy, dx)
        angle = np.array(angles)
        std_angle = sc.stats.circstd(angle)
        return std_angle
    
    def get_normalization(self):
        return self.angle_normalization, self.regression_normalization
    
    def __call__(self, colloids: list) -> float:
        if self.angle_normalization is None and self.regression_normalization is None:
            # use mean of random positions as normaliation
            self.angle_normalization = 0
            self.regression_normalization = 0
            for _ in range(1000):
                x = onp.random.rand(len(colloids), 2) * self.box_length
                y = onp.random.rand(len(colloids), 2) * self.box_length
                *_, regression_error = self.orthogonal_regression_svd(x, y)
                angle_error = self.angle_between_particles(x, y)
                self.angle_normalization += angle_error
                self.regression_normalization += regression_error
            self.angle_normalization /= 1000
            self.regression_normalization /= 1000
            logger.info(f"Normalization factors: {self.angle_normalization=}, {self.regression_normalization=}")
        positions = np.array([colloid.pos for colloid in colloids])
        x = positions[:, 0]
        y = positions[:, 1]
        *_, regression_error = self.orthogonal_regression_svd(x, y)
        regression_error /= self.regression_normalization
        angle_error = self.angle_between_particles(x, y) / self.angle_normalization
        if self.old_residual is None and self.old_angle_error is None:
            self.old_residual = regression_error
            self.old_angle_error = angle_error
            return 0
        reward_regression = regression_error# - self.old_residual
        reward_angle = angle_error #- self.old_angle_error
        self.old_residual = regression_error
        self.old_angle_error = angle_error
        return -(reward_regression + reward_angle)
