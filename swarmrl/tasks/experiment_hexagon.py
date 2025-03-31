import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task
import scipy as sc

logger = logging.getLogger(__name__)


class ExperimentHexagonTask(Task):

    def __init__(self, number_particles: int):
        super().__init__(particle_type=0)
        if number_particles != 7:
            raise ValueError("Number of particles must be 7 for this task.")
        self.number_particles = number_particles

    def delaunay_triangulation(self, positions: np.ndarray) -> np.ndarray:
        triangulation = sc.spatial.Delaunay(positions)
        return triangulation

    def find_central_colloid(self, triangulation: np.ndarray) -> np.ndarray:
        all_edges = np.array(triangulation.simplices).flatten()
        central_colloid = np.argmax(np.bincount(all_edges))
        return central_colloid

    def phi_6(self, positions: np.ndarray, central_colloid: np.ndarray) -> np.ndarray:
        phi_6_list = np.exp(
            1j
            * 6
            * np.angle(
                positions[central_colloid, 0]
                - positions[:, 0]
                + 1j * (positions[central_colloid, 1] - positions[:, 1])
            )
        )
        return phi_6_list.sum() / (len(phi_6_list) - 1)
    
    def distance_central(self, positions: np.ndarray, central_colloid, np.ndarray) -> float:
        distance = np.linalg.norm(
            positions[central_colloid, :] - positions[central_colloid, :]
        )
        return distance

    def __call__(self, positions: np.ndarray) -> float:
        # logger.info(f"Positions: {positions}, with shape {positions.shape}")
        positions = positions.reshape((self.number_particles, 2))
        triangulation = self.delaunay_triangulation(positions)
        central_colloid = self.find_central_colloid(triangulation)
        distance = self.distance_central(positions, central_colloid)
        distance_reward = - np.linalg.norm(distance - 10)
        phi6_reward = np.abs(self.phi_6(positions, central_colloid))
        logger.info(f"distance reward: {distance_reward}, distance: {distance}, phi6: {phi6_reward}")
        reward = phi6_reward - distance_reward
        return reward
