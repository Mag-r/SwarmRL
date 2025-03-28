import jax.numpy as jnp
import numpy as np
import logging
from swarmrl.tasks.task import Task
import scipy as sc

logger = logging.getLogger(__name__)


class ExperimentHexagonTask(Task):

    def __init__(self):
        super().__init__(particle_type=0)

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

    def __call__(self, positions: np.ndarray) -> float:
        triangulation = self.delaunay_triangulation(positions)
        central_colloid = self.find_central_colloid(triangulation)

        reward = np.abs(self.phi_6(positions, central_colloid))
        return reward
