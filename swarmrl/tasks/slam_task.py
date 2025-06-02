import logging
import pickle
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class MappingTask(Task):
    """
    Args:
        Task (Task): Parent class
    """

    def __init__(
        self,
        mapper: nn.Module,
        model_path: str = "",
        lock=Lock(),
        range_pos=253,
        resolution=(64, 64),
    ):
        super().__init__(particle_type=0)
        self.lock = lock
        self.range_pos = range_pos
        self.resolution = resolution
        self.mapper = mapper
        self.init_mapper(model_path=model_path)

    def init_mapper(self, model_path: str = None):
        dummy_input = jax.random.normal(
            jax.random.PRNGKey(0), (1, self.resolution[0], self.resolution[1], 1)
        )
        params = self.mapper.init(jax.random.PRNGKey(0), dummy_input)
        self.model_state = TrainState.create(
            apply_fn=self.mapper.apply, params=params, tx=optax.adam(0.001)
        )
        if model_path:
            with open(model_path, "rb") as f:
                model_params = pickle.load(f)
            self.model_state = self.model_state.replace(params=model_params)

    def set_agent(self, agent):
        self.actor_critic_agent = agent

    def __call__(self, positions: np.ndarray) -> float:
        with self.lock:
            occupancy_map = self.actor_critic_agent.trajectory.occupancy_map[
                -1, ...
            ].copy()
            occupancy_map = jnp.array(occupancy_map)
        x, y = positions[:, :, 0], positions[:, :, 1]
        x = np.array(x / self.range_pos * occupancy_map.shape[0], dtype=np.int32)
        y = np.array(y / self.range_pos * occupancy_map.shape[1], dtype=np.int32)
        occupancy_map[x, y] += 1
        predicted_arena = self.mapper.apply(
            self.model_state.params, occupancy_map[jnp.newaxis, ..., jnp.newaxis]
        )
        predicted_arena = jnp.squeeze(predicted_arena)
        reward = jnp.linalg.norm(predicted_arena - 0.5)
        return reward
