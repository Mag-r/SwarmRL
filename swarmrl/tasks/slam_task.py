import asyncio
import logging
import pickle
from threading import Lock
import threading
import os
import queue

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
import time

from swarmrl.tasks.task import Task
import matplotlib.pyplot as plt

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

        # Initialize the queue and thread for image saving
        self.image_queue = queue.Queue()
        self.image_saving_thread = threading.Thread(target=self._image_saving_worker, daemon=True)
        self.image_saving_thread.start()
        self.iterations = 0
        self.previous_certain_cells = 0
        self.previous_reward = 0.0

    def init_mapper(self, model_path: str = None):
        params = self.mapper.init(jax.random.PRNGKey(0), jnp.ones((1, 64,64, 1)))
        self.model_state = TrainState.create(
            apply_fn=jax.jit(self.mapper.apply), params=params["params"], tx=optax.adam(0.001)
        )
        with open(model_path, "rb") as f:
            model_params, opt_state = pickle.load(f)
        self.model_state = self.model_state.replace(params=model_params, opt_state=opt_state)
        logger.info(f"OccupancyGridModel parameters from {model_path}")

    def set_agent(self, agent):
        self.actor_critic_agent = agent

    def _image_saving_worker(self, folder="images"):
        """Worker thread to save images from the queue."""
        os.makedirs(folder, exist_ok=True)
        while True:
            if not self.image_queue.empty():
                occupancy_map = self.image_queue.get()
                file_path = os.path.join(folder, f"predicted_arena_{self.iterations:04d}.png")
                plt.imsave(file_path, occupancy_map, cmap='gray')



    def __call__(self, positions: np.ndarray) -> float:
        start = time.time()
        self.iterations += 1
        if self.iterations % 3 == 0 and self.iterations > 0:
            with self.lock:
                occupancy_map = jnp.array(self.actor_critic_agent.trajectory.occupancy_map[-1]).squeeze()
            x, y = positions[0, :, 0], positions[0, :, 1]
            x = np.array(x / self.range_pos * occupancy_map.shape[0], dtype=np.int32)
            y = np.array(y / self.range_pos * occupancy_map.shape[1], dtype=np.int32)
            occupancy_map.at[x, y].add(1)
            occupancy_map = jnp.clip(occupancy_map, 0, 1000)

            predicted_arena = self.mapper.apply(
                {"params": self.model_state.params}, occupancy_map[jnp.newaxis, ..., jnp.newaxis]
            )
            predicted_arena = jnp.squeeze(predicted_arena)
            self.image_queue.put(predicted_arena)
            reward = jnp.linalg.norm(predicted_arena - 0.75)
            number_certain_cells = jnp.sum(jnp.abs(predicted_arena - 0.5)>0.45)/64.0
            logger.info(f"Certain cells: {number_certain_cells/0.64}%, difference to previous: {(number_certain_cells - self.previous_certain_cells)*64}")
            reward += number_certain_cells
            reward, self.previous_reward = reward - self.previous_reward, reward
            self.previous_certain_cells = number_certain_cells
            logger.info(f"Reward: {reward}, time taken: {time.time() - start}")
            return reward
        return 0.0

    def stop_image_saving_thread(self):
        """Stop the image saving thread."""
        self.image_queue.put((None, None))  # Send exit signal
        self.image_saving_thread.join()
