import numpy as np
from jax import numpy as jnp
import logging

from swarmrl.engine.engine import Engine
from swarmrl.force_functions.global_force_fn import GlobalForceFunction
import gymnasium as gym
from matplotlib import pyplot as plt
import os
import glob

logger = logging.getLogger(__name__)


class CarBenchmark(Engine):
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        )
        self.colloids = None

    def integrate(self, n_slices: int, force_model: GlobalForceFunction):
        """Perform a real-experiment equivalent of an integration step."""
        # Clear the benchmark_images folder
        images_path = "benchmark_images"
        files = glob.glob(os.path.join(images_path, "*.png"))
        for f in files:
            os.remove(f)
        obs, _ = self.env.reset()
        for _ in range(40):
            self.env.step(np.array([0, 1, 0]))
        obs, *_ = self.env.step(np.array([0, 1, 0]))
        reward_sum = 0
        for i in range(n_slices):
            action = np.squeeze(np.array(force_model.calc_action(obs)))
            reward = 0
            for _ in range(3):
                _, r, *_, terminated = self.env.step(action)
                if terminated:
                    break
                reward += r
            # action = self.env.action_space.sample()
            obs, r, *_, terminated = self.env.step(action)
            if terminated:
                break
            plt.imshow(obs)
            plt.savefig(f"benchmark_images/car{i:03d}.png")
            reward += r
            reward_sum += reward
            force_model.calc_reward(self.colloids, reward)
        logger.info(f"last action {action}")
        logger.info(f"Total reward: {reward_sum}")
