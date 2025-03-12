import numpy as np
from jax import numpy as jnp
import logging
import jax
from functools import partial
from swarmrl.engine.engine import Engine
from swarmrl.force_functions.global_force_fn import GlobalForceFunction


logger = logging.getLogger(__name__)


class SimpleBenchmark(Engine):
    def __init__(self):
        super().__init__()

        self.colloids = None

    def integrate(self, n_slices: int, force_model: GlobalForceFunction):
        """Perform a real-experiment equivalent of an integration step."""

        reward_sum = 0
        previous_obs = float(np.random.randint(0, 2))
        for _ in range(n_slices):
            obs = float(np.random.randint(0, 2))
            action = np.array(force_model.calc_action(obs))

            reward = -1 * (action[0] - obs) ** 2 - (action[1] - previous_obs) ** 2
            previous_obs = obs
 
            reward_sum += reward

            force_model.calc_reward(self.colloids, reward)
        logger.info(f"last action {action}")
        logger.info(f"obs: {obs}, previous_obs: {previous_obs}")
        logger.info(f"Total reward: {reward_sum}")
