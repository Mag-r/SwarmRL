"""
Random exploration module.
"""

from abc import ABC
from functools import partial

import jax
import jax.numpy as np
import logging

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy

logger = logging.getLogger(__name__)
class GlobalOUExploration(ExplorationPolicy, ABC):
    """
    Perform exploration by random moves.
    """

    def __init__(self, action_limits, drift: float = 0.1, volatility: float = 0.1, long_term_mean: float = 0.0, action_dimension: int = 3, epsilon: float = 1.0):
        """
        Constructor for the random exploration module.

        Parameters
        ----------
        drift : float
                Drift of the OU process.
        volatility : float
                Volatility of the OU process.
        long_term_mean : float
                Long term mean of the OU process.
            
        """
        self.drift = drift
        self.volatility = volatility
        self.long_term_mean = long_term_mean
        self.action_dimension = action_dimension
        self.noise = np.zeros(action_dimension)
        self.action_limits = action_limits
        self.epsilon = 1.0

    def reduce_randomness(self, decay: float = 0.95) -> None:
        self.noise *= decay
        self.epsilon *= decay
        self.epsilon = max(self.epsilon, 0.01)
        


#     @partial(jax.jit, static_argnums=(0,))
    def __call__(self, model_actions: np.ndarray, rng_key) -> np.ndarray:
        """
        Add OU noise to the model actions.
        """
        key_normal, key_uniform = jax.random.split(rng_key)

        scale = self.action_limits[:, 1] - self.action_limits[:, 0]
        long_term_noise_shift = self.drift * (self.long_term_mean - self.noise)
        random_noise = self.volatility * scale* jax.random.normal(key_normal, shape=self.noise.shape)

        mask = (jax.random.uniform(key_uniform, shape=self.noise.shape) < self.epsilon).astype(np.float32)

        self.noise += long_term_noise_shift + random_noise 
        self.noise = self.noise.reshape(1, self.action_dimension) * mask

        actions = model_actions + self.noise
        actions = np.clip(actions, self.action_limits[:, 0], self.action_limits[:, 1])
        logger.info(f"noise: {self.noise}")
        return actions
