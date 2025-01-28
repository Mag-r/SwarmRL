"""
Random exploration module.
"""

from abc import ABC
from functools import partial

import jax
import jax.numpy as np

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy


class GlobalOUExploration(ExplorationPolicy, ABC):
    """
    Perform exploration by random moves.
    """

    def __init__(self, drift: float = 0.1, volatility: float = 0.1, long_term_mean: float = 0.0, action_dimension: int = 8):
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

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, model_actions: np.ndarray, seed
    ) -> np.ndarray:
        """
        Return an index associated with the chosen action.

        Parameters
        ----------
        model_actions : np.ndarray (n_colloids,)
                Action chosen by the model for each colloid.

        Returns
        -------
        action : np.ndarray
                Action chosen after the exploration module has operated for
                each colloid.
        """
        key = jax.random.PRNGKey(seed)
        value_range = np.max(model_actions, axis=-1) - np.min(model_actions, axis=-1)
        self.noise = self.drift * (self.long_term_mean - self.noise) + self.volatility * value_range * jax.random.normal(key, shape=self.noise.shape)
        actions = model_actions + self.noise
        return actions
