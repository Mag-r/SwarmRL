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

    def __init__(self, action_limits, drift: float = 0.1, volatility: float = 0.1, long_term_mean: float = 0.0, action_dimension: int = 3):
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

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, model_actions: np.ndarray, seed=42
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

        self.noise += self.drift * (self.long_term_mean - self.noise) + self.volatility * value_range * jax.random.normal(key, shape=self.noise.shape)
        self.noise = self.noise.reshape(1, self.action_dimension)
        actions = model_actions + self.noise
        # Clip the actions to be within the action limits
        actions = np.clip(actions, self.action_limits[:,0], self.action_limits[:,1])
        return actions
