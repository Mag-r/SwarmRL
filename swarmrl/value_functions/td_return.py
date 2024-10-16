"""
Module for the expected returns value function.
"""

import logging
from functools import partial

import jax
import jax.numpy as np

logger = logging.getLogger(__name__)


class TDReturns:
    """
    Class for the expected returns.
    """

    def __init__(self, gamma: float = 0.99, standardize: bool = True):
        """
        Constructor for the Expected returns class

        Parameters
        ----------
        gamma : float
                A decay factor for the values of the task each time step.
        standardize : bool
                If True, standardize the results of the calculation.

        Notes
        -----
        See https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
        for more information.
        """
        self.gamma = gamma
        self.standardize = standardize

        # Set by us to stabilize division operations.
        self.eps = np.finfo(np.float32).eps.item()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, rewards: np.ndarray, expected_values: np.ndarray) -> np.ndarray:
        """
        Call function for the expected returns.
        Parameters
        ----------
        rewards : np.ndarray (n_time_steps, n_particles, dimension)
                A numpy array of rewards to use in the calculation.

        Returns
        -------
        expected_returns : np.ndarray (n_time_steps, n_particles)
                Expected returns for the rewards.
        """
        expected_values = expected_values.copy()
        expected_values = np.array(expected_values)
        expected_values = np.append(expected_values[1:],0)
        logger.debug(f"{self.gamma=}")
        expected_returns = rewards.copy()
        logger.debug(rewards)
        
        expected_returns = expected_returns + self.gamma * expected_values

        logger.debug(f"{expected_returns=}")

        if self.standardize:
            mean_vector = np.mean(expected_returns, axis=0)
            std_vector = np.std(expected_returns, axis=0) + self.eps

            expected_returns = (expected_returns - mean_vector) / std_vector

        return expected_returns
