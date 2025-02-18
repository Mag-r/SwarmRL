"""
Module for the expected returns value function.
"""

import logging
from functools import partial

import jax
import jax.numpy as np

logger = logging.getLogger(__name__)


class TDReturnsSAC:
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

#     @partial(jax.jit, static_argnums=(0,))
    def __call__(self, rewards: np.ndarray, q_value: np.ndarray, temperature: float, next_log_probs: np.ndarray) -> np.ndarray:
        """Gives the expected returns for the SAC algorithm.

        Args:
        rewards (np.ndarray): Immediate rewards earned by the agent.
        q_value (np.ndarray): Predicted Q-value of the current state.
        temperature (float): Temperature parameter of the SAC algorithm (alpha).
        next_log_probs (np.ndarray): Log-probs of the next action.

        Returns:
        np.ndarray: Expected returns for the SAC algorithm.
        """
        logger.debug(f"{self.gamma=}")
        expected_returns = np.zeros_like(rewards)
        logger.debug(f"{rewards=}")
        expected_future_rewards = q_value - temperature * next_log_probs
        if np.shape(expected_future_rewards) != np.shape(expected_returns):
                expected_future_rewards = np.append(expected_future_rewards, 0)
        expected_returns = rewards + self.gamma * expected_future_rewards

        if self.standardize:
                mean_vector = np.mean(expected_returns, axis=0)
                std_vector = np.std(expected_returns, axis=0) + self.eps
                expected_returns = (expected_returns - mean_vector) / std_vector
        return expected_returns
