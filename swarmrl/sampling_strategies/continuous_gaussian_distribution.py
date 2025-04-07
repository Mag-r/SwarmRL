import logging
from abc import ABC

import jax
import jax.numpy as jnp
import numpy as onp
from jaxlib.xla_extension import XlaRuntimeError
from functools import partial

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy

logger = logging.getLogger(__name__)


class ContinuousGaussianDistribution(SamplingStrategy, ABC):
    """
    Class for the continuous Gaussian distribution.
    """

    def __init__(self, action_dimension: int = 3, action_limits: jnp.ndarray = None):
        """
        Initializes the continuous Gaussian distribution.
        Args:
                self.action_dimension (int): The dimension of the action space.
                transform_fn (function): A function to transform the action and log probability.
        """
        self.action_dimension = action_dimension
        self.action_limits = action_limits
        # if self.action_limits:
        #     assert jnp.shape(self.action_limits) == (self.action_dimension, 2), f"Action limits must have shape ({self.action_dimension}, 2). Has shape {jnp.shape(self.action_limits)}"

    def squash_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Squashes the action to the range indicated by action_limits using tanh.
        Args:
            action (jnp.ndarray): The action to squash.
        Returns:
            jnp.ndarray: The squashed action.
        """
        squash_function = lambda x: self.action_limits[:, 0] + (
            jnp.tanh(x) + 1
        ) / 2.0 * (self.action_limits[:, 1] - self.action_limits[:, 0])
        action = jax.vmap(squash_function)(action)
        return action

    @partial(
        jax.jit, static_argnames=["self", "deployment_mode", "calculate_log_probs"]
    )
    def __call__(
        self,
        logits: jnp.ndarray,
        subkey,  # dtype ?
        calculate_log_probs: bool = False,
        deployment_mode: bool = False,
    ) -> tuple[jnp.ndarray, float]:
        """
        Generates an action and its corresponding log probability using a continuous Gaussian distribution.
        Args:
            logits (jnp.ndarray): The logits representing the parameters of the Gaussian distribution.
            rng (jax.random.KeyArray): PRNG key for random sampling.
            calculate_log_probs (bool): Whether to calculate log probabilities.
            deployment_mode (bool): Whether to return deterministic actions.
        Returns:
            tuple[jnp.ndarray, float]: The generated action and its log probability.
        """
        assert (
            logits.shape[1] == 2 * self.action_dimension
        ), f"Logits must have the shape (x, 2 * self.action_dimension). Has shape {logits.shape}"

        mean = logits[:, : self.action_dimension]
        if deployment_mode:
            action = mean
            log_probs = None
        else:
            epsilon = 1e-7
            diag_cov = jnp.exp(logits[:, self.action_dimension :]) * (self.action_limits[:,1] - self.action_limits[:,0])/1000.0 + epsilon
            assert diag_cov.shape == logits[:, self.action_dimension :].shape, f"Diagonal covariance matrix must have the same shape as the logits. Has shape {diag_cov.shape}"
            cov_matrices = jnp.vectorize(lambda d: jnp.diag(d), signature="(n)->(n,n)")(
                diag_cov
            )

            action = jax.random.multivariate_normal(subkey, mean=mean, cov=cov_matrices)
            if calculate_log_probs:
                log_probs = jax.scipy.stats.multivariate_normal.logpdf(
                    action, mean=mean, cov=cov_matrices
                )
                if self.action_limits is not None:
                    correction = 2 * (
                        jnp.log(2) - action - jax.nn.softplus(-2 * action)
                    )
                    correction = correction.sum(axis=-1)
                    log_probs = log_probs - correction
            else:
                log_probs = None
        action = (
            self.squash_action(action) if self.action_limits is not None else action
        )
        logger.debug(f"mean covariance {jnp.mean(cov_matrices)}")
        logger.debug(f"{action=}, {log_probs=}, with shape {action.shape}")
        logger.debug(f"{cov_matrices=}")
        return action, log_probs
