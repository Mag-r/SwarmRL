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
            action (jnp.ndarray): Shape (batch_size, action_dim)
        Returns:
            jnp.ndarray: Shape (batch_size, action_dim)
        """
        low = self.action_limits[:, 0]
        high = self.action_limits[:, 1]
        scale = (high - low) / 2.0
        mid = (high + low) / 2.0
        return jnp.tanh(action) * scale + mid


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
            subkey (jax.random.KeyArray): PRNG key for random sampling.
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
            pre_squash_action = mean
            log_probs = None
        else:
            epsilon = 1e-7
            log_std = logits[:, self.action_dimension:]
            log_std = jnp.clip(log_std, -20, 1)
            std = jnp.exp(log_std)

            pre_squash_action = jax.random.normal(subkey, shape=mean.shape) * std + mean

            if calculate_log_probs:
                log_probs = -0.5 * (((pre_squash_action - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
                log_probs = log_probs.sum(axis=-1)

                correction = (2*(jnp.log(2)-pre_squash_action-jax.nn.softplus(-2*pre_squash_action))).sum(axis=-1)
                log_probs = log_probs - correction
                
                low  = self.action_limits[:, 0]  # (action_dim,)
                high = self.action_limits[:, 1]  # (action_dim,)
                scale = (high - low) / 2         # (action_dim,)
                log_scale_sum = jnp.sum(jnp.log(scale))  # Skalar

                log_probs = log_probs - log_scale_sum
            else:
                log_probs = None
        action = (
            self.squash_action(pre_squash_action) if self.action_limits is not None else pre_squash_action
        )
        logger.debug(f"{action=}, {log_probs=}, with shape {action.shape}")
        return action, log_probs
