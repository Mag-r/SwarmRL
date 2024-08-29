from abc import ABC
import logging

import jax
import jax.numpy as np
import numpy as onp
from dataclasses import fields

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
from swarmrl.actions import MPIAction

logger = logging.getLogger(__name__)


class ContinuousGaussianDistribution(SamplingStrategy, ABC):
    """
    Class for the continuous Gaussian distribution.
    TODO: make shorter indentation
    """

    def __call__(
        self, logits: np.ndarray, number_of_gaussians: int, action_dimension: int = 8
    ) -> tuple[MPIAction, float]:
        """
        Generates an action and its corresponding log probability using a continuous Gaussian distribution.
        Args:
                logits (np.ndarray): The logits representing the parameters of the Gaussian distribution.
                number_of_gaussians (int): The number of Gaussians in the distribution.
                action_dimension (int, optional): The dimensionality of the action. Defaults to 8.
        Returns:
                tuple[MPIAction, float]: A tuple containing the generated action and its log probability.
        """

        assert np.shape(logits) == (
            number_of_gaussians * action_dimension * 2,
        ), f"Logits must have the shape ({number_of_gaussians*action_dimension*2},), have {np.shape(logits)}"
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))

        key, subkey = jax.random.split(rng)
        selected_gaussian = jax.random.randint(
            subkey, minval=0, maxval=number_of_gaussians, shape=(1,)
        )[0]
        mean = logits[
            selected_gaussian
            * action_dimension : (selected_gaussian + 1)
            * action_dimension
        ]
        cov = np.diag(
            logits[
                selected_gaussian
                * action_dimension + number_of_gaussians * action_dimension: (selected_gaussian + 1)
                * action_dimension + number_of_gaussians * action_dimension
            ]
        )

        epsilon = 1e-6
        cov = cov + epsilon * np.eye(cov.shape[0])
        logger.debug(f"{cov=}")
        logger.debug(f"{mean=}")
        assert (np.diag(cov)>0).all(), f"Covariance matrix must be positive definite, {np.diag(cov)=}"
        action_values = jax.random.multivariate_normal(subkey, mean=mean, cov=cov)
        assert not np.isnan(
            action_values
        ).any(), f"Action values must not be NaN."
        log_probs = jax.scipy.stats.multivariate_normal.logpdf(
            action_values, mean=mean, cov=cov
        )
        action_values = action_values.reshape((int(action_dimension / 2), -1))
        action = MPIAction(*action_values)
        return action, log_probs
