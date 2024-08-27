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
    """

    def __call__(self, logits: np.ndarray, number_of_gaussians, action_dimension=8) -> MPIAction:
        """
        Takes a sample from multiple multivariate gaussians distributions. 
        The first number_of_guassians * action_dimension logits define the mean, 
        the remaining number_of_gaussian * action_dimension the cov.

        Args:
                logits (np.ndarray): The logits used to generate the action.
                number_of_gaussians: The number of Gaussians in the distribution.
                action_dimension (int): The dimension of the action.

        Returns:
                MPIAction: The generated action.

        Raises:
                AssertionError: If the shape of the logits is not (number_of_gaussians*action_dimension*2,).
                AssertionError: If the covariance matrix is not positive semidefinite.
        """

        assert np.shape(logits) == (number_of_gaussians*action_dimension*2,), f"Logits must have the shape ({number_of_gaussians*action_dimension*2},), have {np.shape(logits)}"
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))
        
        key, subkey = jax.random.split(rng)
        selected_gaussian = jax.random.randint(subkey, minval = 0, maxval = number_of_gaussians, shape=(1,))[0]
        mean = logits[selected_gaussian*action_dimension:(selected_gaussian+1)*action_dimension]
        cov = np.diag(logits[selected_gaussian*action_dimension:(selected_gaussian+1)*action_dimension])
        
        epsilon = 1e-6
        cov = cov + epsilon * np.eye(cov.shape[0])
        logger.debug(f"{cov=}")
        logger.debug(f"{mean=}")
        # assert np.prod(cov)>0, f"Covariance matrix must be positive definite but is {np.diag(cov)}, with determinant {np.prod(cov)}"
        action_values = jax.random.multivariate_normal(subkey, mean=mean,cov=cov)
        assert not np.isnan(action_values).any(), "Action values must not be NaN"
        log_probs = jax.scipy.stats.multivariate_normal.logpdf(action_values, mean=mean, cov=cov)
        action_values = action_values.reshape((int(action_dimension/2),-1))
        action = MPIAction(*action_values)
        return action, log_probs