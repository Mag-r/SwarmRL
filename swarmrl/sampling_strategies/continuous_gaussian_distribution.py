import logging
from abc import ABC

import jax
import jax.numpy as np
import numpy as onp
from jaxlib.xla_extension import XlaRuntimeError

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy

logger = logging.getLogger(__name__)


class ContinuousGaussianDistribution(SamplingStrategy, ABC):
    """
    Class for the continuous Gaussian distribution.
    """

    def __call__(
        self,
        logits: np.ndarray,
        action_dimension: int = 3,
        calculate_log_probs: bool = False,
        deployment_mode: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Generates an action and its corresponding log probability using a continuous Gaussian distribution.
        Args:
                logits (np.ndarray): The logits representing the parameters of the Gaussian distribution.
        Returns:
                tuple[MPIAction, float]: A tuple containing the generated action and its log probability.
        """

        assert (
            np.shape(logits)[1] == 2 * action_dimension
        ), f"Logits must have the shape (x, 2 * action_dimension). Has shape {np.shape(logits)}"
        rng = jax.random.PRNGKey(onp.random.randint(0, 1236534623))

        _, subkey = jax.random.split(rng)
        mean = logits[:, :action_dimension]
        if deployment_mode:
            action = mean
        else:
            epsilon = 1e-7
            cov = np.array(
                [
                    np.diag(logits[batch_index, action_dimension:])
                    + np.eye(action_dimension) * epsilon
                    for batch_index in range(logits.shape[0])
                ]
            )
            logger.debug(f"{cov=}")
            logger.debug(f"{mean=}")
            # assert (
            #     np.diag(cov) > 0
            # ).all(), f"Covariance matrix must be positive definite, {np.diag(cov)=}"
            try:
                action = jax.random.multivariate_normal(subkey, mean=mean, cov=cov)

            except XlaRuntimeError as e:
                logger.warning(f"Mean: {mean}, Cov: {cov}")
                raise e
            assert (
                not np.isnan(action).any() or not np.isinf(action).any()
            ), "Action values must not be NaN or Inf."

        if calculate_log_probs and not deployment_mode:
            log_probs = jax.scipy.stats.multivariate_normal.logpdf(
                action, mean=mean, cov=cov
            )
        else:
            log_probs = None
        action = action.at[:, 0].set(np.tanh(action.at[:, 0].get()))
        action = action.at[:, 1:].set(np.tanh(action.at[:, 1:].get()) / 2 + 0.5)

        return action, log_probs
