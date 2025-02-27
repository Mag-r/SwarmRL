import logging
from abc import ABC

import jax
import jax.numpy as np
from time import time

from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy

logger = logging.getLogger(__name__)


class ExpertKnowledge(SamplingStrategy, ABC):
    """
    Class to integrate expert knowledge into the action generation process, by selecting the actions by hand.
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
        mean = logits[:, :action_dimension]
        epsilon = 1e-7
        cov = np.array(
            [
                np.diag(logits[batch_index, action_dimension:])
                + np.eye(action_dimension) * epsilon
                for batch_index in range(logits.shape[0])
            ]
        )
        action = np.array([0.0, 0.0, 1.0])
        action = action.repeat(logits.shape[0], axis=0)
        action = action.reshape(logits.shape[0], action_dimension)
        if calculate_log_probs and not deployment_mode:
            log_probs = jax.scipy.stats.multivariate_normal.logpdf(
                action, mean=mean, cov=cov
            )
        else:
            log_probs = None
        return action, log_probs
