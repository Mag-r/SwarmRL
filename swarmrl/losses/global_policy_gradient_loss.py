"""
Module for the implementation of policy gradient loss.

Policy gradient is the most simplistic loss function where critic loss drives the entire
policy learning.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""

import logging
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.td_return import TDReturns

logger = logging.getLogger(__name__)


class GlobalPolicyGradientLoss(Loss):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(self, value_function: TDReturns = TDReturns()):
        """
        Constructor for the reward class.

        Parameters
        ----------
        value_function : ExpectedReturns
        """
        super(Loss, self).__init__()
        self.value_function = value_function
        self.n_particles = None
        self.n_time_steps = None

    def _calculate_loss(
        self,
        network_params: FrozenDict,
        network: Network,
        feature_data: jnp.ndarray,
        carry: jnp.ndarray,
        rewards: jnp.ndarray,
        log_probs: jnp.ndarray,
    ) -> jnp.array:
        """
        Compute the loss of the shared actor-critic network.

        Parameters
        ----------
        network : FlaxModel
            The actor-critic network that approximates the policy.
        network_params : FrozenDict
            Parameters of the actor-critic model used.
        feature_data : np.ndarray (n_time_steps,  feature_dimension)
            Observable data for each time step and particle within the episode.
        action_indices : np.ndarray (n_time_steps)
            The actions taken by the policy for all time steps and particles during one
            episode.
        rewards : np.ndarray (n_time_steps)
            The rewards received for all time steps and particles during one episode.


        Returns
        -------
        loss : float
            The loss of the actor-critic network for the last episode.
        """

        # (n_timesteps, n_possibilities)
        logits, predicted_values,_ = network(network_params, feature_data, carry)
        predicted_values = predicted_values.squeeze()

        logger.debug(f"{log_probs.shape=}")

        returns = self.value_function(rewards, predicted_values)
        logger.debug(f"{returns.shape}")

        logger.debug(f"{predicted_values.shape=}")

        # (n_timesteps, n_particles)
        advantage = returns - predicted_values
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        # logger.info(f"{advantage=}")

        actor_loss = - 1 * ((log_probs * advantage).sum(axis=0)).sum()
        logger.debug(f"{actor_loss=}")

        # Sum over time steps and average over agents.
        critic_loss =  1 * optax.huber_loss(predicted_values, returns).sum(axis=0).sum()
        # logger.info(f"{critic_loss=}, {actor_loss=}")
        return actor_loss + critic_loss

    def compute_loss(self, network: Network, episode_data):
        """
        Compute the loss and update the shared actor-critic network.

        Parameters
        ----------
        network : Network
                actor-critic model to use in the analysis.
        episode_data : np.ndarray (n_timesteps, feature_dimension)
                Observable data for each time step and particle within the episode.

        Returns
        -------

        """
        feature_data = jnp.array(episode_data.feature_sequence)

        iterations, n_particles, *feature_dimension = feature_data.shape
        feature_data = feature_data.reshape((iterations * n_particles, *feature_dimension))
        carry = episode_data.carry

        reward_data = jnp.array(episode_data.rewards)
        log_probs = jnp.array(episode_data.log_probs)

        self.n_time_steps = jnp.shape(feature_data)[0]
        network_grad_fn = jax.value_and_grad(self._calculate_loss)
        _, network_grads = network_grad_fn(
            network.model_state.params,
            network=network,
            feature_data=feature_data,
            carry=carry,
            rewards=reward_data,
            log_probs=log_probs,
        )
        max_grad = jax.tree_util.tree_reduce(
            lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), network_grads, -jnp.inf
        )
        logger.info(f"Maximum gradient: {max_grad}")
        network_grads = jax.tree_map(
            lambda g: jnp.clip(g, -100.0, 100.0), network_grads
        )
        network.update_model(network_grads)
