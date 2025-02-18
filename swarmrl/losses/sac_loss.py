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
import flax

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.utils.utils import gather_n_dim_indices
from swarmrl.value_functions.td_return_sac import TDReturnsSAC

logger = logging.getLogger(__name__)


class SoftActorCriticGradientLoss(Loss):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(
        self,
        value_function: TDReturnsSAC = TDReturnsSAC(),
        temperature: float = 0.2,
        learning_rate: float = 0.01,
        minimum_entropy: float = 0.0,
    ):
        """
        Constructor for the reward class.

        Parameters
        ----------
        value_function : ExpectedReturns
        """
        super(Loss, self).__init__()
        self.value_function = value_function
        self.learning_rate = learning_rate
        self.n_particles = None
        self.n_time_steps = None
        self.error_predicted_reward = None
        self.temperature = temperature
        self.minimum_entropy = minimum_entropy

    def _calculate_loss(
        self,
        target_network_params: FrozenDict,
        target_network: Network,
        network_params: FrozenDict,
        network: Network,
        feature_data: jnp.ndarray,
        next_feature_data: jnp.ndarray,
        carry: jnp.ndarray,
        next_carry: jnp.ndarray,
        rewards: jnp.ndarray,
        actions: jnp.ndarray,
        action_sequence: jnp.ndarray,
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
        # line 12
        next_action, next_log_probs = network.compute_action_training(
            next_feature_data, action_sequence[:, 1:, :], next_carry
        )
        next_q_value = jnp.min(
            jnp.array(
                [
                    target_network.compute_q_values(
                        next_feature_data,
                        next_action,
                        action_sequence[:, 1:, :],
                        next_carry,
                    )
                ]
            )
        )
        desired_q_value = self.value_function(
            rewards,
            next_q_value,
            self.temperature,
            next_log_probs,
        )
        desired_q_value = jax.lax.stop_gradient(desired_q_value)

        # line 13
        first_q_value, second_q_value = network.compute_q_values(
            feature_data, actions, action_sequence[:, :-1, :], carry
        )
        critic_loss = jnp.mean((first_q_value - desired_q_value) ** 2) + jnp.mean(
            (second_q_value - desired_q_value) ** 2
        )
        critic_grad = jax.grad(lambda x: critic_loss)(network_params)
        network.update_model(critic_grad)

        # line 14
        actions, log_probs = network.compute_action_training(
            feature_data, action_sequence[:, :-1, :], carry
        )
        first_q_value, second_q_value = network.compute_q_values(
            feature_data, actions, action_sequence[:, :-1, :], carry
        )
        actor_loss = jnp.mean(
            (jnp.minimum(first_q_value, second_q_value) - self.temperature * log_probs)
        )
        actor_grad = jax.grad(lambda x: actor_loss)(network_params)
        network.update_model(actor_grad)

        self.temperature = self.temperature - self.learning_rate * (
            jnp.mean(log_probs) + self.minimum_entropy
        )
        self.error_predicted_reward = (
            (first_q_value.squeeze() - desired_q_value)
            + (second_q_value.squeeze() - desired_q_value) ** 2
            + jnp.minimum(first_q_value, second_q_value).squeeze()
            - self.temperature * log_probs
        )
        logger.info(f"{self.temperature=}, {actor_loss=}, {critic_loss=}")
        self.polyak_averaging(0.99, target_network_params, network_params)

    def compute_loss(self, network: Network, target_network: Network, episode_data):
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
        next_feature_data = jnp.array(episode_data.next_features)
        iterations, n_particles, *feature_dimension = feature_data.shape
        iterations_next, *_ = next_feature_data.shape
        feature_data = feature_data.reshape(
            (iterations * n_particles, *feature_dimension)
        )[:iterations_next]
        next_feature_data = next_feature_data.reshape(
            ((iterations_next) * n_particles, *feature_dimension)
        )
        next_carry_data = jnp.array(episode_data.next_carry)
        next_carry_data = jnp.squeeze(next_carry_data)[:iterations_next]
        next_carry_data = tuple(jnp.swapaxes(next_carry_data, 0, 1))
        carry = jnp.array(episode_data.carry)
        carry = jnp.squeeze(carry)[:iterations_next]
        carry = tuple(jnp.swapaxes(carry, 0, 1))
        actions = jnp.array(episode_data.actions)[:iterations_next]
        action_sequence = jnp.array(episode_data.action_sequence)[:iterations_next]
        reward_data = jnp.array(episode_data.rewards)[:iterations_next]
        self.n_time_steps = jnp.shape(feature_data)[0]
        self._calculate_loss(
            target_network_params=target_network.model_state.params,
            target_network=target_network,
            network_params=network.model_state.params,
            network=network,
            feature_data=feature_data,
            next_feature_data=next_feature_data,
            carry=carry,
            next_carry=next_carry_data,
            rewards=reward_data,
            actions=actions,
            action_sequence=action_sequence,
        )

    def polyak_averaging(
        self, decay_rate: float, target_network_params, action_network_params
    ):
        """
        Update the target network using Polyak averaging.
        Move values from action netwrok to target network.

        Parameters
        ----------
        decay_rate : float
            The rate at which the target network is updated.
        target_params : jnp.array
            The parameters of the target network.
        params : jnp.array
            The parameters of the network.

        Returns
        -------
        target_params : jnp.array
            The updated parameters of the target network.
        """
        target_network_params = flax.core.unfreeze(target_network_params)
        common_layers = set(target_network_params.keys()).intersection(
            set(action_network_params.keys())
        )
        for layer in common_layers:
            target_network_params[layer] = jax.tree_util.tree_map(
                lambda target, action: decay_rate * target + (1 - decay_rate) * action,
                target_network_params[layer],
                action_network_params[layer],
            )

        return flax.core.freeze(target_network_params)
