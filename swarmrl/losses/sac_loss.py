import logging

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from functools import partial
from threading import Lock

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
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
        minimum_entropy: float = 0.0,
        polyak_averaging_tau: float = 0.005,
        lock: Lock = Lock(),
    ):
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
        self.error_predicted_reward = None
        self.minimum_entropy = minimum_entropy
        self.polyak_averaging_tau = polyak_averaging_tau
        self.running_mean = 0.0
        self.running_std = 1.0
        self.lock = lock

    def _calculate_loss(
        self,
        critic_network_params: FrozenDict,
        critic_network: Network,
        actor_network_params: FrozenDict,
        actor_network: Network,
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
        # line 13
        log_temp = actor_network_params["temperature"]
        desired_q_value, updated_batch_stats_target = self._calculate_desired_q_value(
            critic_network,
            actor_network_params,
            actor_network,
            next_feature_data,
            next_carry,
            rewards,
            action_sequence,
            log_temp,
        )

        (
            critic_loss,
            (critic_loss_per_sample, updated_batch_stats_critic),
        ), critic_grad = jax.value_and_grad(self._calculate_critic_loss, has_aux=True)(
            critic_network_params,
            critic_network,
            feature_data,
            carry,
            actions,
            action_sequence,
            desired_q_value,
        )

        # line 14
        (
            actor_loss,
            (log_probs, actor_loss_per_sample, updated_batch_stats_actor),
        ), actor_grad = jax.value_and_grad(self._calculate_actor_loss, has_aux=True)(
            actor_network_params,
            actor_network,
            critic_network_params,
            critic_network,
            feature_data,
            carry,
            action_sequence,
            jax.lax.stop_gradient(log_temp),
        )
        error_predicted_reward = jnp.squeeze(
            jnp.abs(critic_loss_per_sample) + jnp.abs(actor_loss_per_sample)
        )
        temperature_loss, temperature_grad = jax.value_and_grad(
            self._calculate_temperature_loss
        )(actor_network_params, log_probs)

        mean_critic_grad, mean_actor_grad = self.mean_gradients(critic_grad, actor_grad)
        logger.info(
            f"mean actor grad = {mean_actor_grad}, mean critic grad = {mean_critic_grad}"
        )

        assert actor_grad["temperature"] == 0

        critic_network.update_model(
            critic_grad, updated_batch_stats=updated_batch_stats_critic
        )

        actor_network.update_model(
            actor_grad, updated_batch_stats=updated_batch_stats_actor
        )
        actor_network.update_model(temperature_grad)

        critic_network.polyak_averaging(self.polyak_averaging_tau)
        return actor_loss, critic_loss, temperature_loss, error_predicted_reward

    def mean_gradients(self, critic_grad, actor_grad):
        mean_critic_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x), critic_grad)
        mean_actor_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x), actor_grad)
        mean_critic_grad = jax.tree_util.tree_reduce(
            lambda x, y: x + y, list(mean_critic_grad.values())
        ) / len(mean_critic_grad)
        mean_actor_grad = jax.tree_util.tree_reduce(
            lambda x, y: x + y, list(mean_actor_grad.values())
        ) / len(mean_actor_grad)
        return mean_critic_grad, mean_actor_grad

    # @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    def _calculate_actor_loss(
        self,
        actor_network_params,
        actor_network,
        critic_network_params,
        critic_network,
        feature_data,
        carry,
        action_sequence,
        log_temp,
    ):
        actions, log_probs, updated_batch_stats_actor = (
            actor_network.compute_action_training(
                actor_network_params, feature_data, action_sequence[:, :-1, :], carry
            )
        )
        first_q_value, second_q_value, _ = critic_network.compute_q_values_critic(
            critic_network_params,
            feature_data,
            actions,
            action_sequence[:, :-1, :],
            carry,
        )
        log_probs = jnp.expand_dims(log_probs, axis=-1)
        assert jnp.shape(first_q_value) == jnp.shape(log_probs)
        actor_loss = jnp.exp(log_temp) * log_probs - jnp.minimum(
            first_q_value, second_q_value
        )
        assert jnp.shape(actor_loss) == jnp.shape(first_q_value)
        return jnp.mean(actor_loss), (log_probs, actor_loss, updated_batch_stats_actor)

    # @partial(jax.jit, static_argnames=["self", "critic_network"])
    def _calculate_critic_loss(
        self,
        critic_network_params: FrozenDict,
        critic_network: Network,
        feature_data,
        carry,
        actions,
        action_sequence,
        desired_q_value,
    ):
        """Calculates the critic loss. Line 13 in SpinningUp.

        Args:
            network_params (_type_): _description_
            network (_type_): _description_
            feature_data (_type_): _description_
            carry (_type_): _description_
            actions (_type_): _description_
            action_sequence (_type_): _description_
            desired_q_value (_type_): _description_

        Returns:
            _type_: _description_
        """
        logger.debug(f"{jnp.shape(feature_data)=}")
        logger.debug(f"feature data = {feature_data}")
        first_q_value, second_q_value, updated_batch_stats_critic = (
            critic_network.compute_q_values_critic(
                critic_network_params,
                feature_data,
                actions,
                action_sequence[:, :-1, :],
                carry,
            )
        )

        critic_loss = (first_q_value - desired_q_value) ** 2 + (
            second_q_value - desired_q_value
        ) ** 2
        assert jnp.shape(critic_loss) == jnp.shape(desired_q_value)
        assert jnp.shape(critic_loss) == jnp.shape(first_q_value)
        logger.debug(f"{critic_loss=}")
        logger.debug(f"{desired_q_value=}")
        logger.debug(f"{first_q_value=}")
        logger.debug(f"{second_q_value=}")
        logger.debug(
            f"shape of all = {jnp.shape(critic_loss)}, {jnp.shape(desired_q_value)}, {jnp.shape(first_q_value)}, {jnp.shape(second_q_value)}"
        )
        return jnp.mean(critic_loss), (critic_loss, updated_batch_stats_critic)

    # @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    def _calculate_desired_q_value(
        self,
        critic_network,
        actor_network_params,
        actor_network,
        next_feature_data,
        next_carry,
        rewards,
        action_sequence,
        log_temp,
    ):
        next_action, next_log_probs, _ = actor_network.compute_action_training(
            actor_network_params,
            next_feature_data,
            action_sequence[:, 1:, :],
            next_carry,
        )
        next_log_probs = jnp.expand_dims(next_log_probs, axis=-1)
        first_q_value, second_q_value, updated_batch_stats_target = (
            critic_network.compute_q_values_target(
                next_feature_data,
                next_action,
                action_sequence[:, 1:, :],
                next_carry,
            )
        )
        next_q_value = jnp.minimum(first_q_value, second_q_value)
        desired_q_value = self.value_function(
            rewards,
            next_q_value,
            jnp.exp(log_temp),
            next_log_probs,
        )
        desired_q_value = jax.lax.stop_gradient(desired_q_value)
        assert jnp.shape(desired_q_value) == jnp.shape(first_q_value)
        assert jnp.shape(desired_q_value) == jnp.shape(next_log_probs)
        return desired_q_value, updated_batch_stats_target

    # @partial(jax.jit, static_argnames=["self"])
    def _calculate_temperature_loss(self, actor_network_params, log_probs):
        temperature = actor_network_params["temperature"]
        return -jnp.mean(
            temperature * jax.lax.stop_gradient(log_probs + self.minimum_entropy)
        )

    def normalize_rewards(self, rewards):
        self.running_mean = self.running_mean * 0.99 + jnp.mean(rewards) * 0.01
        self.running_std = self.running_std * 0.99 + jnp.std(rewards) * 0.01
        self.running_std = jnp.maximum(self.running_std, 1e-2)
        return (rewards - self.running_mean) / self.running_std

    def compute_loss(
        self, actor_network: Network, critic_network: Network, episode_data
    ):
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
        with self.lock:
            feature_data = jnp.array(episode_data.feature_sequence).copy()
            next_feature_data = jnp.array(episode_data.next_features).copy()
            iterations, n_particles, *feature_dimension = feature_data.shape
            iterations_next, *_ = next_feature_data.shape
            reward_data = jnp.array(episode_data.rewards)[
                :iterations_next, jnp.newaxis
            ].copy()
            feature_data = feature_data.reshape(
                (iterations * n_particles, *feature_dimension)
            )[:iterations_next]
            next_feature_data = next_feature_data.reshape(
                ((iterations_next) * n_particles, *feature_dimension)
            )
            # if jnp.shape(reward_data)[0] != iterations_next:
                # iterations -= 1
                # iterations_next -= 1
                # next_feature_data = next_feature_data[:iterations_next]
                # feature_data = feature_data[:iterations]
            next_carry_data = jnp.array(episode_data.next_carry).copy()
            next_carry_data = jnp.squeeze(next_carry_data)[:iterations_next]
            next_carry_data = tuple(jnp.swapaxes(next_carry_data, 0, 1))
            carry = jnp.array(episode_data.carry).copy()
            carry = jnp.squeeze(carry)[:iterations_next]
            carry = tuple(jnp.swapaxes(carry, 0, 1))
            actions = jnp.array(episode_data.actions)[:iterations_next].copy()
            action_sequence = jnp.array(episode_data.action_sequence)[
                :iterations_next
            ].copy()
            # reward_data = self.normalize_rewards(reward_data)
            self.n_time_steps = jnp.shape(feature_data)[0]
            logger.info(f"feature_data = {feature_data.shape}")
            logger.info(f"next_feature_data = {next_feature_data.shape}")
            logger.info(f"reward_data = {reward_data.shape}")
            logger.info(f"action_sequence = {action_sequence.shape}")
            logger.info(f"next_feature_data = {next_feature_data.shape}")
            logger.info(f"iterations = {iterations}, {iterations_next}")
            
        if jnp.isnan(reward_data).any():
            raise ValueError("Nan in reward data")
        first_actor_loss, first_critic_loss, first_temperature_loss, _ = (
            self._calculate_loss(
                critic_network_params=critic_network.critic_state.params,
                critic_network=critic_network,
                actor_network_params=actor_network.model_state.params,
                actor_network=actor_network,
                feature_data=feature_data,
                next_feature_data=next_feature_data,
                carry=carry,
                next_carry=next_carry_data,
                rewards=reward_data,
                actions=actions,
                action_sequence=action_sequence,
            )
        )
        # for _ in range(10):
        #     self._calculate_loss(
        #         critic_network_params=critic_network.critic_state.params,
        #         critic_network=critic_network,
        #         actor_network_params=actor_network.model_state.params,
        #         actor_network=actor_network,
        #         feature_data=feature_data,
        #         next_feature_data=next_feature_data,
        #         carry=carry,
        #         next_carry=next_carry_data,
        #         rewards=reward_data,
        #         actions=actions,
        #         action_sequence=action_sequence,
        #     )
        (
            second_actor_loss,
            second_critic_loss,
            second_temperature_loss,
            self.error_predicted_reward,
        ) = self._calculate_loss(
            critic_network_params=critic_network.critic_state.params,
            critic_network=critic_network,
            actor_network_params=actor_network.model_state.params,
            actor_network=actor_network,
            feature_data=feature_data,
            next_feature_data=next_feature_data,
            carry=carry,
            next_carry=next_carry_data,
            rewards=reward_data,
            actions=actions,
            action_sequence=action_sequence,
        )

        logger.info(f"{actor_network.get_exp_temperature()=}")
        logger.info(
            f"first iteration losses = (a,c,t){first_actor_loss, first_critic_loss, first_temperature_loss} \n second iteration losses = {second_actor_loss, second_critic_loss, second_temperature_loss}"
        )
