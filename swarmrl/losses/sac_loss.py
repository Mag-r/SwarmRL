import logging

import jax
import jax.numpy as jnp
import numpy as onp
from flax.core.frozen_dict import FrozenDict
from functools import partial
from threading import Lock

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.value_functions.td_return_sac import TDReturnsSAC

logger = logging.getLogger(__name__)


class SoftActorCriticGradientLoss(Loss):
    """
    Implements the loss function for the Soft Actor-Critic (SAC) algorithm.
    In theory it does not need a lot of hyperparameter-tuning.
    Based on the paper "Soft Actor-Critic Algorithms and Applications" by Haarnoja et al. (2018).

    Notes:
    - Only continous actions are supported.
    - Seperate networks for actor and critic/target are used.
    - Currently only for global agents. (TODO: add support for multi-agent)
    -----
    """

    def __init__(
        self,
        value_function: TDReturnsSAC = TDReturnsSAC(),
        minimum_entropy: float = 0.0,
        polyak_averaging_tau: float = 0.005,
        lock: Lock = Lock(),
        validation_split: float = 0.1,
        fix_temperature: bool = False,
        batch_size = 256,
    ):
        """
        Constructor for the SoftActorCriticGradientLoss class.

        Parameters
        ----------
        value_function : TDReturnsSAC
            The value function used for the loss calculation.
        minimum_entropy : float
            The minimum entropy for the actor network. Best set to -dim(ActionSpace).
        polyak_averaging_tau : float
            The polyak averaging factor for the target network. Default is 0.005.
        lock : Lock
            A lock for thread safety.
        validation_split : float
            The fraction of data to use for validation. Default is 0.1.
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
        self.validation_split = validation_split
        self.fix_temperature = fix_temperature
        self.validation_losses = []
        self.training_losses = []
        self.temperature_history = []
        self.iteration_counter = 0
        self.batch_size = batch_size
        self.error_predicted_reward = jnp.zeros((self.batch_size,))

    def _calculate_loss_validation(
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
    ):
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
            (critic_loss_per_sample, _),
        ) = self._calculate_critic_loss(
            critic_network_params,
            critic_network,
            feature_data,
            carry,
            actions,
            action_sequence,
            desired_q_value,
        )
        (
            actor_loss,
            (log_probs, actor_loss_per_sample, _),
        ) = self._calculate_actor_loss(
            actor_network_params,
            actor_network,
            critic_network_params,
            critic_network,
            feature_data,
            carry,
            action_sequence,
            jax.lax.stop_gradient(log_temp),
        )
        temperature_loss = self._calculate_temperature_loss(
            actor_network_params, log_probs
        )

        return actor_loss, critic_loss, temperature_loss

    def _calculate_loss_apply_gradients(
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

        # mean_critic_grad, mean_actor_grad = self.mean_gradients(critic_grad, actor_grad)
        # logger.info(
        #     f"mean actor grad = {mean_actor_grad}, mean critic grad = {mean_critic_grad}"
        # )

        assert actor_grad["temperature"] == 0

        critic_network.update_model(
            critic_grad, updated_batch_stats=updated_batch_stats_critic
        )

        actor_network.update_model(
            actor_grad, updated_batch_stats=updated_batch_stats_actor
        )
        actor_network.update_model(temperature_grad) if not self.fix_temperature else None

        critic_network.polyak_averaging(self.polyak_averaging_tau)
        return actor_loss, critic_loss, temperature_loss, error_predicted_reward

    def mean_gradients(self, critic_grad, actor_grad):
        """Calculates the mean gradients of the critic and actor networks.
        Args:
            critic_grad: gradients of the critic network.
            actor_grad: gradients of the actor network.
        Returns:
            tuple: mean gradients of the critic and actor networks.
        """
        mean_critic_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x), critic_grad)
        mean_actor_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x), actor_grad)
        mean_critic_grad = jax.tree_util.tree_reduce(
            lambda x, y: x + y, list(mean_critic_grad.values())
        ) / len(mean_critic_grad)
        mean_actor_grad = jax.tree_util.tree_reduce(
            lambda x, y: x + y, list(mean_actor_grad.values())
        ) / len(mean_actor_grad)
        return mean_critic_grad, mean_actor_grad

    @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    def _calculate_actor_loss(
        self,
        actor_network_params: FrozenDict,
        actor_network: Network,
        critic_network_params: FrozenDict,
        critic_network: Network,
        feature_data: jnp.ndarray,
        carry: tuple,
        action_sequence: jnp.ndarray,
        log_temp: jnp.ndarray,
    ) -> tuple[float, tuple]:
        """Calculates the actor loss. Eq 7 in SAC-paper.

        Args:
            actor_network_params (FrozenDict): parameters of the actor network.
            actor_network (Network): actor network.
            critic_network_params (FrozenDict): parameters of the critic network.
            critic_network (Network): critic network.
            feature_data (jnp.ndarray): feature data.
            carry (tuple): carry data.
            action_sequence (jnp.ndarray): action sequence.
            log_temp (jnp.ndarray): log temperature.

        Returns:
            tuple(float, tuple): actor loss and tuple of log probabilities, actor loss per sample, and updated batch stats.
        """
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

    @partial(jax.jit, static_argnames=["self", "critic_network"])
    def _calculate_critic_loss(
        self,
        critic_network_params: FrozenDict,
        critic_network: Network,
        feature_data: jnp.ndarray,
        carry: tuple,
        actions: jnp.ndarray,
        action_sequence: jnp.ndarray,
        desired_q_value: jnp.ndarray,
    ) -> tuple[float, tuple]:
        """Calculates the critic loss. Eq 5 in SAC-paper.

        Args:
            critic_network_params (FrozenDict): parameters of the critic network.
            critic_network (Network): critic network.
            feature_data (jnp.ndarray): feature data.
            carry (tuple): carry data.
            actions (jnp.ndarray): actions taken.
            action_sequence (jnp.ndarray): action sequence.
            desired_q_value (jnp.ndarray): desired Q value.
        Returns:
            tuple(float, tuple): critic loss and tuple of critic loss per sample and updated batch stats for the critic.
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

    @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    def _calculate_desired_q_value(
        self,
        critic_network: Network,
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

    @partial(jax.jit, static_argnames=["self"])
    def _calculate_temperature_loss(
        self, actor_network_params: FrozenDict, log_probs: jnp.ndarray
    ) -> float:
        """Calculates the temperature loss. Eq 18 in SAC-paper.

        Args:
            actor_network_params (FrozenDict): parameters of the actor network.
            log_probs (jnp.ndarray): log probabilities of the actions taken.

        Returns:
            float: temperature loss.
        """
        temperature = actor_network_params["temperature"]
        return -jnp.mean(
            temperature * jax.lax.stop_gradient(log_probs + self.minimum_entropy)
        )

    def normalize_rewards(self, rewards: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize the rewards using running mean and standard deviation.
        Parameters
        ----------
        rewards : jnp.ndarray
            The rewards to normalize.
        Returns
        -------
        jnp.ndarray
            The normalized rewards.
        """

        self.running_mean = self.running_mean * 0.99 + jnp.mean(rewards) * 0.01
        self.running_std = self.running_std * 0.99 + jnp.std(rewards) * 0.01
        self.running_std = jnp.maximum(self.running_std, 1e-2)
        return (rewards - self.running_mean) / self.running_std

    def _split_training_validation(self, data):
        """
        Split the data into training and validation sets.

        Parameters
        ----------
        data : jnp.ndarray
            The data to split.

        Returns
        -------
        tuple
            The training and validation data.
        """
        n_samples = data.shape[0]
        split_index = int(n_samples * (1-self.validation_split))
        split_index = onp.clip(split_index, n_samples-20, n_samples-2)
        return data[:split_index], data[split_index:]

    def compute_loss(
        self, actor_network: Network, critic_network: Network, episode_data
    ) -> None:
        """
        Compute the loss and update the actor, critic and target network.


        Parameters
        ----------
        actor_network : Network
            The policy giving network.
        critic_network : Network
            The critic and target network.
        episode_data : EpisodeData
            The data used for training.

        """
        with self.lock:
            feature_data = jnp.array(episode_data.feature_sequence).copy()
            next_feature_data = jnp.array(episode_data.next_features).copy()
            iterations, n_particles, *feature_dimension = feature_data.shape
            iterations_next, *_ = next_feature_data.shape
            reward_data = jnp.array(episode_data.rewards)[
                :iterations_next, jnp.newaxis
            ].copy()
            carry = jnp.array(episode_data.carry).copy()
            actions = jnp.array(episode_data.actions)[:iterations_next].copy()
            action_sequence = jnp.array(episode_data.action_sequence)[
                :iterations_next
            ].copy()
            next_carry_data = jnp.array(episode_data.next_carry).copy()

        feature_data = feature_data.reshape(
            (iterations * n_particles, *feature_dimension)
        )[:iterations_next]
        next_feature_data = next_feature_data.reshape(
            ((iterations_next) * n_particles, *feature_dimension)
        )
        if jnp.shape(reward_data)[0] != iterations_next:
            iterations -= 1
            iterations_next -= 1
            next_feature_data = next_feature_data[:iterations_next]
            feature_data = feature_data[:iterations]
        assert (
            (jnp.shape(feature_data)[0] == iterations_next)
            and (jnp.shape(next_feature_data)[0] == iterations_next)
            and (jnp.shape(reward_data)[0] == iterations_next)
        ), f"feature_data = {feature_data.shape}, next_feature_data = {next_feature_data.shape}"

        
        feature_training, feature_validation = self._split_training_validation(
            feature_data
        )
        next_feature_training, next_feature_validation = (
            self._split_training_validation(next_feature_data)
        )
        action_training, action_validation = self._split_training_validation(actions)
        reward_training, reward_validation = self._split_training_validation(
            reward_data
        )
        action_sequence_training, action_sequence_validation = (
            self._split_training_validation(action_sequence)
        )
        carry = jnp.squeeze(carry)[:iterations_next]
        carry_training, carry_validation = self._split_training_validation(carry)
        carry_training = tuple(jnp.swapaxes(carry_training, 0, 1))
        carry_validation = tuple(jnp.swapaxes(carry_validation, 0, 1))
        self.n_time_steps = jnp.shape(feature_data)[0]
        next_carry_data = jnp.squeeze(next_carry_data)[:iterations_next]
        next_carry_training, next_carry_validation = self._split_training_validation(
            next_carry_data
        )
        next_carry_training = tuple(jnp.swapaxes(next_carry_training, 0, 1))
        next_carry_validation = tuple(jnp.swapaxes(next_carry_validation, 0, 1))

        if jnp.isnan(reward_data).any():
            raise ValueError("Nan in reward data")
        actor_training_loss = 0.0
        critic_training_loss = 0.0
        temperature_training_loss = 0.0
        if feature_training.shape[0] >self.batch_size:
            num_batches = feature_training.shape[0] // self.batch_size

            for batch_idx in range(num_batches + 1):
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_end = min(batch_end, feature_training.shape[0])
                feature_batch = feature_training[batch_start:batch_end]
                next_feature_batch = next_feature_training[batch_start:batch_end]
                action_batch = action_training[batch_start:batch_end]
                reward_batch = reward_training[batch_start:batch_end]
                action_sequence_batch = action_sequence_training[batch_start:batch_end]
                carry_batch = tuple(c[batch_start:batch_end] for c in carry_training)
                next_carry_batch = tuple(c[batch_start:batch_end] for c in next_carry_training)

                batched_actor_training_loss, batched_critic_training_loss, batched_temperature_trainig_loss, error_predicted_reward = (
                self._calculate_loss_apply_gradients(
                    critic_network_params=critic_network.critic_state.params,
                    critic_network=critic_network,
                    actor_network_params=actor_network.model_state.params,
                    actor_network=actor_network,
                    feature_data=feature_batch,
                    next_feature_data=next_feature_batch,
                    carry=carry_batch,
                    next_carry=next_carry_batch,
                    rewards=reward_batch,
                    actions=action_batch,
                    action_sequence=action_sequence_batch,
                )
                )
                logger.info(f"batch {batch_idx} actor training loss = {batched_actor_training_loss}, critic training loss = {batched_critic_training_loss}, temperature training loss = {batched_temperature_trainig_loss}")
                if self.error_predicted_reward.shape[0] < batch_end:
                    padding_length = batch_end - self.error_predicted_reward.shape[0]
                    self.error_predicted_reward = jnp.pad(
                        self.error_predicted_reward, (0, padding_length), constant_values=0
                    )
                self.error_predicted_reward = self.error_predicted_reward.at[batch_start:batch_end].set(error_predicted_reward)
                actor_training_loss += batched_actor_training_loss/num_batches
                critic_training_loss += batched_critic_training_loss /num_batches
                temperature_training_loss += batched_temperature_trainig_loss /num_batches
            
        else:
            actor_training_loss, critic_training_loss, temperature_training_loss, self.error_predicted_reward = (
                self._calculate_loss_apply_gradients(
                    critic_network_params=critic_network.critic_state.params,
                    critic_network=critic_network,
                    actor_network_params=actor_network.model_state.params,
                    actor_network=actor_network,
                    feature_data=feature_training,
                    next_feature_data=next_feature_training,
                    carry=carry_training,
                    next_carry=next_carry_training,
                    rewards=reward_training,
                    actions=action_training,
                    action_sequence=action_sequence_training,
                )
            )

            
        actor_validation_loss, critic_validation_loss, temperature_validation_loss = (
            self._calculate_loss_validation(
                critic_network_params=critic_network.critic_state.params,
                critic_network=critic_network,
                actor_network_params=actor_network.model_state.params,
                actor_network=actor_network,
                feature_data=feature_validation,
                next_feature_data=next_feature_validation,
                carry=carry_validation,
                next_carry=next_carry_validation,
                rewards=reward_validation,
                actions=action_validation,
                action_sequence=action_sequence_validation,
            )
        )
        self.temperature_history.append(
            actor_network.get_exp_temperature())
        self.training_losses.append(
            (actor_training_loss, critic_training_loss, temperature_training_loss)
        )
        self.validation_losses.append(
            (actor_validation_loss, critic_validation_loss, temperature_validation_loss)
        )

        logger.info(f"{actor_network.get_exp_temperature()=}")
        logger.info(
            f"training losses = (a,c,t){actor_training_loss, critic_training_loss, temperature_training_loss}"
        )
        logger.info(
            f"validation losses = (a,c,t){actor_validation_loss, critic_validation_loss, temperature_validation_loss}"
        )
        if self.iteration_counter % 10 == 0:
            jnp.save(
                "training_losses.npy",
                jnp.array(self.training_losses),
                allow_pickle=True,
            )
            jnp.save(
                "validation_losses.npy",
                jnp.array(self.validation_losses),
                allow_pickle=True,
            )
            jnp.save(
                "temperature_history.npy",
                jnp.array(self.temperature_history),
                allow_pickle=True,
            )
        self.iteration_counter += 1
