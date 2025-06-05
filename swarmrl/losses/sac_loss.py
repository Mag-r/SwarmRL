import logging

import flax.core
import jax
import jax.numpy as jnp
import numpy as onp
from flax.core.frozen_dict import FrozenDict
import flax
from functools import partial
from threading import Lock

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.value_functions.td_return_sac import TDReturnsSAC
import time
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
        occupancy_map: jnp.ndarray,
        next_occupancy_map: jnp.ndarray,
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
            next_occupancy_map,
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
            occupancy_map,
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
            occupancy_map,
        )
        temperature_loss = self._calculate_temperature_loss(
            actor_network_params, log_probs
        )

        return actor_loss, critic_loss, temperature_loss
    
    @partial(jax.jit, static_argnames=["self", "critic_network", "actor_network"])
    def combined_loss_and_grads(
        self,
        critic_params: FrozenDict,
        actor_params: FrozenDict,
        critic_network: Network,
        actor_network: Network,
        feature_data: jnp.ndarray,
        carry: tuple,
        actions: jnp.ndarray,
        action_sequence: jnp.ndarray,
        next_feature_data: jnp.ndarray,
        next_carry: tuple,
        rewards: jnp.ndarray,
        occupancy_map: jnp.ndarray,
        next_occupancy_map: jnp.ndarray,
        minimum_entropy: float,
    ):
        """
        Compute:
        - desired_q via self._calculate_desired_q_value
        - critic_loss via self._calculate_critic_loss
        - actor_loss via self._calculate_actor_loss
        - temperature_loss via self._calculate_temperature_loss

        Returns:
        (total_loss, aux), where `aux` is a dict containing:
            {
            "critic_loss": scalar,
            "actor_loss": scalar,
            "temperature_loss": scalar,
            "batch_stats_critic": ...,
            "batch_stats_actor": ...,
            "batch_stats_tgt": ...,
            "error_pred":   per‐sample abs(critic_err) + abs(actor_err)
            }
        """
        critic_params_frozen = jax.tree_map(lambda x: jax.lax.stop_gradient(x), critic_params)
        actor_params_frozen = jax.tree_map(lambda x: jax.lax.stop_gradient(x), actor_params)
        # 1) Desired Q‐value (calls your existing method)
        desired_q, batch_stats_tgt = self._calculate_desired_q_value(
            critic_network = critic_network,
            actor_network_params=actor_params_frozen,
            actor_network=actor_network,
            next_feature_data=next_feature_data,
            next_carry=next_carry,
            rewards=rewards,
            action_sequence=action_sequence,
            log_temp=jax.lax.stop_gradient(actor_params_frozen["temperature"]),
            next_occupancy_map=next_occupancy_map,
        )

        # 2) Critic loss + stats (calls your existing method)
        critic_loss, (critic_loss_per_sample, batch_stats_critic) = self._calculate_critic_loss(
            critic_network_params= critic_params,
            critic_network=critic_network,
            feature_data=feature_data,
            carry=carry,
            actions=actions,
            action_sequence=action_sequence,
            desired_q_value=desired_q,
            occupancy_map=occupancy_map,
        )

        # 3) Actor loss + stats (calls your existing method)
        actor_loss, (log_probs, actor_loss_per_sample, batch_stats_actor) = self._calculate_actor_loss(
            actor_network_params=actor_params,
            actor_network=actor_network,
            critic_network_params=critic_params_frozen,
            critic_network=critic_network,
            feature_data=feature_data,
            carry=carry,
            action_sequence=action_sequence,
            log_temp=jax.lax.stop_gradient(actor_params_frozen["temperature"]),
            occupancy_map=occupancy_map,
        )

        # 4) Temperature loss (calls your existing method)
        temperature_loss = self._calculate_temperature_loss(
            actor_params,
            log_probs,
            minimum_entropy=minimum_entropy,
        )

        # 5) Form one scalar (mean of each) to backpropagate
        #    (you can weight them differently if you like)
        scalar_critic = jnp.mean(critic_loss)
        scalar_actor = jnp.mean(actor_loss)
        scalar_temp = jnp.mean(temperature_loss)
        total_loss = scalar_critic + scalar_actor + scalar_temp

        aux = {
            "critic_loss": scalar_critic,
            "actor_loss": scalar_actor,
            "temperature_loss": scalar_temp,
            "batch_stats_critic": batch_stats_critic,
            "batch_stats_actor": batch_stats_actor,
            "batch_stats_tgt": batch_stats_tgt,
            # per‐sample error = |critic_err| + |actor_err|
            "error_pred": jnp.abs(critic_loss_per_sample) + jnp.abs(actor_loss_per_sample),
        }
        return total_loss, aux

    def _calculate_loss_apply_gradients(
        self,
        critic_network: Network,
        actor_network: Network,
        feature_data: jnp.ndarray,
        next_feature_data: jnp.ndarray,
        carry: tuple,
        next_carry: tuple,
        rewards: jnp.ndarray,
        actions: jnp.ndarray,
        action_sequence: jnp.ndarray,
        occupancy_map: jnp.ndarray,
        next_occupancy_map: jnp.ndarray,
    ) -> jnp.array:
        """
        (Unchanged signature, but now we call combined_loss_and_grads once to get both grads.)
        """
        # 1) Snapshot current params
        critic_params = critic_network.critic_state.params
        actor_params = actor_network.model_state.params

        # 2) Run one `value_and_grad(..., argnums=(0,1))` to get (grad_critic, grad_actor) in one pass.
        (total_loss, aux), (grad_critic, grad_actor) = jax.value_and_grad(
            self.combined_loss_and_grads, argnums=(0, 1), has_aux=True
        )(
            critic_params,
            actor_params,
            critic_network,
            actor_network,
            feature_data,
            carry,
            actions,
            action_sequence,
            next_feature_data,
            next_carry,
            rewards,
            occupancy_map,
            next_occupancy_map,
            self.minimum_entropy,
        )

        # 3) Apply `grad_critic` via your existing Flax‐state update (inside the lock or scan, as before)
        critic_network.update_model(
            grad_critic, updated_batch_stats=aux["batch_stats_critic"]
        )

        # 4) Polyak‐average the target network
        critic_network.polyak_averaging(self.polyak_averaging_tau)

        # 5) Apply `grad_actor` via your existing Flax‐state update
        actor_network.update_model(
            grad_actor, updated_batch_stats=aux["batch_stats_actor"]
        )

        # 7) Return the sub‐losses and per‐sample error for logging
        return aux["actor_loss"], aux["critic_loss"], aux["temperature_loss"], aux["error_pred"]


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
        occupancy_map: jnp.ndarray,
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
        actions, log_probs, updated_batch_stats_actor, logits = (
            actor_network.compute_action_training(
                actor_network_params, feature_data, action_sequence[:, :-1, :], occupancy_map, carry
            )
        )
        first_q_value, second_q_value, _ = critic_network.compute_q_values_critic(
            critic_network_params,
            feature_data,
            actions,
            action_sequence[:, :-1, :],
            occupancy_map,
            carry,
        )
        log_probs = jnp.expand_dims(log_probs, axis=-1)
        assert jnp.shape(first_q_value) == jnp.shape(log_probs)
        actor_loss = jnp.exp(log_temp) * log_probs - jnp.minimum(
            first_q_value, second_q_value
        )
        # l2_regularization = sum(
        #     jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(actor_network_params)
        # )
        # actor_loss += 1e-7 * l2_regularization
        # variance = logits[:,6:]
        # actor_loss += 3e-8 * jnp.mean(variance)
        # mean = logits[:, :6]
        # actor_loss += 8e-8 * jnp.mean(jnp.square(mean))
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
        occupancy_map: jnp.ndarray,
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
        # logger.debug(f"{jnp.shape(feature_data)=}")
        # logger.debug(f"feature data = {feature_data}")
        first_q_value, second_q_value, updated_batch_stats_critic = (
            critic_network.compute_q_values_critic(
                critic_network_params,
                feature_data,
                actions,
                action_sequence[:, :-1, :],
                occupancy_map,
                carry,
            )
        )

        critic_loss = 1/2*((first_q_value - desired_q_value) ** 2 + (
            second_q_value - desired_q_value
        ) ** 2)
        # l2_regularization = sum(
        #     jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(critic_network_params)
        # )
        # critic_loss += 3e-7 * l2_regularization
        # critic_loss *= 0.3
        assert jnp.shape(critic_loss) == jnp.shape(desired_q_value)
        assert jnp.shape(critic_loss) == jnp.shape(first_q_value)
        # logger.debug(f"{critic_loss=}")
        # logger.debug(f"{desired_q_value=}")
        # logger.debug(f"{first_q_value=}")
        # logger.debug(f"{second_q_value=}")
        # logger.debug(
        #     f"shape of all = {jnp.shape(critic_loss)}, {jnp.shape(desired_q_value)}, {jnp.shape(first_q_value)}, {jnp.shape(second_q_value)}"
        # )
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
        next_occupancy_map,
    ):
        """Calculates the desired Q value, to be used in the critic loss. Eq 5 in SAC-paper.

        Args:
            critic_network (Network): _description_
            actor_network_params (_type_): _description_
            actor_network (_type_): _description_
            next_feature_data (_type_): _description_
            next_carry (_type_): _description_
            rewards (_type_): _description_
            action_sequence (_type_): _description_
            log_temp (_type_): _description_

        Returns:
            _type_: _description_
        """

        next_action, next_log_probs, *_ = actor_network.compute_action_training(
            actor_network_params,
            next_feature_data,
            action_sequence[:, 1:, :],
            next_occupancy_map,
            next_carry,
        )
        next_log_probs = jnp.expand_dims(next_log_probs, axis=-1)
        first_q_value, second_q_value, updated_batch_stats_target = (
            critic_network.compute_q_values_target(
                next_feature_data,
                next_action,
                action_sequence[:, 1:, :],
                next_occupancy_map,
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
        self, actor_network_params: FrozenDict, log_probs: jnp.ndarray, minimum_entropy: float = -6.0
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
            temperature * jax.lax.stop_gradient(log_probs + minimum_entropy)
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
        split_index = onp.clip(split_index, n_samples-5, n_samples-2)
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
       
        start = time.time()

        # Thread-safe data extraction
        with self.lock:
            feature_seq = jnp.array(episode_data.feature_sequence)
            next_feats = jnp.array(episode_data.next_features)
            rewards = jnp.array(episode_data.rewards)
            carry = jnp.array(episode_data.carry)
            next_carry = jnp.array(episode_data.next_carry)
            actions = jnp.array(episode_data.actions)
            action_seq = jnp.array(episode_data.action_sequence)
            occupancy = jnp.array(episode_data.occupancy_map)
            next_occupancy = jnp.array(episode_data.next_occupancy_map)

        # Unpack dimensions
        iterations, n_particles, *feat_dims = feature_seq.shape
        iter_next, *_ = next_feats.shape

        # Reshape and truncate to match next_features
        flat_features = feature_seq.reshape((iterations * n_particles, *feat_dims))[:iter_next]
        flat_next_feats = next_feats.reshape((iter_next * n_particles, *feat_dims))
        truncated_rewards = rewards[:iter_next, None]
        flat_actions = actions[:iter_next]
        flat_action_seq = action_seq[:iter_next]
        flat_occupancy = occupancy[:iterations]
        flat_next_occupancy = next_occupancy[:iter_next]

        if truncated_rewards.shape[0] != flat_next_feats.shape[0]:
            # Adjust in case of mismatch
            mismatch = truncated_rewards.shape[0] - (flat_next_feats.shape[0] // n_particles)
            trim = abs(mismatch)
            iterations -= 1
            iter_next -= 1
            flat_features = flat_features[:iterations]
            flat_next_feats = flat_next_feats[:iter_next]
            truncated_rewards = truncated_rewards[:iter_next]
            flat_actions = flat_actions[:iter_next]
            flat_action_seq = flat_action_seq[:iter_next]
            flat_occupancy = flat_occupancy[:iterations]
            flat_next_occupancy = flat_next_occupancy[:iter_next]

        assert (flat_features.shape[0] == iter_next and
                flat_next_feats.shape[0] == iter_next and
                truncated_rewards.shape[0] == iter_next), \
               f"Shapes mismatch: features {flat_features.shape}, next {flat_next_feats.shape}, rewards {truncated_rewards.shape}"

        # Split training/validation sets
        (feat_train, feat_val) = self._split_training_validation(flat_features)
        (next_feat_train, next_feat_val) = self._split_training_validation(flat_next_feats)
        (act_train, act_val) = self._split_training_validation(flat_actions)
        (rew_train, rew_val) = self._split_training_validation(truncated_rewards)
        (aseq_train, aseq_val) = self._split_training_validation(flat_action_seq)
        (occ_train, occ_val) = self._split_training_validation(flat_occupancy)
        (nocc_train, nocc_val) = self._split_training_validation(flat_next_occupancy)

        carry = carry[:iter_next].squeeze()
        (carry_train, carry_val) = self._split_training_validation(carry)
        carry_train = tuple(jnp.swapaxes(c, 0, 1) for c in carry_train)
        carry_val = tuple(jnp.swapaxes(c, 0, 1) for c in carry_val)
        self.n_time_steps = flat_features.shape[0]

        next_carry = next_carry[:iter_next].squeeze()
        (ncarry_train, ncarry_val) = self._split_training_validation(next_carry)
        ncarry_train = tuple(jnp.swapaxes(c, 0, 1) for c in ncarry_train)
        ncarry_val = tuple(jnp.swapaxes(c, 0, 1) for c in ncarry_val)

        # Check for NaNs
        if jnp.isnan(truncated_rewards).any():
            raise ValueError("NaN detected in reward data")

        # Shuffle training data using JAX PRNG
        perm = onp.random.permutation(feat_train.shape[0])
        feat_train = feat_train[perm]
        next_feat_train = next_feat_train[perm]
        act_train = act_train[perm]
        rew_train = rew_train[perm]
        aseq_train = aseq_train[perm]
        occ_train = occ_train[perm]
        nocc_train = nocc_train[perm]
        carry_train = tuple(c[perm] for c in carry_train)
        ncarry_train = tuple(c[perm] for c in ncarry_train)

        # Initialize losses
        actor_train_loss = 0.0
        critic_train_loss = 0.0
        temp_train_loss = 0.0

        total_train = feat_train.shape[0]
        num_batches = (total_train + self.batch_size - 1) // self.batch_size

        def _batch_loss_and_update(start_idx, batch_size,
                           feat_shape, next_feat_shape,
                           act_shape, rew_shape, aseq_shape,
                           occ_shape, nocc_shape,
                           carry_shapes, ncarry_shapes):
            # Use standard slicing instead of jax.lax.dynamic_slice
            f_batch = feat_train[start_idx:start_idx + batch_size]
            nf_batch = next_feat_train[start_idx:start_idx + batch_size]
            a_batch = act_train[start_idx:start_idx + batch_size]
            r_batch = rew_train[start_idx:start_idx + batch_size]
            seq_batch = aseq_train[start_idx:start_idx + batch_size]
            occb = occ_train[start_idx:start_idx + batch_size]
            noccb = nocc_train[start_idx:start_idx + batch_size]

            carry_b = tuple(
                c[start_idx:start_idx + batch_size] for c in carry_train
            )
            ncarry_b = tuple(
                c[start_idx:start_idx + batch_size] for c in ncarry_train
            )
            start_time_batch = time.time()
            a_loss, c_loss, t_loss, pred_err = self._calculate_loss_apply_gradients(
                critic_network=critic_network,
                actor_network=actor_network,
                feature_data=f_batch,
                next_feature_data=nf_batch,
                carry=carry_b,
                next_carry=ncarry_b,
                rewards=r_batch,
                actions=a_batch,
                action_sequence=seq_batch,
                occupancy_map=occb,
                next_occupancy_map=noccb,
            )
            return a_loss, c_loss, t_loss, pred_err

        feat_shape = feat_train.shape
        next_feat_shape = next_feat_train.shape
        act_shape = act_train.shape
        rew_shape = rew_train.shape
        aseq_shape = aseq_train.shape
        occ_shape = occ_train.shape
        nocc_shape = nocc_train.shape
        carry_shapes = tuple(c.shape for c in carry_train)  # Convert to tuple
        ncarry_shapes = tuple(c.shape for c in ncarry_train)  # Convert to tuple

        # Process batches
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_train)
            batch_size = end_idx - start_idx
            if batch_size <= 0:
                break  # No more data to process
            a_l, c_l, t_l, pred_err = _batch_loss_and_update(
                start_idx, batch_size,
                feat_shape, next_feat_shape,
                act_shape, rew_shape, aseq_shape,
                occ_shape, nocc_shape,
                carry_shapes, ncarry_shapes
            )

            if self.error_predicted_reward.shape[0] < end_idx:
                pad_len = end_idx - self.error_predicted_reward.shape[0]
                self.error_predicted_reward = jnp.pad(self.error_predicted_reward, (0, pad_len))
            self.error_predicted_reward.at[start_idx:end_idx].set(pred_err.reshape(self.error_predicted_reward[start_idx:end_idx].shape))
            actor_train_loss += a_l / num_batches
            critic_train_loss += c_l / num_batches
            temp_train_loss += t_l / num_batches



        # Validation losses
        actor_val_loss, critic_val_loss, temp_val_loss = self._calculate_loss_validation(
            critic_network_params=critic_network.critic_state.params,
            critic_network=critic_network,
            actor_network_params=actor_network.model_state.params,
            actor_network=actor_network,
            feature_data=feat_val,
            next_feature_data=next_feat_val,
            carry=carry_val,
            next_carry=ncarry_val,
            rewards=rew_val,
            actions=act_val,
            action_sequence=aseq_val,
            occupancy_map=occ_val,
            next_occupancy_map=nocc_val
        )

        # Record metrics
        self.temperature_history.append(actor_network.get_exp_temperature())
        self.training_losses.append((actor_train_loss, critic_train_loss, temp_train_loss))
        self.validation_losses.append((actor_val_loss, critic_val_loss, temp_val_loss))

        # Save periodically
        if self.iteration_counter % 10 == 0:
            jnp.save("training_losses.npy", jnp.array(self.training_losses), allow_pickle=True)
            jnp.save("validation_losses.npy", jnp.array(self.validation_losses), allow_pickle=True)
            jnp.save("temperature_history.npy", jnp.array(self.temperature_history), allow_pickle=True)

        self.iteration_counter += 1
        print(f"Data handling + update time: {time.time() - start:.2f}s")

