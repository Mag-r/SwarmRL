import logging
import os
import time

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

import swarmrl as srl
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import time

logger = logging.getLogger(__name__)
action_dimension = 2
action_limits = jnp.array(
    [[-1.0, 1.0], [-0.7, 0.7]]
)

class ParticlePreprocessor(nn.Module):
    hidden_dim: int = 12
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, state: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        b, t, n, d = state.shape
        pos = state.reshape(b * t, n, d)
        x = nn.Dense(self.hidden_dim)(pos)
        pe = self.param("pos_encoding", nn.initializers.normal(0.02), (pos.shape[1], self.hidden_dim))
        x = x + pe

        x = self.attention_helper(x, train)
        x = jnp.mean(x, axis=1)

        return x
    
    def attention_helper(self, x: jnp.ndarray, train: bool=False) -> jnp.ndarray:
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train
        )(y)
        return y

    
class SmallAttentionUNetEncoder(nn.Module):
    """Nur der Encoder (mit Self-Attention) zur Verarbeitung der Occupancy Map."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.bfloat16)  
        x = jnp.clip(x, 0, 1000.0)* (1.0/1000.0)  
        e1 = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_0")(
            x
        )  # (32,32,16)
        e1 = nn.silu(e1)
        e2 = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_1")(
            e1
        )  # (16,16,32)
        e2 = nn.silu(e2)
        e3 = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_2")(
            e2
        )  # (8, 8,48)
        e3 = nn.silu(e3)

        b = jnp.mean(e3, axis=(1, 2))  # ergibt direkt Shape (B,16)
        return b


class ActorNet(nn.Module):
    """SAC Gaussian actor producing mean & log-std, keeps your original signature."""

    preprocessor: Any  # e.g. ParticlePreprocessor()
    encoder: Any 
    hidden_dim: int = 12
    log_std_min: float = -15.0
    log_std_max: float = 1.0
    dropout_rate: float = 0.1

    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=2)
        temperature = self.param(
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(1E-2)), (1,)
        )

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        occupancy_map: jnp.ndarray,
        previous_actions: jnp.ndarray,
        carry: Any = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Args:
          state:            (batch, time, n_particles, dim)
          previous_actions: (batch, time, action_dim)  # unused atm
        Returns:
          concat of (mu, log_std): shape (batch, action_dim*2), and carry (None)
        """
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), state.shape[:1] + state.shape[2:]
            )
        # 1) preprocess
        x = self.preprocessor(state / 253.0, train)  # (batch, time, hidden_dim)

        occ = self.encoder(occupancy_map.reshape((-1,64,64,1)))  # (batch, hidden_dim)
        x = jnp.concatenate([x, occ], axis=-1)  # (batch, t*h + hidden_dim)
        x = nn.LayerNorm()(x)  # (batch, t*h + hidden_dim)
        x = nn.Dense(self.hidden_dim)(x)  # (batch, hidden_dim)
        y = nn.silu(x)  # (batch, hidden_dim)
        for _ in range(4):
            y = nn.LayerNorm()(y)
            y = nn.Dense(self.hidden_dim)(y)
            y = nn.silu(y)


        mu = nn.Dense(action_dimension)(y)
        log_std = nn.Dense(action_dimension)(y)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        out = jnp.concatenate([mu, log_std], axis=-1)
        return out, carry


class CriticNet(nn.Module):
    """Twin Q-networks for SAC, keeps your original signature."""

    preprocessor: Any  # e.g. ParticlePreprocessor()
    encoder: Any
    hidden_dim: int = 12
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        occupancy_map: jnp.ndarray,
        previous_actions: jnp.ndarray,
        action: jnp.ndarray,
        carry: Any = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          state:            (batch, time, n_particles, dim)
          previous_actions: (batch, time, action_dim)  # unused atm
          action:           (batch, action_dim)
        Returns:
          q1, q2 each of shape (batch, 1)
        """
        # 1) preprocess
        x = self.preprocessor(state / 253.0, train)  # (batch, time, hidden_dim)

        occ = self.encoder(occupancy_map.reshape((-1,64,64,1)))  # (batch, hidden_dim)
        a_norm = action - jnp.array(action_limits[:, 0])  
        a_norm = a_norm / (jnp.array(action_limits[:, 1]) - jnp.array(action_limits[:, 0]))

        sa = jnp.concatenate([x, a_norm, occ], axis=-1)
        sa = nn.Dense(self.hidden_dim)(sa)  # (batch, hidden_dim)
        sa = nn.silu(sa)  # (batch, hidden_dim)

        def q_net(name: str, state_action: jnp.ndarray = sa) -> jnp.ndarray:
            z = state_action
            for i in range(4):
                z = nn.LayerNorm()(z)
                z = nn.Dense(self.hidden_dim, name=f"{name}_fc{i}")(z)
                z = nn.silu(z)
                z = nn.Dropout(self.dropout_rate)(z, deterministic=not train)

            q = nn.Dense(1, name=f"{name}_out")(z)
            return q

        q1 = q_net("q1", sa)
        q2 = q_net("q2", sa)
        return q1, q2


def defineRLAgent(
    obs,
    task: srl.tasks.Task,
    learning_rate: float,
    resolution: int = 506,
    sequence_length: int = 4,
    number_particles: int = 7,
    lock=None,
) -> tuple[srl.agents.MPIActorCriticAgent, optax.GradientTransformation]:

    # Define the model

    if learning_rate == 0.0:
        logger.info("Deployment mode")
        optimizer = optax.adam(learning_rate=0.0)
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=1000,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(
                1.0
            ),  # Gradient clipping with a maximum norm of 1.0
            optax.adam(learning_rate=lr_schedule),
        )

    shared_encoder = ParticlePreprocessor()
    occupancy_encoder = SmallAttentionUNetEncoder()
    actor = ActorNet(preprocessor=shared_encoder, encoder=occupancy_encoder)
    critic = CriticNet(preprocessor=shared_encoder, encoder=occupancy_encoder)

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(
        action_dimension=action_dimension, action_limits=action_limits
    )
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.12,
        volatility=0.08,
        action_dimension=action_dimension,
        action_limits=action_limits,
    )

    value_function = srl.value_functions.TDReturnsSAC(gamma=0.99, standardize=False)
    actor_network = srl.networks.ContinuousActionModel(
        flax_model=actor,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
        action_dimension=action_dimension,
        deployment_mode=learning_rate == 0.0,
        rng_key=jax.random.PRNGKey(int(time.time())),
    )
    critic_network = srl.networks.ContinuousCriticModel(
        critic_model=critic,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        action_dimension=action_dimension,
        rng_key=jax.random.PRNGKey(int(time.time())),
    )

    loss = srl.losses.SoftActorCriticGradientLoss(
        value_function=value_function,
        minimum_entropy=-action_dimension*5,
        polyak_averaging_tau=0.2,
        lock=lock,
        validation_split=0.01,
        fix_temperature=False,
        batch_size=256,
    )

    protocol = srl.agents.MPIActorCriticAgent(
        particle_type=0,
        actor_network=actor_network,
        critic_network=critic_network,
        task=task,
        observable=obs,
        loss=loss,
        max_samples_in_trajectory=10000,
        lock=lock,
    )
    # protocol.set_optimizer(optimizer)
    task.set_agent(protocol)
    return protocol, optimizer
