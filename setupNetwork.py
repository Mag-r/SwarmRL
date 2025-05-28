import jax
import flax.linen as nn
import swarmrl as srl
import optax
from jax import numpy as jnp
import logging
import time
import os

logger = logging.getLogger(__name__)
action_dimension = 6
action_limits = jnp.array([[0,70],[0,70],[0,15], [0,15], [-0.8, 0.8], [-0.5, 0.5]])




import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any

class ParticlePreprocessor(nn.Module):
    """Embed each particle, apply self-attention over the n_particles dimension,
       and pool to a per-time-step feature."""
    hidden_dim: int = 64
    num_heads: int = 8

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            state: shape (batch, time, n_particles, dim)
        Returns:
            features: shape (batch, time, hidden_dim)
        """
        # Separate velocity from position
        position = state[:, :, :-2, :]  # Exclude last two entries (velocity)
        velocity = state[:, :, -2:, :]  # Only last two entries (velocity)

        # Process position
        b, t, n, d = position.shape
        x = position.reshape(b * t, n, d)
        x = nn.Dense(self.hidden_dim)(x)          # (b*t, n, hidden_dim)
        x = nn.silu(x)
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
        )(x)                                 # (b*t, n, hidden_dim)
        x = nn.silu(x)
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)                   # (b*t, hidden_dim)
        position_features = x.reshape(b, t, self.hidden_dim)   # (batch, time, hidden_dim)

        # Process velocity
        velocity = velocity.reshape(b * t, -1)  # Flatten velocity
        velocity_features = nn.Dense(8)(velocity)
        velocity_features = nn.silu(velocity_features)
        velocity_features = nn.LayerNorm()(velocity_features)
        velocity_features = velocity_features.reshape(b, t, -1)

        # Concatenate position and velocity features
        return jnp.concatenate([position_features, velocity_features], axis=-1)


class ActorNet(nn.Module):
    """SAC Gaussian actor producing mean & log-std, keeps your original signature."""
    preprocessor: Any  # e.g. ParticlePreprocessor()
    hidden_dims: Tuple[int, ...] = (128, 128)
    log_std_min: float = -15.0
    log_std_max: float = 1.0
    
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
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(1)), (1,)
        )


    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 previous_actions: jnp.ndarray,
                 carry: Any = None,
                 train: bool = False
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
        x = self.preprocessor(state / 253.0)            # (batch, time, hidden_dim)

        # 2) flatten time & features
        b, t, h = x.shape
        x = x.reshape(b, t * h)

        # 3) MLP
        for hd in self.hidden_dims:
            x = nn.Dense(hd)(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)

        # 4) outputs
        mu      = nn.Dense(action_dimension)(x)
        log_std = nn.Dense(action_dimension)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        # 5) concat so you still return a single tensor and carry
        out = jnp.concatenate([mu, log_std], axis=-1)  # (batch, action_dim*2)
        y = nn.BatchNorm(use_running_average=not train)(out)  # (batch, action_dim*2)
        return out, carry


class CriticNet(nn.Module):
    """Twin Q-networks for SAC, keeps your original signature."""
    preprocessor: Any  # e.g. ParticlePreprocessor()
    hidden_dims: Tuple[int, ...] = (128, 64)

    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 previous_actions: jnp.ndarray,
                 action: jnp.ndarray,
                 carry: Any = None,
                 train: bool = False
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
        x = self.preprocessor(state / 253.0)            # (batch, time, hidden_dim)
        b, t, h = x.shape
        x = x.reshape(b, t * h)                       # (batch, t*h)
        action_limits_range = action_limits[:, 1] - action_limits[:, 0]
        action = action / action_limits_range          # (batch, action_dim)

        # 2) concatenate action
        sa = jnp.concatenate([x, action], axis=-1)    # (batch, t*h + action_dim)

        # 3) first Q
        q1 = sa
        for hd in self.hidden_dims:
            q1 = nn.Dense(hd, name=f"q1_fc{hd}")(q1)
            q1 = nn.silu(q1)
        q1 = nn.Dense(1, name="q1_out")(q1)

        # 4) second Q
        q2 = sa
        for hd in self.hidden_dims:
            q2 = nn.Dense(hd, name=f"q2_fc{hd}")(q2)
            q2 = nn.silu(q2)
        q2 = nn.Dense(1, name="q2_out")(q2)
        y = nn.BatchNorm(use_running_average=not train)(q1)  # (batch, 1)
        return q1, q2
    
def defineRLAgent(
    obs, task: srl.tasks.Task, learning_rate: float, resolution: int=506, sequence_length: int=4, number_particles: int = 7, lock=None
) -> srl.agents.MPIActorCriticAgent:
    # Define the model
    
    if learning_rate == 0.0:
        logger.info("Deployment mode")
        optimizer = optax.adam(learning_rate=0.0)
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=10000,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping with a maximum norm of 1.0
            optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)
        )


    shared_encoder = ParticlePreprocessor()
    actor  = ActorNet(preprocessor=shared_encoder)
    critic = CriticNet(preprocessor=shared_encoder)

    

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.08, volatility=0.001, action_dimension=action_dimension, action_limits=action_limits
    )

    value_function = srl.value_functions.TDReturnsSAC(gamma=0.9, standardize=False)
    actor_network = srl.networks.ContinuousActionModel(
        flax_model=actor,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles + 4,
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
            number_particles + 4,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        action_dimension=action_dimension,
        rng_key=jax.random.PRNGKey(int(time.time())),
    )

    loss = srl.losses.SoftActorCriticGradientLoss(
        value_function=value_function,
        minimum_entropy=-action_dimension*2,
        polyak_averaging_tau=0.02,
        lock=lock,
        validation_split=0.01,
        fix_temperature=False,
        batch_size=1012,
    )

    protocol = srl.agents.MPIActorCriticAgent(
        particle_type=0,
        actor_network=actor_network,
        critic_network=critic_network,
        task=task,
        observable=obs,
        loss=loss,
        max_samples_in_trajectory=1000,
        lock=lock
    )
    return protocol
