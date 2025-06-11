import numpy as np
from numba import cuda
import pathlib
import logging
import os
import pint
from flax import linen as nn

from swarmrl.engine.offline_learning import OfflineLearning
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)
import jax
import swarmrl as srl
import optax
from jax import numpy as jnp

cuda.select_device(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

action_dimension = 2
action_limits = jnp.array(
    [[-1.0, 1.0], [-0.7, 0.7]]
)


import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any



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

        def q_net(name: str):
            y = sa
            for i in range(4):
                z = nn.LayerNorm()(y)
                z = nn.Dense(self.hidden_dim, name=f"{name}_fc{i}")(z)
                z = nn.silu(z)
                z = nn.Dropout(self.dropout_rate)(z, deterministic=not train)

            q = nn.Dense(1, name=f"{name}_out")(y)
            return q

        q1 = q_net("q1")
        q2 = q_net("q2")
        return q1, q2



def edge_pad(x, kernel_size):
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    return jnp.pad(
        x,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='edge'
    )

class OccupancyMapper(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.clip(x, 0, 1000.0)/1000.0  # Sicherstellen, dass Input in [0,1] ist
        x = edge_pad(x, (3, 3))
        x = nn.Conv(16, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(32, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(32, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(16, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(1, (3, 3), padding="VALID")(x)
        x = nn.sigmoid(x)
        return x
sequence_length = 1
resolution = 253
number_particles = 14
learning_rate =5e-3

obs = srl.observables.Observable(0)
task = srl.tasks.MappingTask(OccupancyMapper(), model_path="Models/occupancy_mapper_6_2.pkl", resolution=(resolution, resolution))



lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=100,
    decay_rate=0.99,
    staircase=True,
)
optimizer = optax.chain(
        optax.clip_by_global_norm(
            1.0
        ),  # Gradient clipping with a maximum norm of 1.0
        optax.adam(learning_rate=lr_schedule),
    )


shared_preprocessor = ParticlePreprocessor()
shared_encoder = SmallAttentionUNetEncoder()
actor  = ActorNet(preprocessor=shared_preprocessor, encoder=shared_encoder)
critic = CriticNet(preprocessor=shared_preprocessor, encoder=shared_encoder)


# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)

exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2, volatility=0.3, action_limits=action_limits, action_dimension=action_dimension
)
value_function = srl.value_functions.TDReturnsSAC(gamma=0.99, standardize=True)
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
)

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    minimum_entropy=-action_dimension,
    polyak_averaging_tau=0.1,
    validation_split=0.1,
    fix_temperature=False,
    batch_size=1024,
)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    actor_network=actor_network,
    critic_network=critic_network,
    task=task,
    observable=obs,
    loss=loss,
    max_samples_in_trajectory=20000,
)
# Initialize the simulation system

engine = OfflineLearning()



protocol.restore_agent(identifier=task.__class__.__name__)
protocol.restore_trajectory(identifier=f"{task.__class__.__name__}_episode_5")
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(engine, 10000, 10)
