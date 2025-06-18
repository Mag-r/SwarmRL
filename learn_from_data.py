import logging
import os
import pathlib

import jax
import numpy as np
import optax
import pint
from flax import linen as nn
from jax import numpy as jnp
from numba import cuda

import swarmrl as srl
from swarmrl.engine.offline_learning import OfflineLearning
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)

cuda.select_device(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

action_dimension = 4
action_limits = jnp.array(
    [[0, 100], [0, 100], [0, 30], [0, 30]]
)


from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class AttentionBlock(nn.Module):
    hidden_dim: int
    num_heads: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(
            y
        )  # Residual

        return y  # Residual

class ParticlePreprocessor(nn.Module):
    hidden_dim: int = 12
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, state: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        b, t, n, d = state.shape
        state = state.reshape(b * t, n, d)
        pos = state[:, :, :]

        x = nn.Dense(self.hidden_dim)(pos)

        pe = self.param(
            "pos_encoding",
            nn.initializers.normal(0.02),
            (pos.shape[1], self.hidden_dim),
        )
        x = x + pe

        x = AttentionBlock(self.hidden_dim, self.num_heads)(x, train)
        x = jnp.mean(x, axis=1)

        return x

class ActorNet(nn.Module):
    """A simple dense model.
    (batch,time,features)
    When dense at beginning, probably flatten is required
    """
    preprocessor: Any = None  # Placeholder for preprocessor
    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=64)
        temperature = self.param(
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(0.01)), (1,)
        )

    @nn.compact
    def __call__(self, x, previous_actions, carry=None, train:bool = False):
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
        mean = self.param("mean", nn.initializers.zeros, (action_dimension,))
        std = self.param("std", lambda key, shape: jnp.full(shape, -1.0), (action_dimension,))
        batch_size= x.shape[0]
        nn.BatchNorm(use_running_average=not train)(x)
        mean = jnp.tile(mean, (batch_size, 1))
        std = jnp.tile(std, (batch_size, 1))
        actor = jnp.concatenate([mean, std], axis=-1)
        return actor, carry


class CriticNet(nn.Module):
    preprocessor: Any  # ParticlePreprocessor
    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=64)

    @nn.compact
    def __call__(self, x, previous_actions, action, carry=None, train:bool = False):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x = self.preprocessor(x, train=train)
        x = jnp.concatenate([x, action], axis=-1)
        q_1 = nn.Dense(features=12, name="Critic_1")(x)
        q_2 = nn.Dense(features=12, name="Critic_2")(x)
        q_1 = nn.silu(q_1)
        q_2 = nn.silu(q_2)

        q_1 = nn.Dense(features=1)(x)
        q_2 = nn.Dense(features=1)(x)
        y = nn.BatchNorm(use_running_average=not train)(q_1)
        return q_1, q_2
sequence_length = 1
resolution = 253
number_particles = 7
learning_rate =5e-4

obs = srl.observables.Observable(0)
task = srl.tasks.ExperimentHexagonTask(number_particles=number_particles)


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
shared_encoder = ParticlePreprocessor()
actor = ActorNet(preprocessor=shared_encoder)
critic = CriticNet(preprocessor=shared_encoder)

# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(
    action_dimension=action_dimension, action_limits=action_limits
)

exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2,
    volatility=0.3,
    action_limits=action_limits,
    action_dimension=action_dimension,
    epsilon=0.0
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
    polyak_averaging_tau=0.05,
    validation_split=0.1,
    fix_temperature=False,
    batch_size=5220,
)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    actor_network=actor_network,
    critic_network=critic_network,
    task=task,
    observable=obs,
    loss=loss,
    max_samples_in_trajectory=200,
)
# Initialize the simulation system

engine = OfflineLearning()


protocol.restore_agent(identifier=task.__class__.__name__)
protocol.restore_trajectory(identifier=f"{task.__class__.__name__}_episode_1_save")
# protocol.actor_network.set_temperature(1e-3)
# protocol.set_optimizer(optimizer)
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(engine, 10000, 10)
