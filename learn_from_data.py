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

action_dimension = 2
action_limits = jnp.array([[-0.2, 0.2], [-0.2, 0.2]])




import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple


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
        pos = state[:, :-2, :]
        vel = state[:, -2:, :]

        x = nn.Dense(self.hidden_dim)(pos)

        x = AttentionBlock(self.hidden_dim, self.num_heads)(x, train)
        x = jnp.mean(x, axis=1)
        v = vel.reshape(b * t, -1)
        v = nn.Dense(4)(v)
        v = nn.silu(v)

        return jnp.concatenate([x, v], axis=-1)


class ActorNet(nn.Module):
    preprocessor: Any
    hidden_dim: int = 12
    num_heads: int = 2
    dropout_rate: float = 0.1
    log_std_min: float = -10.0
    log_std_max: float = 0.5

    def setup(self):
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=2)
        self.temperature = self.param(
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(1)), (1,)
        )

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        previous_actions: jnp.ndarray,
        carry: Any = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, Any]:
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), state.shape[:1] + state.shape[2:]
            )
        x = self.preprocessor(state, train=train)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        y = nn.silu(x)
        for i in range(4):
            y = nn.LayerNorm()(y)
            y = nn.Dense(self.hidden_dim)(y)
            y = nn.silu(y)

        mu = nn.Dense(action_dimension)(y)
        mu = jnp.tanh(mu) * 3.0

        log_std = nn.Dense(action_dimension)(y)-1.5
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        o = nn.BatchNorm(use_running_average=not train)(mu)
        return jnp.concatenate([mu, log_std], axis=-1), carry


class CriticNet(nn.Module):
    preprocessor: Any
    hidden_dim: int = 12
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        previous_actions: jnp.ndarray,
        action: jnp.ndarray,
        carry: Any = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.preprocessor(state, train=train)

        a_norm = (action - action_limits[:, 0]) / (
            action_limits[:, 1] - action_limits[:, 0]
        )
        a_norm = nn.Dense(self.hidden_dim)(a_norm)
        sa = jnp.concatenate([x, a_norm], axis=-1)
        sa = nn.Dense(self.hidden_dim)(sa)
        sa = nn.silu(sa)

        def q_net(name: str):
            z = sa
            for i in range(4):
                z = nn.LayerNorm()(z)
                z = nn.Dense(self.hidden_dim, name=f"{name}_fc{i}")(z)
                z = nn.silu(z)
                z = nn.Dropout(self.dropout_rate)(z, deterministic=not train)
            q = nn.Dense(1, name=f"{name}_out")(z)
            return q

        q1 = q_net("q1")
        q2 = q_net("q2")
        y = nn.BatchNorm(use_running_average=not train)(sa)
        return q1, q2
sequence_length = 1
resolution = 253
number_particles = 30
learning_rate = 3e-2

obs = srl.observables.Observable(0)
task = srl.tasks.BallRacingTask()


lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=100,
    decay_rate=0.99,
    staircase=True,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_schedule),
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
)
value_function = srl.value_functions.TDReturnsSAC(gamma=0.99, standardize=False)
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
)

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    minimum_entropy=-action_dimension,
    polyak_averaging_tau=0.05,
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


# protocol.restore_agent(identifier=task.__class__.__name__)
protocol.restore_trajectory(identifier=f"{task.__class__.__name__}_episode_15")
# protocol.actor_network.set_temperature(1e-3)
protocol.set_optimizer(optimizer)
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(engine, 10000, 10)
