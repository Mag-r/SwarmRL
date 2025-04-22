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

action_dimension = 7

class ActorNet(nn.Module):
    """A simple dense model.
    (batch,time,features)
    When dense at beginning, probably flatten is required
    """
 
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
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(0.01)), (1,)
        )

    @nn.compact
    def __call__(self, x, previous_actions, carry=None, train:bool = False):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x = x.reshape((batch_size, sequence_length, -1))
        x = x/253
        x = x.at[:, :, :-2].set(x[:, :, :-2] - jnp.expand_dims(x[:, :, -1], axis=-1))
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
        x = x.reshape((batch_size, -1))
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(features=action_dimension*2)(x)
        return x, carry


class CriticNet(nn.Module):
    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=32)

    @nn.compact
    def __call__(self, x, previous_actions, action, carry=None, train:bool = False):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x = x.reshape((batch_size,sequence_length, -1))
        x = x/253 
        x = x.at[:, :, :-2].set(x[:, :, :-2] - jnp.expand_dims(x[:, :, -1], axis=-1))
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
        # carry, x = self.lstm(carry, x)
        # x = nn.sigmoid(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.SelfAttention(num_heads=11)(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x.reshape((batch_size, -1))
        x = jnp.concatenate([x, action], axis=-1)
        q_1 = nn.Dense(features=32, name="Critic_1_1")(x)
        q_2 = nn.Dense(features=32, name="Critic_2_1")(x)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)
        q_1 = nn.Dropout(rate=0.3)(q_1, deterministic=not train)
        q_2 = nn.Dropout(rate=0.4)(q_2, deterministic=not train)
        q_1 = nn.BatchNorm(use_running_average=not train)(q_1)
        q_2 = nn.BatchNorm(use_running_average=not train)(q_2)
        q_1 = nn.Dense(features=24, name="Critic_1_2")(q_1)
        q_2 = nn.Dense(features=24, name="Critic_2_2")(q_2)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)
        q_1 = nn.BatchNorm(use_running_average=not train)(q_1)
        q_2 = nn.BatchNorm(use_running_average=not train)(q_2)
        q_1 = nn.Dense(features=24, name="Critic_1_3")(q_1)
        q_2 = nn.Dense(features=24, name="Critic_2_3")(q_2)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)
        q_1 = nn.BatchNorm(use_running_average=not train)(q_1)
        q_2 = nn.BatchNorm(use_running_average=not train)(q_2)
        q_1 = nn.Dense(features=1)(x)
        q_2 = nn.Dense(features=1)(x)
        return q_1, q_2

sequence_length = 2
resolution = 253
number_particles = 30
learning_rate = 3e-3

obs = srl.observables.Observable(0)
task = srl.tasks.ExperimentBallMovingTask()



lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=100,
    decay_rate=0.99,
    staircase=True,
)
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)

actor = ActorNet()
critic = CriticNet()


# Define a sampling_strategy
action_limits = jnp.array([[0,70],[0,70],[0,50], [0,50], [1, 5], [-0.8, 0.8], [-0.5, 0.5]])
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)

exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2, volatility=0.3, action_limits=action_limits, action_dimension=action_dimension
)
value_function = srl.value_functions.TDReturnsSAC(gamma=0.9, standardize=True)
actor_network = srl.networks.ContinuousActionModel(
    flax_model=actor,
    optimizer=optax.inject_hyperparams(optax.adam)(learning_rate=1E-5),
    input_shape=(
        1,
        sequence_length,
        number_particles + 3,
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
        number_particles + 3,
        2,
    ),  # batch implicitly 1 ,time,H,W,channels for conv
    action_dimension=action_dimension,
)

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    minimum_entropy=-action_dimension,
    polyak_averaging_tau=0.005,
    validation_split=0.1,
    fix_temperature=False,
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
protocol.restore_trajectory(identifier=f"{task.__class__.__name__}_episode_11")
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(engine, 1000, 10)
