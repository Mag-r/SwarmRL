# %%

import logging
import os

import jax
import numpy as np
import optax
from flax import linen as nn
from jax import numpy as jnp
from numba import cuda
from threading import Lock

import swarmrl as srl
from swarmrl.engine.car_benchmark import CarBenchmark
from swarmrl.observables.car_image import CarImage
from swarmrl.tasks.MPI_chain import ChainTask
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)

# %%
action_dimension = 3


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
        self.lstm = self.ScanLSTM(features=24)
        temperature = self.param(
            "temperature", lambda key, shape: jnp.full(shape, 0.0), (1,)
        )

    @nn.compact
    def __call__(self, x, previous_actions, carry=None):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((batch_size, sequence_length, -1))

        # x = jnp.concatenate([x, previous_actions], axis=-1)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Action Net: new carry initialized")
        # carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))

        actor = nn.Dense(features=256, name="Actor_1")(x)
        actor = nn.relu(actor)

        actor = nn.Dense(features=action_dimension * 2, name="Actor_out")(actor)

        return actor, carry


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
        self.lstm = self.ScanLSTM(features=24)

    @nn.compact
    def __call__(self, x, previous_actions, action, carry=None):
        batch_size, sequence_length = x.shape[0], x.shape[1]

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # x = jnp.concatenate([x, previous_actions], axis=-1)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Action Net: new carry initialized")
        # carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))
        x = jnp.concatenate([x, action], axis=-1)

        q_1 = nn.Dense(features=256)(x)
        q_2 = nn.Dense(features=256)(x)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)

        q_1 = nn.Dense(features=1)(q_1)
        q_2 = nn.Dense(features=1)(q_2)
        return q_1, q_2


# %%

# cuda.select_device(0)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lock = Lock()
resolution = 96
learning_rate = 2.5e-4
sequence_length = 2
lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=3000,
    decay_rate=0.95,
    staircase=True,
)
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule, eps=1e-7)


logger.info("Initializing observables and tasks")
obs = CarImage()

# task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
# print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)
task = ChainTask()

actor = ActorNet()
critic = CriticNet()
exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2, volatility=0.3
)


# Define a sampling_strategy
action_limits = jnp.array([[-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).transpose()
print(action_limits.shape)
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(
    action_dimension=action_dimension, action_limits=action_limits
)
# sampling_strategy = srl.sampling_strategies.ExpertKnowledge()
# Value function to use
value_function = srl.value_functions.TDReturnsSAC(gamma=0.95, standardize=True)

actor_network = srl.networks.ContinuousActionModel(
    flax_model=actor,
    optimizer=optimizer,
    input_shape=(
        1,
        sequence_length,
        resolution,
        resolution,
        1,
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
        resolution,
        resolution,
        1,
    ),  # batch implicitly 1 ,time,H,W,channels for conv
    action_dimension=action_dimension,
)

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    minimum_entropy=-action_dimension,
    polyak_averaging_tau=0.005,
    lock=lock,
)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    actor_network=actor_network,
    critic_network=critic_network,
    task=task,
    observable=obs,
    loss=loss,
    lock=lock,
    max_samples_in_trajectory=20,
)
# Initialize the simulation system
total_reward = []

system_runner = CarBenchmark()

# learning_rate = np.random.rand() * np.power(10.0, np.random.randint(-16, -3))
# protocol.restore_agent()
rl_trainer = Trainer([protocol], lock=lock)
print(f"Start training, with learning rate {learning_rate}", flush=True)
if __name__ == "__main__":
    reward = rl_trainer.perform_rl_training(system_runner, 1000, 100)
    total_reward.append(reward)

    np.save("total_reward.npy", total_reward)
