# %%

import logging
import os

import jax
import numpy as np
import optax
from flax import linen as nn
from jax import numpy as jnp
from numba import cuda

import swarmrl as srl
from swarmrl.engine.car_benchmark import CarBenchmark
from swarmrl.observables.car_image import CarImage
from swarmrl.tasks.MPI_chain import ChainTask
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)

# %%
action_dimension = 3


class ActoCriticNet(nn.Module):
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
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(0.2)), (1,)
        )

    @nn.compact
    def __call__(self, x, previous_actions, action, carry=None):
        batch_size, sequence_length = x.shape[0], x.shape[1]


        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        x = nn.Conv(features=24, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((batch_size, sequence_length, -1))
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, previous_actions], axis=-1)
        x = nn.LayerNorm()(x)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Action Net: new carry initialized")
        carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))
        x = nn.LayerNorm()(x)

        actor = nn.Dense(features=12, name="Actor_1")(x)
        actor = nn.relu(actor)
        actor = nn.LayerNorm()(actor)

        actor = nn.Dense(features=action_dimension * 2, name="Actor_out")(actor)
        actor = actor.at[:, 3:].set(jnp.log1p(jnp.exp(actor.at[:, 3:].get())))

        if action is not None:
            x = jnp.concatenate([x, action], axis=-1)
            q_1 = nn.Dense(features=12)(x)
            q_2 = nn.Dense(features=12)(x)
            q_1 = nn.relu(q_1)
            q_2 = nn.relu(q_2)

            q_1 = nn.Dense(features=1)(q_1)
            q_2 = nn.Dense(features=1)(q_2)
        else:
            q_1 = None
            q_2 = None

        return actor, q_1, q_2, carry


class TargetNet(nn.Module):
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

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        x = nn.Conv(features=24, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((batch_size, sequence_length, -1))
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, previous_actions], axis=-1)
        x = nn.LayerNorm()(x)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Action Net: new carry initialized")
        carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))
        x = nn.LayerNorm()(x)
        x = jnp.concatenate([x, action], axis=-1)

        q_1 = nn.Dense(features=12)(x)
        q_2 = nn.Dense(features=12)(x)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)

        q_1 = nn.Dense(features=1)(q_1)
        q_2 = nn.Dense(features=1)(q_2)
        return q_1, q_2


# %%


cuda.select_device(0)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


resolution = 96
learning_rate = 1e-4
sequence_length = 4


logger.info("Initializing observables and tasks")
obs = CarImage()

# task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
# print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)
task = ChainTask()

actor_critic = ActoCriticNet()
target = TargetNet()
exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2, volatility=0.3
)

# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution()
# sampling_strategy = srl.sampling_strategies.ExpertKnowledge()
# Value function to use
value_function = srl.value_functions.TDReturnsSAC(gamma=0.99, standardize=True)

network = srl.networks.ContinuousActionModel(
    flax_model=actor_critic,
    optimizer=optax.adam(learning_rate=learning_rate),
    input_shape=(
        10,
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
target_network = srl.networks.ContinuousTargetModel(
    flax_model=target,
    optimizer=optax.adam(learning_rate=learning_rate),
    input_shape=(
        10,
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

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    learning_rate=learning_rate,
    minimum_entropy=-action_dimension,
)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    network=network,
    target_network=target_network,
    task=task,
    observable=obs,
    loss=loss,
)
# Initialize the simulation system
total_reward = []

system_runner = CarBenchmark()

# learning_rate = np.random.rand() * np.power(10.0, np.random.randint(-16, -3))
# protocol.restore_agent()
rl_trainer = Trainer([protocol])
print(f"Start training, with learning rate {learning_rate}", flush=True)
reward = rl_trainer.perform_rl_training(system_runner, 100, 100)
total_reward.append(reward)

np.save("total_reward.npy", total_reward)
