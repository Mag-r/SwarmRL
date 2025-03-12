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
from swarmrl.engine.simple_benchmark import SimpleBenchmark
from swarmrl.observables.obs_for_simple_benchmark import SimpleObservable
from swarmrl.tasks.MPI_chain import ChainTask
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)

# %%
action_dimension = 2


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
        x = x.reshape((batch_size, sequence_length, -1))

        # x = jnp.concatenate([x, previous_actions], axis=-1)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Action Net: new carry initialized")
        carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))

        actor = nn.Dense(features=4, name="Actor_1")(x)
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
        x = x.reshape((batch_size, sequence_length, -1))

        # x = jnp.concatenate([x, previous_actions], axis=-1)
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Target Net: new carry initialized")
        # carry, x = self.lstm(carry, x)
        x = x.reshape((batch_size, -1))
        x = jnp.concatenate([x, action], axis=-1)

        q_1 = nn.Dense(features=4)(x)
        q_2 = nn.Dense(features=4)(x)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)

        q_1 = nn.Dense(features=1)(q_1)
        q_2 = nn.Dense(features=1)(q_2)

        return q_1, q_2


# %%

# cuda.select_device(0)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


learning_rate = 3e-4
sequence_length = 2

lr_schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=300,
    decay_rate=0.99,
    staircase=True,
)
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)

logger.info("Initializing observables and tasks")
obs = SimpleObservable()

task = ChainTask()

actor = ActorNet()
critic = CriticNet()

exploration_policy = srl.exploration_policies.GlobalOUExploration(
    drift=0.2, volatility=0.3
)


# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(
    action_dimension=action_dimension
)
# sampling_strategy = srl.sampling_strategies.ExpertKnowledge()
# Value function to use
value_function = srl.value_functions.TDReturnsSAC(gamma=0.0, standardize=True)

actor_network = srl.networks.ContinuousActionModel(
    flax_model=actor,
    optimizer=optimizer,
    input_shape=(1, 1, 1),  # batch implicitly 1 ,time,H,W,channels for conv
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
    action_dimension=action_dimension,
    deployment_mode=learning_rate == 0.0,
)
critic_network = srl.networks.ContinuousCriticModel(
    critic_model=critic,
    action_dimension=action_dimension,
    optimizer=optimizer,
    input_shape=(1, 1, 1),  # batch implicitly 1 ,time,H,W,channels for conv
)

loss = srl.losses.SoftActorCriticGradientLoss(
    value_function=value_function,
    minimum_entropy=-action_dimension,
    polyak_averaging_tau=0.005,
)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    actor_network=actor_network,
    critic_network=critic_network,
    task=task,
    observable=obs,
    loss=loss,
)
# Initialize the simulation system
total_reward = []

system_runner = SimpleBenchmark()

# learning_rate = np.random.rand() * np.power(10.0, np.random.randint(-16, -3))
# protocol.restore_agent()
rl_trainer = Trainer([protocol])
print(f"Start training, with learning rate {learning_rate}", flush=True)
if __name__ == "__main__":
    reward = rl_trainer.perform_rl_training(system_runner, 10000, 50)
    total_reward.append(reward)

    np.save("total_reward.npy", total_reward)
