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
action_limits = jnp.array([[0,70],[0,70],[0,30], [0,30], [-0.8, 0.8], [-0.5, 0.5]])


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
            "temperature", lambda key, shape: jnp.full(shape, jnp.log(1)), (1,)
        )

    @nn.compact
    def __call__(self, state, previous_actions, carry=None, train:bool = False):
        batch_size, sequence_length = state.shape[0], state.shape[1]
        state = state.reshape((batch_size, sequence_length, -1))
        state = state/253


        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), state.shape[:1] + state.shape[2:]
            )
        x = nn.Dense(features=64)(state)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x.reshape((batch_size, -1))
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
    def __call__(self, state, previous_actions, action, carry=None, train:bool = False):
        batch_size, sequence_length = state.shape[0], state.shape[1]
        state = state.reshape((batch_size,sequence_length, -1))
        state = state/253 
        x = nn.Dense(features=64)(state)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
        # carry, x = self.lstm(carry, x)
        # x = nn.PReLU()(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        action_limits_range = action_limits[:, 1] - action_limits[:, 0]
        action = action / action_limits_range
        x = x.reshape((batch_size, -1))
        x = jnp.concatenate([x, action], axis=-1)
        x = nn.Dense(features=64)(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        q_1 = nn.Dense(features=32, name="Critic_1_1")(x)
        q_2 = nn.Dense(features=32, name="Critic_2_1")(x)
        q_1 = nn.silu(q_1)
        q_2 = nn.silu(q_2)
        q_1 = nn.BatchNorm(use_running_average=not train)(q_1)
        q_2 = nn.BatchNorm(use_running_average=not train)(q_2)

        q_1 = nn.Dense(features=1)(x)
        q_2 = nn.Dense(features=1)(x)
        return q_1, q_2

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
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)

    actor = ActorNet()
    critic = CriticNet()
    

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.03, volatility=0.1, action_dimension=action_dimension, action_limits=action_limits
    )

    value_function = srl.value_functions.TDReturnsSAC(gamma=0.8, standardize=True)
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
        minimum_entropy=-action_dimension,
        polyak_averaging_tau=0.05,
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
        max_samples_in_trajectory=20000,
        lock=lock
    )
    return protocol
