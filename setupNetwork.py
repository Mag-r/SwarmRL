import jax
import flax.linen as nn
import swarmrl as srl
import optax
from jax import numpy as jnp
import logging

import os

logger = logging.getLogger(__name__)
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
        x = x.reshape((x.shape[0], -1))
        x = (x-jnp.mean(x, keepdims=True)) / (jnp.std(x, keepdims=True) + 1e-6)
        x= nn.Dense(features=12)(x)
        x = nn.sigmoid(x)
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
        self.lstm = self.ScanLSTM(features=64)

    @nn.compact
    def __call__(self, x, previous_actions, action, carry=None, train:bool = False):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x = x.reshape((batch_size, -1))
        mean = jnp.mean(x, keepdims=True)
        std = jnp.std(x, keepdims=True)
        x = (x - mean) / (std + 1e-6)
        x = x.reshape((batch_size, -1))
        x = jnp.concatenate([x, action], axis=-1)
        q_1 = nn.Dense(features=12, name="Critic_1")(x)
        q_2 = nn.Dense(features=12, name="Critic_2")(x)
        q_1 = nn.sigmoid(q_1)
        q_2 = nn.sigmoid(q_2)
        q_1 = nn.Dropout(rate=0.1)(q_1, deterministic=not train)
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
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.99,
            staircase=True,
        )
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)

    actor = ActorNet()
    critic = CriticNet()
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.2, volatility=0.3
    )
    

    # Define a sampling_strategy
    action_limits = jnp.array([[0,70],[0,70],[0,50], [0,50], [0.01, 5], [-0.8, 0.8], [-0.5, 0.5]])
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)

    value_function = srl.value_functions.TDReturnsSAC(gamma=0.7, standardize=True)
    actor_network = srl.networks.ContinuousActionModel(
        flax_model=actor,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles + 2,
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
            number_particles + 2,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        action_dimension=action_dimension,
    )

    loss = srl.losses.SoftActorCriticGradientLoss(
        value_function=value_function,
        minimum_entropy=-action_dimension*2,
        polyak_averaging_tau=0.05,
        lock=lock,
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
        max_samples_in_trajectory=2000,
        lock=lock
    )
    return protocol
