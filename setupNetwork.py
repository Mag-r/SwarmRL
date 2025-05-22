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


class ParticlePreprocessor(nn.Module):
    
    def setup(self):
        self.phi_dense = nn.Dense(32)
        self.com_dense = nn.Dense(32)

    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: shape (batch, time, n_particles, dim)
        Returns:
            features: shape (batch, time, 2 * hidden_dim)
        """
        batch_size, sequence_length, n_particles, dim = state.shape

        com = jnp.mean(state, axis=2) 
        glob = self.com_dense(com.reshape(-1, dim))
        glob = nn.silu(glob)
        glob = glob.reshape(batch_size, sequence_length, -1)

        rel = state - com[:, :, None, :]  
        rel = rel.reshape(-1, n_particles, dim)  

        # Self-attention over particles
        x = nn.Dense(32)(rel)
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(num_heads=4)(x)

        phi = jnp.mean(x, axis=1)
        phi = self.phi_dense(phi)
        phi = nn.silu(phi)
        phi = phi.reshape(batch_size, sequence_length, -1)

        return jnp.concatenate([phi, glob], axis=-1)  # shape: (B, T, 2 * hidden_dim)

class ActorNet(nn.Module):
    """A simple dense model.
    (batch,time,features)
    When dense at beginning, probably flatten is required
    """
    preprocessor: ParticlePreprocessor
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
        state = self.preprocessor(state/253.0)
        
        action_limits_range = action_limits[:, 1] - action_limits[:, 0]
        previous_actions = previous_actions.reshape((batch_size, sequence_length, -1))
        previous_actions = previous_actions / action_limits_range
        previous_actions = nn.Dense(features=32)(previous_actions)
        previous_actions = nn.silu(previous_actions)
        previous_actions = nn.BatchNorm(use_running_average=not train)(previous_actions)

        x = nn.Dense(features=32)(state)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jnp.concatenate([x, previous_actions], axis=-1)
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), state.shape[:1] + state.shape[2:]
            )
        x = nn.SelfAttention(num_heads=8)(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jnp.concatenate([x, state], axis=-1)
        x = x.reshape((batch_size, -1))
        x = nn.Dense(features=32)(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(features=action_dimension)(x)
        std = self.param("std", lambda key, shape: jnp.full(shape, -4.0), (action_dimension,))
        std = jnp.tile(std, (batch_size, 1))
        x = jnp.concatenate([x, std], axis=-1)
        return x, carry


class CriticNet(nn.Module):

    preprocessor: ParticlePreprocessor

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
        action_limits_range = action_limits[:, 1] - action_limits[:, 0]
        state = self.preprocessor(state/253.0)

        
        previous_actions = previous_actions.reshape((batch_size, sequence_length, -1))
        previous_actions = previous_actions / action_limits_range
        previous_actions = nn.Dense(features=32)(previous_actions)
        previous_actions = nn.silu(previous_actions)
        previous_actions = nn.BatchNorm(use_running_average=not train)(previous_actions)
        x = nn.Dense(features=32)(state)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jnp.concatenate([x, previous_actions], axis=-1)
        x = nn.SelfAttention(num_heads=8)(x)
        x = nn.Dense(features=32)(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
        # carry, x = self.lstm(carry, x)
        # x = nn.PReLU()(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.SelfAttention(num_heads=8)(x)
        x = nn.silu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x.reshape((batch_size, -1))
        action = action / action_limits_range
        action = nn.Dense(features=12)(action)
        action = nn.silu(action)
        x = jnp.concatenate([x, action], axis=-1)
        q_1 = nn.Dense(features=32, name="Critic_1_1")(x)
        q_2 = nn.Dense(features=32, name="Critic_2_1")(x)
        q_2 = nn.Dropout(rate=0.2)(q_2, deterministic=not train)
        q_1 = nn.silu(q_1)
        q_2 = nn.silu(q_2)
        q_1 = nn.BatchNorm(use_running_average=not train)(q_1)
        q_2 = nn.BatchNorm(use_running_average=not train)(q_2)
        
        q_1 = nn.Dense(features=16, name="Critic_1_2")(q_1)
        q_2 = nn.Dense(features=16, name="Critic_2_2")(q_2)
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


    shared_encoder = ParticlePreprocessor()
    actor  = ActorNet(preprocessor=shared_encoder)
    critic = CriticNet(preprocessor=shared_encoder)

    

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.05, volatility=0.1, action_dimension=action_dimension, action_limits=action_limits
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
        minimum_entropy=-action_dimension*5,
        polyak_averaging_tau=0.005,
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
