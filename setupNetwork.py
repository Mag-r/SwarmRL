import jax
import flax.linen as nn
import swarmrl as srl
import optax
from jax import numpy as jnp

action_dimension = 3
resolution = 506
sequence_length = 4

class ActoCriticNet(nn.Module):
    """A simple dense model.
    (batch,time,features)
    When dense at beginning, probably flatten is required
    """
    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.ConvLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=12, kernel_size=(3, 3))

    @nn.remat
    @nn.compact
    def __call__(self, x, action, carry=None):
        batch_size = x.shape[0]
        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:])
            print("Action: new carry initialized")
        carry, memory = self.lstm(carry, x)

        # Apply relu activation to the LSTM output (memory)
        x = nn.relu(memory)
        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        x = x.reshape((batch_size, -1))
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        # x = nn.LayerNorm()(x)
        actor = nn.Dense(features=12, name="Actor_1")(x)
        actor = nn.relu(actor)
        actor = nn.Dense(features=12, name="Actor_2")(actor)
        actor = nn.relu(actor)
        actor = nn.Dense(features=action_dimension * 2, name="Actor_3")(actor)
        actor = nn.relu(actor)
        if action is not None:
            x = jnp.concatenate([x, action], axis=-1)
            q_1 = nn.Dense(features=12)(x)
            q_2 = nn.Dense(features=12)(x)
            q_1 = nn.relu(q_1)
            q_2 = nn.relu(q_2)
            q_1 = nn.Dense(features=12)(q_1)
            q_2 = nn.Dense(features=12)(q_2)
            # q_1 = q_1 + x
            # q_2 = q_2 + x
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
            nn.ConvLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=12, kernel_size=(3, 3))

    @nn.remat
    @nn.compact
    def __call__(self, x, action, carry = None):
        batch_size = x.shape[0]

        # Initialize carry if it's not provided
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), x.shape[:1] + x.shape[2:]
            )
            print("Target: new carry initialized")
        carry, memory = self.lstm(carry, x)

        # Apply relu activation to the LSTM output (memory)
        x = nn.relu(memory)
        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        x = x.reshape((batch_size, -1))

        x = nn.Dense(features=12)(x)
        x = nn.relu(x)

        x = jnp.concatenate([x, action], axis=-1)
        q_1 = nn.Dense(features=12)(x)
        q_2 = nn.Dense(features=12)(x)
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)
        q_1 = nn.Dense(features=12)(q_1)
        q_2 = nn.Dense(features=12)(q_2)
        # q_1 = q_1 + x
        # q_2 = q_2 + x
        q_1 = nn.relu(q_1)
        q_2 = nn.relu(q_2)
        q_1 = nn.Dense(features=1)(q_1)
        q_2 = nn.Dense(features=1)(q_2)
        return q_1, q_2


def defineRLAgent(obs, task: srl.tasks.Task ,learning_rate: float)->srl.agents.MPIActorCriticAgent:
    # Define the model
    actor_critic = ActoCriticNet()
    target = TargetNet()
    exploration_policy = srl.exploration_policies.GlobalOUExploration(drift=0.2, volatility = 0.003)

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution()

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
    
    loss = srl.losses.SoftActorCriticGradientLoss(value_function=value_function, learning_rate=1E-4, minimum_entropy=-3)

    protocol = srl.agents.MPIActorCriticAgent(
        particle_type=0,
        network=network,
        target_network=target_network,
        task=task,
        observable=obs,
        loss=loss,
    )
    return protocol
