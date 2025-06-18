import numpy as onp
from jax import numpy as jnp
from flax import linen as nn
import setupNetwork
import pysr
import pickle
import optax
import jax

import swarmrl as srl
from matplotlib import pyplot as plt

preprocessor = setupNetwork.ParticlePreprocessor()
critic = setupNetwork.CriticNet(preprocessor=preprocessor)
sequence_length = 1
number_particles = 7
critic_network = srl.networks.ContinuousCriticModel(
    critic_model=critic,
    action_dimension=setupNetwork.action_dimension,
    deployment_mode=True,
    optimizer = optax.adam(learning_rate=0.0),
    input_shape=(
            1,
            sequence_length,
            number_particles,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
)
critic_network.restore_model_state(directory="Models", filename="ActorCriticAgent_0_critic_ExperimentHexagonTask")

filename = "training_data/trajectory_ExperimentHexagonTask_episode_1.pkl"
with open(filename, "rb") as f:
    trajectory_data = pickle.load(f)

state = jnp.array(trajectory_data["feature_sequence"])[:,0,...]

actions = jnp.array(trajectory_data["actions"])
print("State shape:", state.shape, "Actions shape:", actions.shape)

q_1, q_2, *_ = critic_network.compute_q_values_critic(
    params =critic_network.critic_state.params,
    observables= state,
    actions=actions,
    previous_actions=actions,
    carry=0.0,
)
q_values = jnp.min(jnp.stack([q_1, q_2], axis=-1),axis=-1)
print("Q-values shape:", q_values.shape)

pysr_input = jnp.concatenate([state.reshape(state.shape[0], -1), actions], axis=-1)
pysr_model = pysr.PySRRegressor(
    model_selection="best",
    niterations=10,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sin", "cos"],
    maxsize=1000,
    maxdepth=5,
)
pysr_model.fit(pysr_input, q_values)
print("Pysr model:", pysr_model)