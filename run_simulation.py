# %%
import swarmrl as srl

from swarmrl.observables.top_down_image import TopDownImage
from jax import numpy as np
import matplotlib.pyplot as plt
from swarmrl.components import Colloid
import open3d as o3d
import logging
import flax.linen as nn
from swarmrl.tasks.dummy_task import DummyTask
import optax
from swarmrl.actions.mpi_action import MPIAction
from swarmrl.engine.gaurav_sim import *
from swarmrl.trainers.global_continuous_trainer import GlobalContinuousTrainer as Trainer
import pint


# %%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n'
)


resolution=1280
number_of_gaussians=10
action_dimension=8
rafts = o3d.io.read_triangle_mesh("coloured_rafts.ply")

obs = TopDownImage(
    np.array([10000.0, 10000.0, 0.1]), image_resolution=np.array([resolution]*2), particle_type=0, custom_mesh=rafts, is_2D=True, save_images=False
)
task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)

class imageNet(nn.Module):
    """A simple dense model."""

    @nn.compact
    def __call__(self, x):
        x = nn.avg_pool(x,window_shape=(3,3),strides=(3,3))
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,window_shape=(3,3),strides=(3,3))
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,window_shape=(3,3),strides=(3,3))
        x = x.flatten()
        
        y = nn.Dense(features=256)(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        y = nn.relu(y)

        y = nn.Dense(features=32)(y)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        y = nn.relu(y)
        
        y = nn.Dense(features=1)(y)  # Critic
        x = nn.Dense(features=number_of_gaussians*action_dimension*2)(x)  # Actor
        #pass output designed for variance through relu function (last number_of_gaussians*action_dimension)
        x = x.at[number_of_gaussians * action_dimension:].set(nn.relu(x.at[number_of_gaussians * action_dimension:].get()))
        
        return x, y

class velocityNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        y = nn.Dense(features=1)(x)
        y = nn.relu(y)
        x = nn.Dense(features=number_of_gaussians * action_dimension*2)(x)
        x = x.at[number_of_gaussians * action_dimension :].set(
            nn.relu(x.at[number_of_gaussians * action_dimension :].get())
        )
        return x, y



exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0) # check this

# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution()

# Value function to use
value_function = srl.value_functions.GlobalExpectedReturns(gamma=0.1, standardize=True)

# Define the model
actor_critic = ActoCriticNet()


# %%
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define parameters in SI units
params = GauravSimParams(
            ureg=ureg,
            box_length=Q_(10000, "micrometer"),
            time_step=Q_(5e-3, "second"),
            time_slice=Q_(1, "second"),
            snapshot_interval=Q_(0.002, "second"),
            raft_radius=Q_(150, "micrometer"),
            raft_repulsion_strength=Q_(1e-7, "newton"),
            dynamic_viscosity=Q_(1e-3, "Pa * s"),
            fluid_density=Q_(1000, "kg / m**3"),
            lubrication_threshold=Q_(15, "micrometer"),
            magnetic_constant=Q_(4 * np.pi * 1e-7, "newton /ampere**2"),
            capillary_force_data_path=pathlib.Path(
                "/work/clohrmann/mpi_collab/capillaryForceAndTorque_sym6"
            ),  # TODO
        )

# Initialize the simulation system
system_runner = GauravSim(params=params, out_folder="./", with_precalc_capillary=True,save_h5=True)
mag_mom = Q_(1e-8, "ampere * meter**2")
for i in range(1):
    system_runner.add_colloids(pos = [np.random.rand()*10000,np.random.rand()*10000, 0]* ureg.micrometer, alpha = np.random.rand()*2*np.pi, magnetic_moment = 1E-8* ureg.ampere * ureg.meter**2)


# %%
network = srl.networks.ContinuousFlaxModel(
    flax_model=actor_critic,
    optimizer=optax.adam(learning_rate=0.001),
    input_shape=(1,resolution,resolution,1), #1 are required for CNN
    sampling_strategy=sampling_strategy,
    exploration_policy=exploration_policy,
    number_of_gaussians=number_of_gaussians,
    action_dimension=action_dimension,
)
loss = srl.losses.GlobalPolicyGradientLoss(value_function=value_function)

protocol = srl.agents.MPIActorCriticAgent(
    particle_type=0,
    network=network,
    task=task,
    observable=obs,
    loss=loss,
)
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(system_runner, 200,20)
np.savetxt("rewards.txt",reward)
