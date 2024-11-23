# %%
from numba import cuda
cuda.select_device(0)

import swarmrl as srl
from swarmrl.observables.top_down_image import TopDownImage
from swarmrl.observables.pos_rotation import PosRotationObservable
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
import os

# %%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n'
)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sequence_length=5
resolution=2024
number_of_gaussians=1
action_dimension=8
N_part = 3
rafts = o3d.io.read_triangle_mesh("coloured_rafts.ply")

# obs = TopDownImage(
#     np.array([10000.0, 10000.0, 0.1]), image_resolution=np.array([resolution]*2), particle_type=0, custom_mesh=rafts, is_2D=True, save_images=True
# )
obs = PosRotationObservable(
    np.array([10000.0, 10000.0, 0.1]), particle_type=0
)

task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)

class ActoCriticNet(nn.Module):
    """A simple dense model.
    (batch,time,features)
    When dense at beginning, probably flatten is required
    """

<<<<<<< HEAD

=======
    features: int = 12  # Number of LSTM features
>>>>>>> multi_network_gaurav

    def setup(self):
        # Define a scanned LSTM cell
        self.ScanLSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

    @nn.compact
    def __call__(self, x, carry=None):
<<<<<<< HEAD
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3))
        # x = nn.Conv(features=32, kernel_size=(3, 3), strides=(3, 3))(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3))

        # Apply ConvLSTM
        lstm = self.ScanLSTM(features= 12, kernel_size=(3, 3))

        # Initialize carry if it's not provided
=======
        # Define the LSTM cell
        lstm = self.ScanLSTM(features=self.features)
>>>>>>> multi_network_gaurav
        if carry is None:
            carry = lstm.initialize_carry(jax.random.PRNGKey(0), x.shape[1:])
            print("new carry initialized")
        carry, memory = lstm(carry, x)
<<<<<<< HEAD

        # Apply relu activation to the LSTM output (memory)
        x = nn.relu(memory)
        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,window_shape=(3,3),strides=(3,3))

        x = nn.Conv(features=12, kernel_size=(3, 3), strides=(3, 3))(x)
        x = nn.relu(x)

        
        
        x = x.flatten()
        # x = x.reshape((sequence_length, -1))
        # memory = memory.reshape((sequence_length, -1))
        # x = memory + x
        # x = x.reshape((sequence_length, -1))
        # x = nn.relu(x)

        x = nn.Dense(features=12)(x)
=======
        x = nn.Dense(features=self.features)(x)
        x = x + memory
>>>>>>> multi_network_gaurav
        x = nn.relu(x)      
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = x.flatten()

<<<<<<< HEAD
        y = nn.Dense(features=12)(x)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        y = nn.relu(y)

=======
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        y = nn.Dense(features=12)(x)
        x = nn.Dense(features=12)(x)
        
        x = nn.relu(x)
        y = nn.relu(y)
        x = nn.LayerNorm()(x)
        y = nn.LayerNorm()(y)
>>>>>>> multi_network_gaurav
        y = nn.Dense(features=12)(y)
        x = nn.Dense(features=12)(x)
        x = nn.relu(x)
        y = nn.relu(y)

        y = nn.Dense(features=1)(y)  # Critic
        x = nn.Dense(features=number_of_gaussians*action_dimension*2 + number_of_gaussians)(x)  # Actor
        # pass output designed for variance through relu function (last number_of_gaussians*action_dimension)
        x = x.at[number_of_gaussians * action_dimension: - number_of_gaussians].set(nn.softplus(x.at[number_of_gaussians * action_dimension: - number_of_gaussians].get()))

        return x, y, carry 


exploration_policy = srl.exploration_policies.RandomExploration(probability=0.0) # check this

# Define a sampling_strategy
sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution()

# Value function to use
value_function = srl.value_functions.TDReturns(gamma=0.9, standardize=True)

# Define the model
actor_critic = ActoCriticNet()


# %%
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define parameters in SI units
params = GauravSimParams(
            ureg=ureg,
            box_length=Q_(10000, "micrometer"),
            time_step=Q_(1e-3, "second"),
            time_slice=Q_(1e-2, "second"),
            snapshot_interval=Q_(0.002, "second"),
            raft_radius=Q_(150, "micrometer"),
            raft_repulsion_strength=Q_(1e-7, "newton"),
            dynamic_viscosity=Q_(1e-3, "Pa * s"),
            fluid_density=Q_(1000, "kg / m**3"),
            lubrication_threshold=Q_(15, "micrometer"),
            magnetic_constant=Q_(4 * np.pi * 1e-7, "newton /ampere**2"),
            capillary_force_data_path=pathlib.Path(
                "/work/clohrmann/mpi_collab/capillaryForceAndTorque_sym6"
            ),
        )

# Initialize the simulation system
system_runner = GauravSim(params=params, out_folder="./", with_precalc_capillary=True,save_h5=True)
mag_mom = Q_(1e-8, "ampere * meter**2")
for i in range(N_part):
    system_runner.add_colloids(pos = [np.random.rand()*10000,np.random.rand()*10000, 0]* ureg.micrometer, alpha = np.random.rand()*2*np.pi, magnetic_moment = 1E-8* ureg.ampere * ureg.meter**2)


# %%
network = srl.networks.ContinuousFlaxModel(
    flax_model=actor_critic,
    optimizer=optax.adam(learning_rate=0.01),
    input_shape=(sequence_length, N_part * 6), #batch implicitly 1 ,time,H,W,channels for conv
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
protocol.restore_agent()
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(system_runner, 10000, 30)
np.savetxt("rewards.txt",reward)
