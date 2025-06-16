import numpy as np
from jax import numpy as jnp
from numba import cuda
import pathlib
import logging
import setupNetwork
import os
import pint
from flax import linen as nn
import jax

from swarmrl.observables.basler_camera_MPI import BaslerCameraObservable
from swarmrl.tasks.experiment_chain import ExperimentChainTask
from swarmrl.tasks.experiment_hexagon import ExperimentHexagonTask
from swarmrl.engine.gaurav_sim import GauravSim, GauravSimParams
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)
from swarmrl.tasks.ball_movement_task import ExperimentBallMovingTask
from swarmrl.tasks.slam_task import MappingTask
from swarmrl.tasks.ball_race_task import BallRacingTask
from swarmrl.engine.gaurav_experiment import GauravExperiment
from threading import Lock

cuda.select_device(0)

# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(16, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)

        x = nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)

        x = nn.ConvTranspose(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)

        x = nn.ConvTranspose(16, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)

        x = nn.Conv(1, (3, 3), strides=(1, 1), padding="SAME")(x)

        return nn.sigmoid(x)


def edge_pad(x, kernel_size):
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    return jnp.pad(
        x,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='edge'
    )
    
class OccupancyMapper(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.clip(x, 0, 1000.0)*(1.0/1000.0)  # Sicherstellen, dass Input in [0,1] ist
        x = edge_pad(x, (3, 3))
        x = nn.Conv(16, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(32, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(32, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(16, (3, 3), padding="VALID")(x)
        x = nn.silu(x)

        x = edge_pad(x, (3, 3))
        x = nn.Conv(1, (3, 3), padding="VALID")(x)
        x = nn.sigmoid(x)
        return x
sequence_length = 1
resolution = 253

number_particles = 14
learning_rate = 3e-4

lock = Lock()
obs = BaslerCameraObservable(
    [resolution, resolution], Autoencoder(), model_path="Models/autoencoder_5_9.pkl", number_particles=number_particles
)

task = MappingTask(lock=lock, mapper=OccupancyMapper(), model_path="Models/occupancy_mapper_6_2.pkl", resolution=(resolution, resolution))
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define parameters in SI units
params = GauravSimParams(
    ureg=ureg,
    box_length=Q_(10000, "micrometer"),
    time_step=Q_(1e-2, "second"),
    time_slice=Q_(1e-1, "second"),
    snapshot_interval=Q_(0.002, "second"),
    raft_radius=Q_(150, "micrometer"),
    raft_repulsion_strength=Q_(1e-7, "newton"),
    dynamic_viscosity=Q_(1e-3, "Pa * s"),
    fluid_density=Q_(1000, "kg / m**3"),
    lubrication_threshold=Q_(15, "micrometer"),
    magnetic_constant=Q_(4 * np.pi * 1e-7, "newton /ampere**2"),
    capillary_force_data_path=pathlib.Path(
        "/home/gardi/Downloads/spinning_rafts_sim2/2019-05-13_capillaryForceCalculations-sym6/capillaryForceAndTorque_sym6"
    ),
)

# Initialize the simulation system
sim = GauravSim(
    params=params, out_folder="./", with_precalc_capillary=False, save_h5=False
)
experiment = GauravExperiment(sim)

protocol, opt = setupNetwork.defineRLAgent(
    obs, task, learning_rate=learning_rate, sequence_length=sequence_length, lock=lock, number_particles=number_particles
)

protocol.restore_agent(identifier=task.__class__.__name__)
# protocol.restore_trajectory(identifier=f"{task.__class__.__name__}_episode_1")
protocol.set_optimizer(opt)
rl_trainer = Trainer([protocol], lock=lock, deployment_mode=learning_rate == 0.0)
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(experiment, 1000, 10)
