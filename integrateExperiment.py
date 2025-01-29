import numpy as np
from jax import numpy as jnp
import swarmrl as srl

# %%
from numba import cuda

cuda.select_device(0)

import swarmrl as srl
from swarmrl.observables.basler_camera_MPI import BaslerCameraObservable
from jax import numpy as np
import logging
import flax.linen as setupNetwork
from swarmrl.tasks.dummy_task import DummyTask
from swarmrl.engine.gaurav_sim import *
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)
import pint
import os
import setupNetwork
from swarmrl.engine.gaurav_experiment import GauravExperiment
import setupNetwork


# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
host = "localhost"
port = 9090
sequence_length = 4
resolution = 506
number_of_gaussians = 1
action_dimension = 8


obs = BaslerCameraObservable(particle_type=0)
task = DummyTask(np.array([10000, 10000, 0]), target=np.array([5000, 5000, 0]))
print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)


# %%
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
        "/work/clohrmann/mpi_collab/capillaryForceAndTorque_sym6"
    ),
)

# Initialize the simulation system
sim = GauravSim(
    params=params, out_folder="./", with_precalc_capillary=False, save_h5=False
)

experiment = GauravExperiment(sim)

# %%
protocol = setupNetwork.defineRLAgent(obs, task, 0.0)

# protocol.restore_agent()
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(experiment, 10000, 10)

