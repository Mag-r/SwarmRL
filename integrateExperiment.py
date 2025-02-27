import numpy as np
from numba import cuda
import pathlib
import logging
import setupNetwork
import os
import pint

from swarmrl.observables.basler_camera_MPI import BaslerCameraObservable
from swarmrl.tasks.experiment_task import ExperimentTask
from swarmrl.engine.gaurav_sim import GauravSim, GauravSimParams
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)
from swarmrl.engine.gaurav_experiment import GauravExperiment


cuda.select_device(0)

# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sequence_length = 4
resolution = 506
action_dimension = 3
number_particles = 7

obs = BaslerCameraObservable([resolution, resolution])
task = ExperimentTask(number_particles=number_particles)

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

# %%
protocol = setupNetwork.defineRLAgent(obs, task, 0.001)

# protocol.restore_agent()
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(experiment, 10000, 100)