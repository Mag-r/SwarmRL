import logging
import os
import pathlib

import numpy as np
import open3d as o3d
import pint
from numba import cuda

import setupNetwork
from swarmrl.engine.gaurav_sim import GauravSim, GauravSimParams
from swarmrl.observables.top_down_image import TopDownImage
from swarmrl.tasks.dummy_task import DummyTask
from swarmrl.tasks.MPI_chain import ChainTask
from swarmrl.trainers.global_continuous_trainer import (
    GlobalContinuousTrainer as Trainer,
)

cuda.select_device(0)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n",
)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

N_part = 4
resolution = 506
learning_rate = 1E-3
sequence_length = 4


logger.info("Initializing observables and tasks")
rafts = o3d.io.read_triangle_mesh("modified_raft.ply")
obs = TopDownImage(
    np.array([10000.0, 10000.0, 0.1]),
    image_resolution=np.array([resolution] * 2),
    particle_type=0,
    custom_mesh=rafts,
    is_2D=True,
    save_images=False,
)

# task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
# print(f"task initialized, with normalization = {task.get_normalization()}", flush=True)
task = ChainTask()

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.define("Gauss = 1e-4 * tesla = G")
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
# /home/gardi/Downloads/spinning_rafts_sim2/2019-05-13_capillaryForceCalculations-sym6

# Initialize the simulation system
total_reward = []
for j in range(1, 100):
    system_runner = GauravSim(
        params=params, out_folder="./", with_precalc_capillary=True, save_h5=True
    )
    mag_mom = Q_(1e-8, "ampere * meter**2")
    for _ in range(N_part):
        system_runner.add_colloids(
            pos=[np.random.rand() * 8000 + 1000, np.random.rand() * 8000 + 1000, 0]
            * ureg.micrometer,
            alpha=np.random.rand() * 2 * np.pi,
            magnetic_moment=1e-8 * ureg.ampere * ureg.meter**2,
        )
    # learning_rate = np.random.rand() * np.power(10.0, np.random.randint(-16, -3))
    protocol = setupNetwork.defineRLAgent(
        obs,
        task,
        learning_rate=learning_rate,
        resolution=resolution,
        sequence_length=sequence_length,
    )
    protocol.restore_agent() if j > 1 else None
    rl_trainer = Trainer([protocol])
    print(f"Start training, with learning rate {learning_rate}", flush=True)
    reward = rl_trainer.perform_rl_training(system_runner, 40, 10)
    total_reward.append(reward)
    logger.info(
        f"Resetting System, reward for this episode: {np.sum(reward)}, average reward is: {np.sum(total_reward)/j}, with learning rate: {learning_rate}"
    )
np.save("total_reward.npy", total_reward)
