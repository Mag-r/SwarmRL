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
from swarmrl.tasks.dummy_task import DummyTask
import optax
from swarmrl.actions.mpi_action import MPIAction
from swarmrl.engine.gaurav_sim import *
from swarmrl.trainers.global_continuous_trainer import GlobalContinuousTrainer as Trainer
import pint
import os
import setupNetwork


# %%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s\n'
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
N_part = 2
resolution = 506
rafts = o3d.io.read_triangle_mesh("modified_raft.ply")

logger.info("Initializing observables and tasks")
obs = TopDownImage(
    np.array([10000.0, 10000.0, 0.1]), batch_size=10, image_resolution=np.array([resolution]*2), particle_type=0, custom_mesh=rafts, is_2D=True, save_images=False
)

task = DummyTask(np.array([10000,10000,0]),target= np.array([5000,5000,0]))
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
system_runner = GauravSim(params=params, out_folder="./", with_precalc_capillary=True,save_h5=True)
mag_mom = Q_(1e-8, "ampere * meter**2")
for i in range(N_part):
    system_runner.add_colloids(pos=[np.random.rand()*10000,np.random.rand()*10000, 0]* ureg.micrometer, alpha=np.random.rand()*2*np.pi, magnetic_moment = 1E-8* ureg.ampere * ureg.meter**2)


# %%
protocol = setupNetwork.defineRLAgent(obs, task, 0.003)

protocol.restore_agent()
rl_trainer = Trainer([protocol])
print("start training", flush=True)
reward = rl_trainer.perform_rl_training(system_runner, 10000, 10)
np.savetxt("rewards.txt", reward)
