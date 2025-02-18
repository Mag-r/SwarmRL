from matplotlib.animation import FuncAnimation
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

file_path = "trajectory.hdf5"


with h5py.File(file_path, "r") as f:
    pos = f["rafts"]["Unwrapped_Positions"][:]
fig, ax = plt.subplots()
ax.set_xlim(0, 10000)
ax.set_ylim(0, 10000)
scat = ax.scatter(pos[0, :, 0], pos[0, :, 1])



print(f"Creating animation, number of frames {len(pos)}", flush=True)

progress_bar = tqdm(total=len(pos))

def update(frame):
    scat.set_offsets(pos[frame, :, :])
    progress_bar.update(frame)
    print(f"Frame {frame}/{len(pos)}", flush=True)
    return (scat,)

ani = FuncAnimation(fig, update, frames=range(len(pos)), blit=True)
progress_bar.close()
ani.save("trajectory.mp4", writer="ffmpeg", fps=5)
print("Done",flush=True)
