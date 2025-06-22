import os
import pickle

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from tqdm import tqdm

arena = np.load("race_track.npy").astype(np.float32)
arena = (arena - np.min(arena)) / (np.max(arena) - np.min(arena))
arena = cv2.resize(arena, (64, 64))
reference_solution = arena.copy()
arena = nn.max_pool(
    arena[np.newaxis, ..., np.newaxis], window_shape=(2, 2), padding="SAME"
)[0, :, :, 0]
arena = arena.astype(np.float32)

num_agents = 14  # Number of agents
occupancy_map = np.zeros((64, 64), dtype=np.float32)
plt.imshow(arena, cmap="gray")
plt.savefig("arena.png")


def integrate(iterations=1, num_agents=num_agents):
    for _ in range(iterations):
        x = np.random.randint(0, 64, num_agents)
        y = np.random.randint(0, 64, num_agents)
        valid_positions = arena[x, y] < 0.5  # Check if the position is valid
        x = x[valid_positions]
        y = y[valid_positions]
        occupancy_map[x, y] += 1  # Increment the occupancy map at the agent


def edge_pad(x, kernel_size):
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    return jnp.pad(
        x, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge"
    )


class OccupancyMapper(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.clip(x, 0, 1000.0) / 1000.0  # Sicherstellen, dass Input in [0,1] ist
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


def load_train_state(path):
    file_path = os.path.join(path, "train_state.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No train state file found at {file_path}")
    with open(file_path, "rb") as f:
        params, opt_state = pickle.load(f)
    return params, opt_state


n_cells = 64  # Size of the occupancy map
# Initialize the model and optimizer
model = OccupancyMapper()
model_summary = model.tabulate(
    jax.random.PRNGKey(0), jnp.ones(list([1, n_cells, n_cells, 1]))
)
print(model_summary)
params = model.init(jax.random.PRNGKey(0), jnp.ones(list([1, n_cells, n_cells, 1])))
lr_schedule = optax.exponential_decay(
    init_value=5e-3, transition_steps=300, decay_rate=0.95, staircase=True
)
optimizer = optax.adam(learning_rate=lr_schedule)
state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optimizer)
params, opt_state = load_train_state(
    "occupancy_model_checkpoint"
)  # Load the model if it exists
state = state.replace(params=params, opt_state=opt_state)


def conv(reference_solution, occupancy_map, integrate, state, num_agents=num_agents):
    predictions = []
    losses = []
    uncertains = []
    for i in tqdm(range(10000)):
        integrate(iterations=10, num_agents=num_agents)  # Update the occupancy map based on agent movements
        prediction = state.apply_fn(
            {"params": state.params}, jnp.reshape(occupancy_map, (1, 64, 64, 1))
        )
        predictions.append(prediction[0, :, :, 0])
        uncertain = (prediction[0, :, :, 0] < 0.99) & (prediction[0, :, :, 0] > 0.01)

        error = 0.5 * jnp.mean(
            jnp.abs(
                (prediction[0, :, :, 0] > 0.99)
                - reference_solution
                - uncertain.sum() / (64 * 64)
            )
        )  # Calculate the error between prediction and reference solution
        uncertains.append(uncertain.sum() / (64 * 64))
        losses.append(error)
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        ax[0].imshow(prediction[0, :, :, 0], cmap="gray")
        ax[0].set_title("Predicted Arena")
        ax[1].loglog(range(len(losses)), losses, label="BCE Loss")
        ax[1].set_title("Relative Number of wrong classifications")
        ax[1].set_xlabel("Iteration * 10")
        ax[1].set_ylabel("Error")

        ax[2].loglog(range(len(uncertains)), uncertains)
        ax[2].set_title("Uncertain cells [%]")
        ax[2].set_xlabel("Iteration * 10")
        ax[2].set_ylabel("Uncertainty")
        plt.savefig(f"predictions/prediction_{i:03d}_{num_agents}.png")
        plt.close()
        if error < 0.08:
            print(f"Convergence reached at iteration {i}")
            return i

conv_reached=[]
for k in range(1, 30):
    num_agents = k
    for l in range(10):
        print(f"Starting convergence for {num_agents} agents")
        occupancy_map = np.zeros((64, 64), dtype=np.float32)
        iter = conv(reference_solution, occupancy_map, integrate, state, num_agents)
        conv_reached.append(k)
        conv_reached.append(iter)
np.save("convergence_results.npy", np.array(conv_reached))
