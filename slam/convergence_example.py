import os
import pickle

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from flax.training.train_state import TrainState

arena = np.load("slit_arena.npy").astype(np.float32)
arena = (arena - np.min(arena)) / (np.max(arena) - np.min(arena))
arena = cv2.resize(arena, (64, 64))

arena = nn.max_pool(arena[np.newaxis,...,np.newaxis], window_shape=(3, 2), padding="SAME")[0, :, :, 0]
arena = arena.astype(np.float32)
arena.at[:,31:34].set(0)
iterations = 10
num_agents = 14  # Number of agents
occupancy_map = np.zeros((64, 64), dtype=np.float32)
agents = np.random.rand(num_agents, 2) * 10 + 10  # 14 agents in a 64x64 arena
plt.imshow(arena, cmap="gray")
plt.savefig("arena.png")
def integrate():
    for _ in range(iterations):
        x = agents[:, 0].astype(int)
        y = agents[:, 1].astype(int)
        occupancy_map[x, y] += 1

        dx = np.random.randint(-1, 2, size=(num_agents,))
        dy = np.random.randint(-1, 2, size=(num_agents,))

        # Compute proposed new positions
        new_x = np.clip(x + dx, 0, 63)
        new_y = np.clip(y + dy, 0, 63)

        # Only move if not blocked by arena
        can_move = arena[new_x, new_y] != 1
        agents[:, 0] = np.where(can_move, new_x, agents[:, 0])
        agents[:, 1] = np.where(can_move, new_y, agents[:, 1])


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
params, opt_state= load_train_state("occupancy_model_checkpoint")  # Load the model if it exists
state = state.replace(params=params, opt_state=opt_state)


predictions = []
for i in tqdm(range(1000)):
    integrate()  # Update the occupancy map based on agent movements
    prediction = state.apply_fn(
        {"params": state.params}, jnp.reshape(occupancy_map, (1, 64, 64, 1))
    )
    predictions.append(prediction[0, :, :, 0])
    plt.imshow(prediction[0, :, :, 0], cmap="gray")
    plt.savefig(f"predictions/prediction_{i:03d}.png")