import numpy as np
import matplotlib.pyplot as plt
import jax
import optax
from flax.training.train_state import TrainState
from jax import numpy as jnp
from flax import linen as nn
from perlin_noise import PerlinNoise
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.checkpoints import save_checkpoint
from flax.core import unfreeze

n_cells = 30


def generate_data(n_cells, seed):
    grid = np.zeros((n_cells, n_cells))
    occupancy_map = np.zeros_like(grid)

    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[0, :] = 1
    grid[-1, :] = 1

    noise = PerlinNoise(octaves=40, seed=seed)
    x, y = np.meshgrid(np.linspace(0, 1, n_cells), np.linspace(0, 1, n_cells))
    noise = np.vectorize(lambda x, y: noise([x, y]))(x, y)
    grid[1:-1, 1:-1] = -((noise[1:-1, 1:-1] < 0.2) - 1)
    x, y = 2, 2
    for i in range(np.random.randint(0, 100000)):
        dx, dy = np.random.randint(-1, 2, size=2)
        if grid[x + dx, y + dy] != 1 and grid[x + 2 * dx, y + 2 * dy] != 1:
            x += dx
            y += dy
        occupancy_map[x, y] += 1
    return grid, occupancy_map


def circle_grid(n_cells, radius):
    grid = np.zeros((n_cells, n_cells))
    occupancy_map = np.zeros_like(grid)
    center = (n_cells // 2, n_cells // 2)
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[0, :] = 1
    grid[-1, :] = 1
    for x in range(n_cells):
        for y in range(n_cells):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius**2:
                grid[x, y] = 1
    x, y = center
    for i in range(np.random.randint(100, 10000)):
        dx, dy = np.random.randint(-1, 2, size=2)
        if grid[x + dx, y + dy] != 1 and grid[x + 2 * dx, y + 2 * dy] != 1:
            x += dx
            y += dy
        occupancy_map[x, y] += 1
    return grid, occupancy_map


input_map = []
output_grid = []
for i in range(100):
    grid, occupancy_map = circle_grid(n_cells, radius=i * 2+2) if i < 2 else generate_data(n_cells, seed=np.random.randint(0, 100000))
    input_map.append(occupancy_map)
    output_grid.append(grid)
    grid, occupancy_map = generate_data(n_cells, seed=i)
    input_map.append(occupancy_map)
    output_grid.append(grid)
input_map = np.array(input_map)
output_grid = np.array(output_grid)


class OccupancyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, (3, 3), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x


# Prepare the data
input_data = input_map[..., None]  # Add batch and channel dimensions
target_data = output_grid[..., None]  # Add batch and channel dimensions

print(input_data.shape)
print(target_data.shape)


# Define a loss function
def binary_cross_entropy_loss(logits, labels):
    return -jnp.mean(
        5 * labels * jnp.log(logits + 1e-7) + (1 - labels) * jnp.log(1 - logits + 1e-7)
    )


def train_step(state, input_data, target_data):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, input_data)
        loss = binary_cross_entropy_loss(logits, target_data)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# Initialize the model and optimizer
model = OccupancyNet()
params = model.init(jax.random.PRNGKey(0), jnp.ones(input_data.shape[1:]))
state = TrainState.create(
    apply_fn=model.apply, params=params["params"], tx=optax.adam(1e-5)
)

# Training loop
for epoch in range(10001):  # Number of epochs
    state, loss = train_step(state, jnp.array(input_data), jnp.array(target_data))
    # print(f"Epoch {epoch + 1}, Loss: {loss}")
    if epoch % 100 == 0:
        print(f"Step {epoch}, Loss: {loss}")

        fig, ax = plt.subplots(3, 3)
        for i in range(3):
            ax[i, 0].imshow(input_data[i+1, ..., 0], cmap="gray")
            ax[i, 0].set_title("Input Map")
            prediction = state.apply_fn(
                {"params": state.params}, input_data[i+1 : i + 2]
            )[0, ..., 0]
            ax[i, 1].imshow(prediction, cmap="hot")
            ax[i, 1].set_title("prediction")
            ax[i, 2].imshow((prediction > 0.8) != (target_data[i+1, ..., 0]), cmap="gray")
            ax[i, 2].set_title(f"Difference {np.sum(target_data[i+1, ..., 0])}")
        plt.savefig(f"occupancy_map_{epoch}.png")
        plt.close(fig)
