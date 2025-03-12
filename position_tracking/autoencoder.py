import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import read_position_data
import cv2
from matplotlib import pyplot as plt
import pickle

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
# Load data
(
    ground_truth_positions,
    input_data,
    validation_ground_truth_positions,
    validation_input_data,
) = read_position_data.get_data()
input_data = (input_data > 1.0).astype(float)
validation_input_data = (validation_input_data > 1.0).astype(float)


# Function to convert positions to binary images with larger circles
def positions_to_binary_image(positions, img_size=(216, 216), radius=1):
    images = np.zeros((len(positions), *img_size, 1), dtype=np.float32)
    for i, pos_list in enumerate(positions):
        for x, y in pos_list:  # Assuming each entry is a list of (x, y) tuples
            cv2.circle(images[i, :, :, 0], (int(x / 8), int(y / 8)), radius, 1.0, -1)
    return images


# Convert ground truth positions to binary target images
ground_truth_images = positions_to_binary_image(ground_truth_positions)
validation_ground_truth_images = positions_to_binary_image(
    validation_ground_truth_positions
)


# Convolutional Autoencoder Definition
class Autoencoder(nn.Module):
    @nn.remat
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(16, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)


        x = nn.ConvTranspose(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(16, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(1, (3, 3), strides=(1, 1), padding="SAME")(x)

        return nn.sigmoid(x)  # Output activation


# Weighted Binary Cross Entropy Loss


def weighted_bce_loss(params, apply_fn, batch, target, weight):
    logits = apply_fn(params, batch)
    loss = -(
        weight * target * jnp.log(logits + 1e-8)
        + (1 - target) * jnp.log(1 - logits + 1e-8)
    )
    return jnp.mean(loss)


# Training step
@jax.jit
def train_step(state, batch, target, weight):
    loss, grads = jax.value_and_grad(weighted_bce_loss)(
        state.params, state.apply_fn, batch, target, weight
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


# Create train state
def create_train_state(rng, model):
    dummy_input = jnp.ones((1, 216, 216, 1))
    params = model.init(rng, dummy_input)
    model_summary = model.tabulate(rng, dummy_input)
    print(model_summary)
    optimizer = optax.adam(1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


example_image = validation_input_data[0]

# Training loop
rng = jax.random.PRNGKey(0)
model = Autoencoder()
state = create_train_state(rng, model)

weight = 40  # Adjust this weight based on imbalance
def save_model(state, path):
    with open(path, "wb") as f:
        pickle.dump(state.params, f)

for epoch in range(10000):
    for batch, target in zip(
        np.array_split(input_data, 20), np.array_split(ground_truth_images, 20)
    ):
        state, loss = train_step(state, batch, target, weight)
    logger.info(f"Epoch {epoch+1}, Loss: {loss:.6f}")
    if epoch % 100 == 0:
        validation_loss = weighted_bce_loss(
            state.params,
            state.apply_fn,
            validation_input_data,
            validation_ground_truth_images,
            weight,
        )
        logger.info(f"Validation Loss: {validation_loss:.6f}")
        example_output = state.apply_fn(state.params, example_image[None, ...])
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(example_image[:, :, 0], cmap="gray")
        ax[1].imshow(validation_ground_truth_images[0, :, :, 0], cmap="gray")
        ax[2].imshow(example_output[0, :, :, 0], cmap="gray")
        plt.savefig(f"output_{epoch}.png")
        plt.close()
        save_model(state, f"model_{epoch}.pkl")
