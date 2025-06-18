import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import pre
import cv2
from matplotlib import pyplot as plt
import pickle
from functools import partial

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
# Load data
scale=1
def load_and_split_data(scale, start_index=0, batch_size=200):
    input_data = np.load("detected_images.npy")[start_index:start_index+batch_size,::scale,::scale]
    ground_truth_positions = np.load("detected_centers.npy")[start_index:start_index+batch_size]

    validation_input_data = input_data[-2:]
    validation_ground_truth_positions = ground_truth_positions[-2:]
    input_data = input_data[:-2]
    ground_truth_positions = ground_truth_positions[:-2]

    def positions_to_binary_image(positions, img_size=(int(253/scale), int(253/scale)), radius=1):
        images = np.zeros((len(positions), *img_size, 1), dtype=np.float32)
        for i, pos_list in enumerate(positions):
            for x, y in pos_list:  # Assuming each entry is a list of (x, y) tuples
                cv2.circle(images[i, :, :, 0], (int(x/scale), int(y/scale)), radius, 1.0, -1)
        return images


# Convert ground truth positions to binary target images
    ground_truth_images = positions_to_binary_image(ground_truth_positions)
    validation_ground_truth_images = positions_to_binary_image(
        validation_ground_truth_positions
    )
    # print(f"shape of input data: {input_data.shape}, ground truth images: {ground_truth_images.shape}")
    return input_data,validation_input_data,ground_truth_images,validation_ground_truth_images




# Convolutional Autoencoder Definition
class Autoencoder(nn.Module):
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

@partial(jax.jit, static_argnums=(1,))
def weighted_bce_loss(params, apply_fn, batch, target, weight):
    logits = apply_fn(params, batch)
    loss = -(
        weight * target * jnp.log(logits + 1e-6)
        + (1 - target) * jnp.log(1 - logits + 1e-6)
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
    dummy_input = jnp.ones((1, int(253/scale), int(253/scale), 3))
    params = model.init(rng, dummy_input)
    model_summary = model.tabulate(rng, dummy_input)
    print(model_summary)
    lr_schedule = optax .schedules.exponential_decay(1e-5, 100, 0.9)
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


_, validation_input_data, _, validation_ground_truth_images = load_and_split_data(scale, start_index=0, batch_size=100)
example_image = validation_input_data[1]
plot_image = example_image.copy()
plot_image[:,:,0], plot_image[:,:,2] = plot_image[:,:,2], plot_image[:,:,0]  # Swap channels for visualization
# Modify data loading to process 200 images at a time
batch_size = 50

rng = jax.random.PRNGKey(0)
model = Autoencoder()
state = create_train_state(rng, model)

weight = np.ones_like(validation_ground_truth_images[0])
weight = 10 # Adjust this weight based on imbalance

def save_model(state, path):
    with open(path, "wb") as f:
        pickle.dump(state.params, f)

def load_model(path):
    with open(path, "rb") as f:
        logger.info(f"Loading model from {path}")
        return pickle.load(f)

loaded_params = load_model("autoencoder_model/autoencoder_hex.pkl")
state = state.replace(params=loaded_params)
training_losses = []
validation_losses = []
try:
    for epoch in range(10000):
        losses = 0
        for start_idx in range(0, 50, batch_size):
            batch,_,target,_ = load_and_split_data(scale, start_index=start_idx, batch_size=batch_size)
            for _ in range(1):  # Perform 5 training steps per batch
                state, loss = train_step(state, batch, target, weight)
                losses += loss
        logger.info(f"Epoch {epoch+1}, Loss: {losses:.6f}")
        if np.isnan(loss):
            raise ValueError("Loss is NaN. Reduce learning rate.")
        validation_loss = weighted_bce_loss(
                state.params,
                state.apply_fn,
                validation_input_data,
                validation_ground_truth_images,
                weight,
            )
        training_losses.append(losses)
        validation_losses.append(validation_loss)
        plt.figure()
        plt.loglog(training_losses, label="Training Loss")
        plt.loglog(validation_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("autoencoder_model/losses.png")
        plt.close()
        if epoch % 10 == 0:
            logger.info(f"Validation Loss: {validation_loss:.6f}")
            example_output = state.apply_fn(state.params, example_image[None, ...])
            if np.max(example_output) < 0.1:
                raise ValueError("Model is not learning, output is too low.")
            fig, ax = plt.subplots(1, 3)

            ax[0].imshow((plot_image-np.min(plot_image)) / (np.max(plot_image)-np.min(plot_image)))
            ax[0].set_title("Input Image")
            ax[1].imshow(validation_ground_truth_images[1, :, :, 0], cmap="hot")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(example_output[0, :, :, 0], cmap="hot")
            ax[2].set_title("Reconstructed Image")
            for axis in ax:
                axis.set_xticks([])
                axis.set_yticks([])
            plt.savefig(f"autoencoder_model/output_{epoch}.png")
            plt.close()
            save_model(state, f"autoencoder_model/model_{epoch}.pkl")
finally:
    save_model(state, "autoencoder_model/final_model.pkl")
    plt.figure()
    plt.loglog(training_losses, label="Training Loss")
    plt.loglog(validation_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("autoencoder_model/losses.png")
    plt.close()

