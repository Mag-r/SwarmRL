import logging
import os
import pickle
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.core import unfreeze
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from jax import numpy as jnp

import flax.serialization
import os
import cv2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=100):
    occupancy_map = np.zeros((n_cells, n_cells), dtype=np.float32)
    input_maps = np.zeros((n_cells, n_cells), dtype=np.float32)[np.newaxis, ...]
    for t in range(0, len(positions)):
        x, y = positions[t, :-4, 0], positions[t, :-4, 1]
        x = np.array(x / range_pos * n_cells, dtype=np.int32)
        y = np.array(y / range_pos * n_cells, dtype=np.int32)
        idx = np.where((x >= 0) & (x < n_cells) & (y >= 0) & (y < n_cells))
        occupancy_map[x[idx], y[idx]] += 1
        if t % sampling_frequency == 0:
            input_maps = np.concatenate(
                (input_maps, occupancy_map[np.newaxis, ...]), axis=0
            )

    # input_maps = input_maps[1:]  # Remove the initial zero map
    return input_maps


n_cells = 64
range_pos = 253

output_grid = np.load("circles_arena.npy")
output_grid = (output_grid - np.min(output_grid)) / (
    np.max(output_grid) - np.min(output_grid)
)

output_grid_2 = np.load("race_track.npy")
output_grid_2 = (output_grid_2 - np.min(output_grid_2)) / (
    np.max(output_grid_2) - np.min(output_grid_2)
)

output_grid_3 = np.load("slit_arena.npy")
output_grid_3 = (output_grid_3 - np.min(output_grid_3)) / (
    np.max(output_grid_3) - np.min(output_grid_3)
)

output_grid_4 = np.load("final_arena.npy").astype(np.float32)
output_grid_4 = (output_grid_4 - np.min(output_grid_4)) / (
    np.max(output_grid_4) - np.min(output_grid_4)
)
output_grid_4 = cv2.resize(output_grid_4, (64, 64))
output_grid_4 = output_grid_4.astype(np.float32)

output_grid = np.array(output_grid)
output_grid_2 = np.array(output_grid_2)
output_grid_3 = np.array(output_grid_3)
output_grid_4 = np.array(output_grid_4)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(output_grid, cmap="binary")
ax[0].set_title("Circles Arena")
ax[1].imshow(output_grid_2, cmap="binary")
ax[1].set_title("Race Track")
ax[2].imshow(output_grid_3, cmap="binary")
ax[2].set_title("Slit Arena")
plt.savefig("arena_maps.png")
input_map = np.zeros((1, n_cells, n_cells), dtype=np.float32)
input_map_2 = np.zeros((1, n_cells, n_cells), dtype=np.float32)
input_map_3 = np.zeros((1, n_cells, n_cells), dtype=np.float32)
input_map_4 = np.zeros((1, n_cells, n_cells), dtype=np.float32)

for i in range(1):
    positions = np.load("trajectory_files/traj_circles.pkl", allow_pickle=True)[
        "features"
    ].squeeze()

    print(f"positions has shape{np.shape(positions)}")
    input = np.array(
        create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=100)
    )
    input_map = np.concatenate((input_map, input), axis=0)

    positions = np.load("trajectory_files/traj_0.pkl", allow_pickle=True)[
        "features"
    ].squeeze()
    positions = positions[:6000]

    print(f"positions has shape{np.shape(positions)}")
    input = np.array(
        create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=100)
    )

    input_map_2 = np.concatenate((input_map_2, input), axis=0)

    positions = np.load("trajectory_files/traj_slit.pkl", allow_pickle=True)[
        "features"
    ].squeeze()
    print(f"positions has shape{np.shape(positions)}")
    input = np.array(
        create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=100)
    )
    input_map_3 = np.concatenate((input_map_3, input), axis=0)
    
    positions = np.load("trajectory_files/traj_final.pkl", allow_pickle=True)[
        "features"
    ].squeeze()
    print(f"positions has shape{np.shape(positions)}")
    input = np.array(
        create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=10)
    )
    input_map_4 = np.concatenate((input_map_4, input), axis=0)
    
    positions = np.load("trajectory_files/traj_final_2.pkl", allow_pickle=True)[
        "features"
    ].squeeze()
    print(f"positions has shape{np.shape(positions)}")
    input = np.array(
        create_occupancy_map(positions, n_cells, range_pos, sampling_frequency=10)
    )
    input_map_4 = np.concatenate((input_map_4, input), axis=0)

print(
    f"circles input map shape: {input_map.shape}, \n race input map shape: {input_map_2.shape}, \n slit input map shape: {input_map_3.shape}"
)

input_data = input_map[..., None]  # Add batch and channel dimensions
target_data = output_grid[None, ..., None]  # Add batch and channel dimensions
input_data_2 = input_map_2[..., None]  # Add batch and channel dimensions
target_data_2 = output_grid_2[None, ..., None]  # Add batch and channel dimensions
input_data_3 = input_map_3[..., None]  # Add batch and channel dimensions
target_data_3 = output_grid_3[None, ..., None]  # Add batch and channel dimensions
input_data_4 = input_map_4[..., None]  # Add batch and channel dimensions
target_data_4 = output_grid_4[None, ..., None]  # Add batch and channel dimensions

print(input_data.shape)
print(target_data.shape)
input_list = [input_data, input_data_2, input_data_3, input_data_4]
target_list = [target_data, target_data_2, target_data_3, target_data_4]
batch_size =56
key = jax.random.PRNGKey(int(time.time()))

print(np.array(target_list).shape)

def batch_generator(input_list, target_list, batch_size, key):
    n_sets = len(input_list)
    assert (
        batch_size % n_sets == 0
    ), "batch_size muss durch die Anzahl Arenen teilbar sein"
    samples_per_set = batch_size // n_sets
    samples_per_set =[8,8,8,32]

    while True:
        inputs = []
        targets = []

        for i in range(n_sets):
            data_len = input_list[i].shape[0]
            key, subkey = jax.random.split(key)
            sample_idxs = jax.random.choice(
                subkey, data_len, shape=(samples_per_set[i],), replace=False
            )

            for j in range(samples_per_set[i]):
                key, rot_key = jax.random.split(key)
                sample = input_list[i][sample_idxs[j]]
                target = target_list[i][0]

                if jax.random.uniform(rot_key, ()) < 0.5:
                    inputs.append(sample)
                    targets.append(target)
                else:
                    inputs.append(sample.swapaxes(0, 1))
                    targets.append(target.swapaxes(0, 1))
        inputs.append(np.zeros_like(inputs[0]))
        targets.append(np.ones_like(targets[0])*0.5)

        yield jnp.stack(inputs), jnp.stack(targets)


# Erstelle Generator
gen = batch_generator(input_list, target_list, batch_size, key)


# class DWConv(nn.Module):
#     """Depthwise-Separable-Convolution (DWSC)"""

#     out_chan: int
#     stride: int = 1

#     @nn.compact
#     def __call__(self, x):
#         in_chan = x.shape[-1]
#         # 1) Depthwise
#         x = nn.Conv(
#             features=in_chan,
#             kernel_size=(3, 3),
#             strides=(self.stride, self.stride),
#             padding="SAME",
#             feature_group_count=in_chan,  # depthwise
#             use_bias=True,
#         )(x)
#         x = nn.silu(x)
#         # 2) Pointwise
#         x = nn.Conv(
#             features=self.out_chan,
#             kernel_size=(1, 1),
#             strides=(1, 1),
#             padding="SAME",
#             use_bias=True,
#         )(x)
#         x = nn.silu(x)
#         return x


# class SmallUnetDWSC(nn.Module):
#     """Kompaktes U-Net mit Depthwise-Separable-Convs, ~85k Parameter"""

#     @nn.compact
#     def __call__(self, x):
#         # ----------------
#         # Encoder
#         # ----------------
#         # Block 1: 1 -> 32, einfacher DWSC, Stride=2
#         x1 = DWConv(out_chan=32, stride=2)(x)  # 64x64x1 --> 32x32x32

#         # Block 2: 32 -> 64, 2×DWSC, erster mit Stride=2
#         y = DWConv(out_chan=64, stride=2)(x1)  # 32x32x32 --> 16x16x64
#         x2 = DWConv(out_chan=64, stride=1)(y)  # 16x16x64 --> 16x16x64

#         # Block 3: 64 -> 96, 2×DWSC, erster mit Stride=2
#         z = DWConv(out_chan=96, stride=2)(x2)  # 16x16x64 -->  8x 8x96
#         x3 = DWConv(out_chan=96, stride=1)(z)  #  8x 8x96 -->  8x 8x96

#         # ----------------
#         # Bottleneck: 96 -> 96, 2×DWSC, Stride=1
#         b = DWConv(out_chan=96, stride=1)(x3)  #  8x 8x96 -->  8x 8x96
#         b = DWConv(out_chan=96, stride=1)(b)  #  8x 8x96 -->  8x 8x96

#         # ----------------
#         # Decoder
#         # ----------------
#         # Up-Block 1:  8x8x96 --> Upsample -> 16x16x96 -> 2×DWSC -> 16x16x64
#         u1 = jax.image.resize(b, (b.shape[0], 16, 16, 96), method="nearest")
#         u1 = DWConv(out_chan=64, stride=1)(u1)
#         u1 = DWConv(out_chan=64, stride=1)(u1)
#         # Skip-Connection: x2 (16x16x64)
#         u1 = jnp.concatenate([u1, x2], axis=-1)  # 16x16x128
#         u1 = nn.Conv(features=64, kernel_size=(1, 1), padding="SAME", use_bias=True)(u1)
#         u1 = nn.silu(u1)

#         # Up-Block 2: 16x16x64 --> Upsample -> 32x32x64 -> 2×DWSC -> 32x32x32
#         u2 = jax.image.resize(u1, (u1.shape[0], 32, 32, 64), method="nearest")
#         u2 = DWConv(out_chan=32, stride=1)(u2)
#         u2 = DWConv(out_chan=32, stride=1)(u2)
#         # Skip-Connection: x1 (32x32x32)
#         u2 = jnp.concatenate([u2, x1], axis=-1)  # 32x32x64
#         u2 = nn.Conv(features=32, kernel_size=(1, 1), padding="SAME", use_bias=True)(u2)
#         u2 = nn.silu(u2)

#         # Up-Block 3: 32x32x32 --> Upsample -> 64x64x32 -> 1×DWSC -> 64x64x16
#         u3 = jax.image.resize(u2, (u2.shape[0], 64, 64, 32), method="nearest")
#         u3 = DWConv(out_chan=16, stride=1)(u3)
#         # Skip vom Input (64x64x1):
#         u3 = jnp.concatenate([u3, x], axis=-1)  # 64x64x17
#         u3 = nn.Conv(features=16, kernel_size=(1, 1), padding="SAME", use_bias=False)(
#             u3
#         )
#         u3 = nn.silu(u3)

#         # ----------------
#         # Final Output-Layer: 16 -> 1
#         out = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME", use_bias=False)(
#             u3
#         )
#         out = nn.sigmoid(out)
#         return out


        


class SmallAttentionUNet(nn.Module):
    """Kompakter U-Net mit Self-Attention im Bottleneck."""

    @nn.compact
    def __call__(self, x):
        # --- Encoder ---
        x = jnp.clip(x, 0, 1000.0)/1000.0  # Sicherstellen, dass Input in [0,1] ist
        # Block 1: 1→16, Downsample auf (32×32)
        e1 = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_0")(
            x
        )  # (32,32,16)
        e1 = nn.silu(e1)

        # Block 2: 16→32, Downsample auf (16×16)
        e2 = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_1")(
            e1
        )  # (16,16,32)
        e2 = nn.silu(e2)

        # Block 3: 32→48, Downsample auf (8×8)
        e3 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="enc_conv_2")(
            e2
        )  # (8, 8,48)
        e3 = nn.silu(e3)

        # --- Bottleneck mit Self-Attention ---
        b = self.attention_helper(e3)  # (8,8,48)

        # --- Decoder ---
        # Up 1: (8×8)→(16×16), 48→32
        d1 = jax.image.resize(
            b, (b.shape[0], 16, 16, 48), method="nearest"
        )  # (16,16,48)
        d1 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(d1)  # (16,16,32)
        d1 = nn.silu(d1)
        d1 = jnp.concatenate([d1, e2], axis=-1)  # Skip aus e2 → (16,16,64)
        d1 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(d1)  # (16,16,32)
        d1 = nn.silu(d1)

        # Up 2: (16×16)→(32×32), 32→16
        d2 = jax.image.resize(
            d1, (d1.shape[0], 32, 32, 16), method="nearest"
        )  # (32,32,32)
        d2 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(d2)  # (32,32,16)
        d2 = nn.silu(d2)
        d2 = jnp.concatenate([d2, e1], axis=-1)  # Skip aus e1 → (32,32,32)
        d2 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(d2)  # (32,32,16)
        d2 = nn.silu(d2)

        # Up 3: (32×32)→(64×64), 16→8
        d3 = jax.image.resize(
            d2, (d2.shape[0], 64, 64, 16), method="nearest"
        )  # (64,64,16)
        d3 = nn.Conv(features=8, kernel_size=(3, 3), padding="SAME")(d3)  # (64,64,8)
        d3 = nn.silu(d3)
        # Kein Skip von ursprünglichem x, weil x hat 1 Kanal; wir verzichten hier bewusst
        # auf einen zusätzlichen Sicherheits-Skip (könnte aber optional sein)

        # --- Final Output ---
        out = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d3)  # (64,64,1)
        return nn.sigmoid(out)
    def attention_helper(self, x):
        B, H, W, C = x.shape
        # 1) Query/Key: reduzierte Dimension C//8, Value: C
        query = nn.Conv(features=C // 8, kernel_size=(1, 1), padding="SAME", name="enc_attention_q")(x)
        key = nn.Conv(features=C // 8, kernel_size=(1, 1), padding="SAME", name="enc_attention_k")(x)
        value = nn.Conv(features=C, kernel_size=(1, 1), padding="SAME", name="enc_attention_v")(x)

        # Flatten räumlich in (B, N, D)
        N = H * W
        q_flat = query.reshape((B, N, C // 8))  # (B, 64, C//8)
        k_flat = key.reshape((B, N, C // 8))  # (B, 64, C//8)
        v_flat = value.reshape((B, N, C))  # (B, 64, C)

        # 2) Attention‐Score: (B, N, N)
        attn_scores = jnp.einsum("bqi,bki->bqk", q_flat, k_flat)  # (B,64,64)
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (B,64,64)

        # 3) Weighted Sum für Output
        attn_out = jnp.einsum("bqk,bkv->bqv", attn_weights, v_flat)  # (B,64,C)
        attn_out = attn_out.reshape((B, H, W, C))  # (B,8,8,C)

        # 4) Learnable Skalierungs-Parameter gamma
        gamma = self.param("gamma", nn.initializers.zeros, (1,))
        return x + gamma * attn_out


def edge_pad(x, kernel_size):
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    return jnp.pad(
        x,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='edge'
    )

class OccupancyMapper(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.clip(x, 0, 1000.0)/1000.0  # Sicherstellen, dass Input in [0,1] ist
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

def dice_loss(pred: jnp.ndarray, true: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    pred: (B, H, W, 1) mit Werten in [0,1]
    true: (B, H, W, 1) mit Werten in {0,1}
    """
    pred_flat = pred.reshape((pred.shape[0], -1))
    true_flat = true.reshape((true.shape[0], -1))
    intersection = jnp.sum(pred_flat * true_flat, axis=-1)
    summ = jnp.sum(pred_flat, axis=-1) + jnp.sum(true_flat, axis=-1)
    dice = (2.0 * intersection + eps) / (summ + eps)
    return 1.0 - dice.mean()


# Define a loss function
def binary_cross_entropy_loss(logits, labels):
    return -jnp.mean(
        labels * jnp.log(logits + 1e-7)
        + 1.0 * (1 - labels) * jnp.log(1 - logits + 1e-7)
    )


def combined_loss(logits, labels):
    bce_loss = binary_cross_entropy_loss(logits, labels)
    dice = dice_loss(logits, labels)
    return bce_loss + dice  # You can adjust the weighting of the losses if needed


def train_step(state, input_data, target_data):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, input_data)
        loss = combined_loss(logits, target_data)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def save_train_state(state, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "train_state.pkl")
    with open(file_path, "wb") as f:
        model_params = state.params
        opt_state = state.opt_state
        pickle.dump((model_params, opt_state), f)
        
def save_encoder_state(state, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "encoder_state.pkl")

    # Nur Encoder-Parameter extrahieren
    encoder_keys = ["enc_conv_0", "enc_conv_1", "enc_conv_2", "enc_attention_q", "enc_attention_k", "enc_attention_v", "gamma"]
    encoder_params = {
        k: v for k, v in state.params.items() if k in encoder_keys
    }

    with open(file_path, "wb") as f:
        pickle.dump(encoder_params, f)


def load_train_state(path):
    file_path = os.path.join(path, "train_state.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No train state file found at {file_path}")
    with open(file_path, "rb") as f:
        params, opt_state = pickle.load(f)
    return params, opt_state

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

losses = np.array([])
losses_circles = np.array([])
losses_slit = np.array([])
losses_race = np.array([])
losses_final = np.array([])
print("Starting training...", flush=True)
for epoch in range(30001):  # Number of epochs
    input_batch, target_batch = next(gen)
    state, loss = train_step(state, input_batch, target_batch)
    # print(f"Epoch {epoch + 1}, Loss: {loss}")
    losses = np.append(losses, loss)
    plt.figure(figsize=(10, 6))
    epochs = np.arange(0, losses.shape[0] - 1)
    plt.loglog(losses, label="Loss", linewidth=0.5)
    if len(losses_circles) > 0:
        eval_points = np.arange(1000, losses_circles.shape[0] * 1000 + 1000, 1000)
        plt.scatter(
            eval_points, losses_circles, label="Loss Circles", marker="x", s=20, c="red"
        )
        plt.scatter(
            eval_points, losses_race, label="Loss Race", s=20, c="green", marker="x"
        )
        plt.scatter(
            eval_points, losses_slit, label="Loss Slit", s=20, c="black", marker="x"
        )
        plt.scatter(
            eval_points, losses_final, label="Loss Final", s=20, c="blue", marker="x"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
    if epoch % 100 == 0 and epoch > 0:
        _, loss_circles = train_step(state, input_data, target_data)
        _, loss_race = train_step(state, input_data_2, target_data_2)
        _, loss_slit = train_step(state, input_data_3, target_data_3)
        _, loss_final = train_step(state, input_data_4, target_data_4)
        losses_circles = np.append(losses_circles, loss_circles)
        losses_race = np.append(losses_race, loss_race)
        losses_slit = np.append(losses_slit, loss_slit)
        losses_final = np.append(losses_final, loss_final)
        logger.info(f"Step {epoch}, loss: {loss:.4f}")
        occ = np.ones((1, n_cells, n_cells, 1), dtype=np.float32)
        occ[:,32:,:,0]=0
        occ[:, :32, 32:, 0] = 1000
        prediction = state.apply_fn({"params": state.params}, occ)[0, ..., 0]
        plt.figure(figsize=(6, 6))
        plt.imshow(prediction, cmap="binary")
        plt.title(f"Prediction at epoch {epoch}")
        plt.savefig(f"prediction_{epoch}.png")
        plt.close()
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(input_data[3, ..., 0], cmap="binary")
        ax[0, 0].set_title("Input Map")
        prediction = state.apply_fn({"params": state.params}, input_data[3:4])[
            0, ..., 0
        ]
        ax[0, 1].imshow(prediction, cmap="binary")
        ax[0, 1].set_title("prediction")
        ax[0, 2].imshow((prediction > 0.8) != (target_data[0, ..., 0]), cmap="binary")
        ax[0, 2].set_title(
            f"Difference {np.sum(np.array(target_data[0, ..., 0]!= (prediction > 0.8)))}"
        )

        ax[1, 0].imshow(input_data_4[-1, ..., 0], cmap="binary")
        prediction = state.apply_fn({"params": state.params}, input_data_4[-2:-1])[
            0, ..., 0
        ]
        ax[1, 1].imshow(prediction, cmap="binary")
        ax[1, 2].imshow((prediction > 0.8) != (target_data_4[0, ..., 0]), cmap="binary")
        ax[1, 2].set_title(
            f"Difference {np.sum(np.array(target_data_4[0, ..., 0]!= (prediction > 0.8)))}"
        )

        plt.savefig(f"map_{epoch}.png")
        plt.close(fig)
        save_train_state(state, "occupancy_model_checkpoint")
        # save_encoder_state(state, "occupancy_model_checkpoint")
