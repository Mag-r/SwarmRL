# %%
import numpy as np 
import cv2 as cv
import time
# import read_position_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flax import linen as nn
import skimage
import jax
import pickle
from flax.training.train_state import TrainState
import optax
from scipy.ndimage import label, find_objects
from skimage.feature import peak_local_max


# %%
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
model = Autoencoder()
size = 253
dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, size, size, 1))
params = model.init(jax.random.PRNGKey(0), dummy_input)

model_state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(0.001)
    )
# Load pretrained model
with open("autoencoder_model/autoencoder_hex.pkl", "rb") as f:
    model_params= pickle.load(f)

model_state = model_state.replace(
    params=model_params
)

# %%

input_data = cv.imread("../images/camera_image_0000.png")
input_data = cv.resize(input_data, (size, size))
input_data = cv.cvtColor(input_data, cv.COLOR_RGB2BGR)

print(input_data.shape)
# input_data = cv2.resize(input_data, (size,size))

fig, ax = plt.subplots(1,3)

ax[0].imshow(input_data)

start = time.time()

cleaned_image = model.apply(model_state.params, input_data.reshape(1, size, size, 3))

ax[1].imshow(cleaned_image[0,...], cmap='gray')

cleaned_image = cleaned_image[:,:]
image = np.array(cleaned_image, dtype=np.uint8) 

positions = peak_local_max(
            np.array(cleaned_image.squeeze()), min_distance=3, threshold_abs=0.6, num_peaks=7
        )
contour_image = np.array(input_data, dtype=np.uint8)
for position in positions:
    center = tuple([position[1], position[0]])
    cv.circle(contour_image, center, 2, (255, 0, 0), -1)

print(len(positions))

ax[2].imshow(contour_image)
print(time.time()-start)

plt.savefig("blob_detection.png")
plt.show()

