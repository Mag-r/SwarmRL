# %%
import numpy as np 
import cv2
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
with open("../Models/autoencoder_4_8.pkl", "rb") as f:
    model_params= pickle.load(f)

model_state = model_state.replace(
    params=model_params
)

# %%

input_data = plt.imread("../images/camera_image_0000.png")
input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
print(input_data.shape)
# input_data = cv2.resize(input_data, (size,size))

mean = np.mean(input_data)
std = np.std(input_data)
input_data = (input_data - mean) / std
input_data = np.expand_dims(input_data, axis=-1)
fig, ax = plt.subplots(1,3)

ax[0].imshow(input_data)



start = time.time()

cleaned_image = model.apply(model_state.params, input_data.reshape(1, size, size, 1))
cleaned_image = cleaned_image > 0.7
cleaned_image = np.squeeze(cleaned_image)

ax[1].imshow(cleaned_image[:,:], cmap='gray')

cleaned_image = cleaned_image[:,:]
image = np.array(cleaned_image*255, dtype=np.uint8) 

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 0
print([cv2.arcLength(contour, False) for contour in contours])
contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_area]
contour_image = cv2.cvtColor(input_data.squeeze(), cv2.COLOR_GRAY2RGB)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Calculate and print the positions of the contours
positions = [cv2.boundingRect(contour) for contour in contours]

for i, pos in enumerate(positions):
    x, y, w, h = pos
    print(f"Position: x={x}, y={y}, width={w}, height={h}")

print(len(positions))

ax[2].imshow(contour_image)
print(time.time()-start)

plt.savefig("blob_detection.png")
plt.show()





# %%
