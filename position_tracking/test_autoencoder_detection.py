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
with open("../Models/model_trained.pkl", "rb") as f:
    model_params= pickle.load(f)

model_state = model_state.replace(
    params=model_params
)

# %%

input_data = plt.imread("unlabeled_images_for_train/images/camera_image_0000.png")
input_data = input_data[::2,::2,0]
print(input_data.shape)
# input_data = cv2.resize(input_data, (size,size))

mean = np.mean(input_data)
std = np.std(input_data)
input_data = (input_data - mean) / std
input_data = np.expand_dims(input_data, axis=-1)
fig, ax = plt.subplots(1,3)
ax[0].imshow(input_data, cmap='gray')



start = time.time()
# thresholded_image = (input_data > 1.9).astype(np.float32)
# ax[1].imshow(thresholded_image.reshape(size,size, 1), cmap='gray')
cleaned_image = model.apply(model_state.params, input_data.reshape(1, size, size, 1))
cleaned_image = cleaned_image > 0.9
cleaned_image = np.squeeze(cleaned_image)
ax[1].imshow(cleaned_image[:,:], cmap='gray')
cleaned_image = cleaned_image[:,:]
image = np.array(cleaned_image*255, dtype=np.uint8) 
# params = cv2.SimpleBlobDetector_Params()
# params.filterByColor = True
# params.blobColor = 255

# params.filterByArea = False
# params.maxArea = 400
# params.minArea = 1

# params.filterByConvexity = False  # Allow irregular shapes
# params.filterByInertia = False  # Allow elongated shapes
# params.filterByCircularity = False
# params.minCircularity = 0.4
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(image.astype(np.uint8))
# Find contours
# Label connected components
labeled_image, num_features = label(image)

# Find bounding boxes of labeled regions
objects = find_objects(labeled_image)

# Calculate and print the positions of the objects
positions = []
for obj in objects:
    y, x = obj
    x_start, x_end = x.start, x.stop
    y_start, y_end = y.start, y.stop
    width = x_end - x_start
    height = y_end - y_start
    positions.append((x_start, y_start, width, height))
    print(f"Position: x={x_start}, y={y_start}, width={width}, height={height}")
    # Draw rectangles around detected objects
    for pos in positions:
        x, y, w, h = pos
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Calculate and print the positions of the contours
positions = [cv2.boundingRect(contour) for contour in contours]
for pos in positions:
    x, y, w, h = pos
    print(f"Position: x={x}, y={y}, width={w}, height={h}")

# Display the image with contours
ax[2].imshow(contour_image)
# print(len(keypoints))
print(time.time()-start)
# image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.figure()
# ax[2].imshow(image_with_keypoints, cmap="gray")
plt.savefig("blob_detection.png")
# plt.show()




