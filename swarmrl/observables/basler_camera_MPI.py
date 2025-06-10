"""
Position observable computer.
"""

from abc import ABC
from typing import List
import os
import queue
import threading
import shutil
import jax.numpy as np
import numpy as onp
import jax
from flax.training.train_state import TrainState
import pickle
import optax
import cv2
from pypylon import pylon
import logging
from matplotlib import pyplot as plt
from flax import linen as nn
import time
from matplotlib import pyplot as plt
from swarmrl.observables.observable import Observable
from skimage.feature import peak_local_max

logger = logging.getLogger(__name__)


class BaslerCameraObservable(Observable, ABC):
    """
    Class to receive images from Gauravs Camera.
    Detects the colloids in the image and returns their positions.
    The camera is a Basler acA40011027-USB3.
    """

    camParam = {
        "camName": "40011027",
        "topicName": "/image",
        "fps": 10,
        "exposureTime": 5000,
        "width": 2048,
        "height": 2048,
        "xOffset": 0,
        "yOffset": 0,
        "xReverse": False,
        "yReverse": False,
        "scale": 0.5,
        "lightSource": "Daylight5000K",
        "balanceRatioSelector_r": "Red",
        "balanceRatio_r": 1.3,
        "balanceRatioSelector_g": "Green",
        "balanceRatio_g": 1.0,
        "balanceRatioSelector_b": "Blue",
        "balanceRatio_b": 1.6,
        "colored": True,
    }

    def __init__(
        self,
        resolution: List[int],
        autoencoder: nn.Module,
        model_path: str = None,
        number_particles: int = 7,
    ):
        """
        Constructor for the observable.
        """
        super().__init__(particle_type=0)  # better way to do this?
        self.number_particles = number_particles
        self.resolution = resolution
        tlf = pylon.TlFactory.GetInstance()
        detected_cameras = tlf.EnumerateDevices()
        self.camera = None
        self.image_count = 470
        for cam in detected_cameras:
            serial_number = cam.GetFriendlyName().split()[-1][1:-1]
            if serial_number == self.camParam["camName"]:
                self.camera = pylon.InstantCamera(tlf.CreateDevice(cam))
                self.init_camera()
                logger.info(f"Camera {self.camParam['camName']} found and init.")
                break
        if self.camera is None or not self.camera.IsPylonDeviceAttached():
            logger.error(f"Camera {self.camParam['camName']} not found.")
            raise Exception(f"Camera {self.camParam['camName']} not found.")
        # if os.path.exists("images"):
        #     shutil.rmtree("images")
        # os.makedirs("images")
        self.image_queue = queue.Queue()
        self.image_saving_thread = threading.Thread(
            target=self.save_images_async, daemon=True
        )
        self.image_count = 0
        self.image_saving_thread.start()
        self.autoencoder = autoencoder
        self.init_autoencoder(model_path)
        self.threshold = 0.4


    def init_autoencoder(self, model_path: str = None):
        """Initialize the autoencoder.
        This method initializes the autoencoder model and loads the parameters from a file if provided.

        Args:
            model_path (str, optional): _description_. Defaults to None.
        """
        dummy_input = jax.random.normal(
            jax.random.PRNGKey(0), (1, self.resolution[0], self.resolution[1], 1)
        )
        params = self.autoencoder.init(jax.random.PRNGKey(0), dummy_input)
        self.model_state = TrainState.create(
            apply_fn=self.autoencoder.apply, params=params, tx=optax.adam(0.001)
        )
        if model_path:
            with open(model_path, "rb") as f:
                model_params = pickle.load(f)
            self.model_state = self.model_state.replace(params=model_params)

    def init_camera(self):
        """Intialize the camera with the parameters defined in camParam.
        """
        self.camera.Open()
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.AcquisitionFrameRate.SetValue(self.camParam["fps"])
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.ExposureAuto.SetValue("Off")
        self.camera.ExposureTime.SetValue(self.camParam["exposureTime"])
        self.camera.DeviceLinkThroughputLimitMode.SetValue("Off")
        self.camera.ReverseX.SetValue(self.camParam["xReverse"])
        self.camera.ReverseY.SetValue(self.camParam["yReverse"])
        self.camera.Width.SetValue(self.camParam["width"])
        self.camera.Height.SetValue(self.camParam["height"])
        self.camera.OffsetX.SetValue(self.camParam["xOffset"])
        self.camera.OffsetY.SetValue(self.camParam["yOffset"])
        self.camera.PixelFormat.SetValue("RGB8")
        if not self.camera.PixelFormat().count("Mono"):
            self.camera.LightSourcePreset.SetValue(self.camParam["lightSource"])
            self.camera.BalanceWhiteAuto.SetValue("Off")
            self.camera.BalanceRatioSelector.SetValue(
                self.camParam["balanceRatioSelector_r"]
            )
            self.camera.BalanceRatio.SetValue(self.camParam["balanceRatio_r"])
            self.camera.BalanceRatioSelector.SetValue(
                self.camParam["balanceRatioSelector_g"]
            )
            self.camera.BalanceRatio.SetValue(self.camParam["balanceRatio_g"])
            self.camera.BalanceRatioSelector.SetValue(
                self.camParam["balanceRatioSelector_b"]
            )
            self.camera.BalanceRatio.SetValue(self.camParam["balanceRatio_b"])
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def compute_observable(self, colloids: np.ndarray) -> np.ndarray:
        """
        Abuse of the compute_observable method to return the image directly obtained from the experiment.
        Parameters
        ----------
        colloids : should be None
        """
        if colloids is not None:
            raise ValueError("Colloids should be None for this observable.")
        if self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                image = grabResult.GetArray()
                if not self.camParam["colored"]:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                grabResult.Release()
            else:
                logger.error("Grab failed.")
                image = np.zeros(self.resolution)
        if self.image_queue.qsize() > 3:
            logger.warning(
                f"Image queue is starting to fill. Current size {self.image_queue.qsize()}"
            )
        image = cv2.resize(image, (506,506))
        self.image_queue.put(image.copy())
        image = cv2.resize(image, (self.resolution[0], self.resolution[1]))
        # image[105:150, 70:180,:] = 0

        positions = self.extract_positions(image)
        while positions.shape[1] < self.number_particles:
            padding = self.number_particles - positions.shape[1]
            padding = positions[:, :padding, :].copy()
            positions = np.concatenate((positions, padding), axis=1)
        return positions


    def extract_positions(self, original_image: np.ndarray) -> np.ndarray:
        """
        Extracts the positions of the colloids from the image.
        """
        image = original_image.copy()
        # mask = (image[:, :, 2] > image[:, :, 1]) & (image[:, :, 2] > image[:, :, 0])
        # image[mask] = 0

        image = cv2.resize(image, (self.resolution[0], self.resolution[1]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.array(image, dtype=np.float32)
        image = np.reshape(image, (1, self.resolution[0], self.resolution[1], 3))
        cleaned_image = self.model_state.apply_fn(self.model_state.params, image)
        positions = self.peak_detection(cleaned_image)
        if len(positions) != self.number_particles:
            logger.warning(
                f"Number of particles detected {len(positions)} is not equal to the expected number of particles {self.number_particles}."
            )
            logger.info(f"index of image: {self.image_count}")
            if len(positions) ==0:
                plt.imsave(
                    f"images/cleaned_image_.png",
                    np.array(cleaned_image.squeeze(), dtype=np.uint8),
                )
                raise ValueError(
                    "No particles detected. Is the light on?."
                )
            
        contour_image = onp.array(original_image, dtype=np.uint8)
        for position in positions:
            center = tuple([position[1], position[0]])
            cv2.circle(contour_image, center, 2, (255, 0, 0), -1)
        self.image_queue.put(contour_image)

        return positions.reshape(1, -1, 2)

    def peak_detection(self, cleaned_image):
        """
        Detects peaks in the image using the skimage feature module.
        """
        coordinates = peak_local_max(
            onp.array(cleaned_image.squeeze()), min_distance=4, threshold_abs=self.threshold, num_peaks=self.number_particles
        )
        return coordinates

    def save_images_async(self):
        """
        Saves the images from the camera to a file. 253x253 images are saved as latest_camera_image.png
        506x506 images are saved as camera_image_XXXX.png.
        """
        while True:
            if not self.image_queue.empty():
                image = self.image_queue.get()
                if image.shape[0] == 253:
                    image = cv2.resize(image, (506, 506))
                    plt.imsave(f"images/latest_camera_image.png", image)
                elif image.shape[0] == 506:
                    plt.imsave(
                        f"images/camera_image_{self.image_count:04d}.png",
                        image,
                    ) if image.shape[0] == 506 else None
                    self.image_count = self.image_count + 1
            else:
                threading.Event().wait(0.1)
