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

from swarmrl.observables.observable import Observable

logger = logging.getLogger(__name__)


class BaslerCameraObservable(Observable, ABC):
    """
    Class to receive images from Gauravs Camera.
    some abuse of the observable class to return the image directly.
    """

    camParam = {
        "camName": "40011027",
        "topicName": "/image",
        "fps": 10,
        "exposureTime": 5000,
        "width": 2048,
        "height": 2048,
        "xOffset": 128,
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
        self.image_count = 0
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
        self.threshold = 0.8

    def init_autoencoder(self, model_path: str = None):
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
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

                image = cv2.resize(image, (self.resolution[0], self.resolution[1]))
                grabResult.Release()
            else:
                logger.error("Grab failed.")
                image = np.zeros(self.resolution)
        if self.image_queue.qsize() > 3:
            logger.warning(
                f"Image queue is starting to fill. Current size {self.image_queue.qsize()}"
            )
        # self.image_queue.put(image)
        positions = self.extract_positions(image)
        if positions.shape[1] < self.number_particles:
            padding = self.number_particles - positions.shape[1]
            padding = np.zeros((1, padding, 2))
            positions = np.concatenate((positions, padding), axis=1)
        elif positions.shape[1] > self.number_particles:
            positions = positions[:, : self.number_particles, :]
        return positions

    def extract_positions(self, original_image: np.ndarray) -> np.ndarray:
        """
        Extracts the positions of the colloids from the image.
        """
        image = cv2.resize(original_image, (self.resolution[0], self.resolution[1]))
        image = np.array(image, dtype=np.float32)
        image = (image - np.mean(image)) / np.std(image)
        image = np.reshape(image, (1, self.resolution[0], self.resolution[1], 1))
        cleaned_image = self.model_state.apply_fn(self.model_state.params, image)
        processed_image, contours = self.threshold_and_extract_contours(cleaned_image)

        original_image = onp.array(original_image, dtype=np.uint8)
        contour_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(original_image, contours, -1, (255, 0, 0), 2)
        self.image_queue.put(original_image)
        # contour_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(contour_image, contours, -1, 255, -1)
        # self.image_queue.put(contour_image)
        min_length = 0.01
        contours = [
            contour for contour in contours if cv2.arcLength(contour, True) > min_length
        ]
        attempts = 0
        while len(contours) != self.number_particles and attempts < 5:
            self.threshold = (
                self.threshold + (len(contours) - self.number_particles) * 0.01
            )
            self.threshold = np.clip(self.threshold, 0.5, 0.98)
            logger.warning(
                f"Detected {len(contours)} of {self.number_particles}. Threshold changed to {self.threshold}, in attempt {attempts}."
            )
            processed_image, contours = self.threshold_and_extract_contours(
                cleaned_image
            )
            attempts = attempts + 1
        # image = np.squeeze(image)
        # image = np.array(image, dtype=np.uint8)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        # self.image_queue.put(image)
        positions = np.array([cv2.boundingRect(contour) for contour in contours])
        positions = positions[:, :2] + positions[:, 2:] / 2

        positions = (positions - positions.mean()) / positions.std()

        if len(positions) != self.number_particles:
            logger.warning(
                f"Number of particles detected {len(positions)} is not equal to the expected number of particles {self.number_particles}."
            )

        return positions.reshape(1, -1, 2)

    def threshold_and_extract_contours(self, cleaned_image):
        processed_image = cleaned_image > self.threshold
        processed_image = np.squeeze(processed_image)
        processed_image = onp.array(processed_image * 255, dtype=onp.uint8)

        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return processed_image, contours

    def save_images_async(self):
        while True:
            if not self.image_queue.empty():
                image = self.image_queue.get()
                plt.imsave(
                    f"images/camera_image_{self.image_count:04d}.png",
                    image,
                )
                plt.imsave(f"images/latest_camera_image.png", image)
                self.image_count = self.image_count + 1
            else:
                threading.Event().wait(0.1)
