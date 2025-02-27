import jax.numpy as jnp
import numpy as np
from jax import random
import logging
import cv2
from flax import linen as nn
from swarmrl.tasks.task import Task
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class ExperimentTask(Task):

    def __init__(self, number_particles: int):
        super().__init__(particle_type=0)
        self.first_sweep_detector_params = cv2.SimpleBlobDetector_Params()
        self.first_sweep_detector_params.filterByArea = True
        self.first_sweep_detector_params.maxArea = 230
        self.first_sweep_detector_params.minArea = 10
        self.first_sweep_detector_params.filterByColor = True
        self.first_sweep_detector_params.blobColor = 255
        self.first_sweep_detector_params.filterByConvexity = False
        self.first_sweep_detector_params.filterByInertia = False
        self.first_detector = cv2.SimpleBlobDetector_create(
            self.first_sweep_detector_params
        )
        
        self.number_particles = number_particles
        self.old_residual = None
        self.old_angle_error = None
        self.angle_noramlization = 1.6
        self.regression_normalization = 2000

    def detect_blobs(self, image: np.ndarray) -> np.ndarray:
        image = image[0,0,:,:,0]
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std
        
        image = image > 3.5
        image = np.array(image * 255, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        image = cv2.erode(image, kernel, iterations=2)
        image = image.astype(np.uint8)
        keypoints = self.first_detector.detect(image)
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        logger.info(f"Detected {len(keypoints)} particles.")

        # plt.imshow(image_with_keypoints, cmap="gray")
        # plt.savefig("images/latest_camera_image_threshold.png")
        # positions = np.zeros((self.number_particles, 2))
        positions= np.array([])
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            positions=np.append(positions, np.array([[x, y]]),axis=0) if positions.size else np.array([[x, y]])
        return positions

    def orthogonal_regression_svd(self, x, y):
        data = np.column_stack((x, y))

        mean = np.mean(data, axis=0)
        centered_data = data - mean

        _, _, Vt = np.linalg.svd(centered_data)

        direction = Vt[0]  # Erste Zeile von Vt entspricht der Hauptachse

        if np.abs(direction[0]) < 1e-10:
            slope = np.inf
            intercept = np.nan
        else:
            slope = direction[1] / direction[0]
            intercept = mean[1] - slope * mean[0]

        distances = np.abs(
            direction[1] * (x - mean[0]) - direction[0] * (y - mean[1])
        ) / np.linalg.norm(direction)

        rmse = np.sqrt(np.mean(distances**2))

        return slope, intercept, rmse

    def regression_error(self, positions: np.ndarray) -> float:
        """
        Performs regression on position of colloids and returns the error.
        Minimize this.

        Args:
            colloids (list): Particle in the system

        Returns:
            float: MSE of regression
        """
        x = positions[:, 0]
        y = positions[:, 1]
        *_, residual = self.orthogonal_regression_svd(x, y)
        return residual

    def angle_between_particles(self, positions: np.ndarray) -> float:
        """
        Calculate the angle between two particles and the x-axis and takes the std of all pairs.
        Minimize this.

        Args:
            colloids (list): Particle in the system

        Returns:
            float: Angle between the particles
        """
        x = positions[:, 0]
        y = positions[:, 1]
        angle = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
        std_angle = np.std(angle)
        return std_angle

    def __call__(self, image: np.ndarray) -> float:
        
        if image is None:
            logger.warning("No image provided.")
            return 0
        image = np.array(image, dtype=np.uint8)
        positions = self.detect_blobs(image)
        regression_error = self.regression_error(positions)/self.regression_normalization
        angle_error = self.angle_between_particles(positions)/self.angle_noramlization
        if self.old_residual is None and self.old_angle_error is None:
            self.old_residual = regression_error
            self.old_angle_error = angle_error
            return 0
        reward = - regression_error - angle_error
        self.old_residual = regression_error
        self.old_angle_error = angle_error
        return reward
