import jax.numpy as jnp
import numpy as np
from jax import random
import logging
import cv2
from flax import linen as nn
from swarmrl.tasks.task import Task
from matplotlib import pyplot as plt
import scipy as sc

logger = logging.getLogger(__name__)


class ExperimentTask(Task):

    def __init__(self, number_particles: int):
        super().__init__(particle_type=0)
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.detector_params.filterByArea = True
        self.detector_params.maxArea = 50
        self.detector_params.minArea = 25
        self.detector_params.filterByColor = True
        self.detector_params.blobColor = 255
        self.detector_params.filterByConvexity = False
        self.detector_params.filterByInertia = False
        self.detector_params.filterByCircularity = True
        self.detector_params.minCircularity = 0.7
        self.blob_detector = cv2.SimpleBlobDetector_create(
            self.detector_params
        )
        self.large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.number_particles = number_particles
        self.old_residual = None
        self.old_angle_error = None
        self.angle_normalization = 1.62
        self.regression_normalization = 1.5

    def detect_blobs(self, image: np.ndarray) -> np.ndarray:
        image = image[0,:,:,0]
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std
        image = cv2.erode(image, self.large_kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.small_kernel, iterations=3)
        image = image > 1.9
        image = np.array(image * 255, dtype=np.uint8)

        keypoints = self.blob_detector.detect(image)
        logger.info(f"Detected {len(keypoints)} particles.")

        # image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(image_with_keypoints, cmap="gray")
        # plt.savefig("images/latest_camera_image.png")
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

        direction = Vt[0]  

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

    def angle_between_particles(self, x, y) -> float:
        """
        Calculate the angle between two particles and the x-axis and takes the std of all pairs.
        Minimize this.

        Args:
            colloids (list): Particle in the system

        Returns:
            float: Angle between the particles
        """
        idx = np.triu_indices(len(x), k=1)
        dx = x[idx[0]] - x[idx[1]]
        dy = y[idx[0]] - y[idx[1]]
        angles = np.arctan2(dy, dx)
        angle = np.array(angles) % np.pi
        std_angle = sc.stats.circstd(angle)
        return std_angle

    def __call__(self, positions: np.ndarray) -> float:
        x = positions[0, :, 0]
        y = positions[0, :, 1]
        *_, regression_error = self.orthogonal_regression_svd(x, y)
        regression_error /= self.regression_normalization
        angle_error = self.angle_between_particles(x, y) / self.angle_normalization
        if self.old_residual is None and self.old_angle_error is None:
            self.old_residual = regression_error
            self.old_angle_error = angle_error
            return 0
        reward_regression = regression_error# - self.old_residual
        reward_angle = angle_error #- self.old_angle_error
        logger.info(f"Reward regression: {reward_regression}, reward angle: {reward_angle}")
        self.old_residual = regression_error
        self.old_angle_error = angle_error
        return -(reward_regression + reward_angle)
