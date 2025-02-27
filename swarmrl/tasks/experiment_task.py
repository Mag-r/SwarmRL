import jax.numpy as np
from jax import random
import logging
import cv2
from flax import linen as nn
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class ExperimentTask(Task):

    def __init__(self, number_particles: int):
        super().__init__(particle_type=0)
        self.first_sweep_detector_params = cv2.SimpleBlobDetector_Params()
        self.first_sweep_detector_params.filterByArea = True
        self.first_sweep_detector_params.maxArea = 130
        self.first_sweep_detector_params.minArea = 50
        self.first_sweep_detector_params.filterByColor = True
        self.first_sweep_detector_params.blobColor = 255
        self.second_sweep_detector_params = cv2.SimpleBlobDetector_Params()
        self.first_sweep_detector_params.filterByArea = True
        self.first_sweep_detector_params.maxArea = 2000
        self.first_sweep_detector_params.minArea = 50
        self.first_detector = cv2.SimpleBlobDetector_create(
            self.first_sweep_detector_params
        )
        self.second_detector = cv2.SimpleBlobDetector_create(
            self.second_sweep_detector_params
        )
        self.number_particles = number_particles
        self.old_residual = None

    def detect_blobs(self, image: np.ndarray) -> np.ndarray:
        positions = np.zeros((self.number_particles, 2))
        image = image > 1
        image = np.array(image * 255, dtype=np.uint8)
        image = nn.avg_pool(image, (2, 2), strides=(1, 1), padding="Same") == 63
        image = np.array(image * 255, dtype=np.uint8)
        keypoints = self.first_detector.detect(image)
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            positions[i] = np.array([x, y])
            radius = int(kp.size / 2 * 1.6)
            cv2.circle(image, (x, y), radius, (0, 0, 0), -1)
        number_detected_single_particles = len(keypoints)
        keypoints = self.second_detector.detect(image)
        positions[number_detected_single_particles:] = np.array(
            [[int(kp.pt[0]), int(kp.pt[1])] for kp in keypoints]
        ).repeat(2, axis=0)
        number_detected_double_particles = len(keypoints)
        logger.info(
            f"Detected {number_detected_single_particles} single particles and "
            f"{number_detected_double_particles} double particles. "
            f"\n Total number of detected particles: "
            f"{number_detected_single_particles + number_detected_double_particles*2} / {self.number_particles}"
        )
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
        positions = self.detect_blobs(image)
        regression_error = self.regression_error(positions)
        if self.old_residual is None:
            self.old_residual = regression_error
            return 0
        reward = regression_error - self.old_residual
        self.old_residual = regression_error
        return -reward - self.angle_between_particles(positions)
