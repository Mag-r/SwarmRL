"""
Position observable computer.
"""

from abc import ABC

import jax.numpy as np

from swarmrl.observables.observable import Observable


class CarImage(Observable, ABC):
    """
    Position in box observable.
    """

    def __init__(self, particle_type: int = 0):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        super().__init__(particle_type=particle_type)
        self.image_resolution = (96, 96)

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale.

        Parameters
        ----------
        image : np.ndarray
                Image to convert to grayscale.

        Returns
        -------
        np.ndarray
                Grayscale image.
        """
        return np.dot(image[..., :3], np.array([0.2989, 0.5870, 0.1140]))

    def compute_observable(self, image: np.ndarray) -> np.ndarray:
        image = self.convert_to_grayscale(image)
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std
        return np.asarray(
            image.reshape(1, self.image_resolution[0], self.image_resolution[1], 1)
        )
