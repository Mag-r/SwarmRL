"""
Position observable computer.
"""

from abc import ABC

import jax.numpy as np

from swarmrl.observables.observable import Observable
import logging

logger = logging.getLogger(__name__)

class SimpleObservable(Observable, ABC):
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

    def compute_observable(self, input: int) -> np.ndarray:
        input = np.array([input])
        input = input[np.newaxis, np.newaxis, ...]
        return input
