"""
Position observable computer.
"""

from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.observables.observable import Observable


class ImageObservable(Observable, ABC):
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



    def compute_observable(self, colloids: np.ndarray) -> np.ndarray:
        """
        Abuse of the compute_observable method to return the image directly obtained from the experiment.
        

        Parameters
        ----------
        colloids : np.ndarray Image of the experiment.
        """
        return colloids
