"""
Position observable computer.
"""

from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.observables.observable import Observable


class PosRotationObservable(Observable, ABC):
    """
    Position in box observable.
    """

    def __init__(self, box_length: np.ndarray, particle_type: int = 0):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        super().__init__(particle_type=particle_type)
        self.box_length = box_length
        self.old_obs = None

    def single_pos(self, index: int, colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        index : int
                Index of the colloid for which the observable should be computed.
        colloids : list
                Colloids in the system.
        """
        colloid = colloids[index]

        data = onp.copy(colloid.pos[:2])

        return np.array(data) / self.box_length[:2]
    
    def single_angle(self, index: int, colloids: list):
        """
        Compute the angle of the colloid.

        Parameters
        ----------
        index : int
                Index of the colloid for which the observable should be computed.
        colloids : list
                Colloids in the system.
        """
        colloid = colloids[index]

        data = onp.copy(colloid.alpha)

        return np.array(data)/(2*np.pi)

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the current state observable for all colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.
        """
        indices = range(len(colloids))
        all_pos = np.array([self.single_pos(i, colloids) for i in indices])
        all_angle = np.array([self.single_angle(i, colloids) for i in indices])
        observable = np.concatenate([all_pos.flatten(), all_angle.flatten()]).flatten()
        
        difference = observable - self.old_obs if self.old_obs is not None else observable
        self.old_obs = observable
        observable = np.concatenate([observable, difference]).flatten()
        observable = np.expand_dims(observable, axis=0)
        return np.array(observable)
