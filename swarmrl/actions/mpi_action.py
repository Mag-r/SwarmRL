"""
Main module for actions.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class MPIAction:
    """
    Holds the values which are applied to the magnetic Field for Gaurav`s microbots.
    """

    id = 0
    amplitudes: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray
    offsets: np.ndarray
