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
    magnetic_field: np.ndarray = np.array([0.0, 0.0])
    keep_magnetic_field: float = 1.0
