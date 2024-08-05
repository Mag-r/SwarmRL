"""
Main module for actions.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class Action:
    """
    Holds the values which are applied to the magnetic Field for Gaurav`s microbots.
    """

    id = 0
    magnitude: float
    phase: float 
