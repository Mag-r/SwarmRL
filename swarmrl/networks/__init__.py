"""
Helper module to instantiate several modules.
"""

from swarmrl.networks.flax_network import FlaxModel
from swarmrl.networks.continuous_flax_network import ContinuousFlaxModel

__all__ = [FlaxModel.__name__, 
           ContinuousFlaxModel.__name__,
           ]
