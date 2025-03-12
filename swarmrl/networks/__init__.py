"""
Helper module to instantiate several modules.
"""

from swarmrl.networks.flax_network import FlaxModel
from swarmrl.networks.continuous_action_network import ContinuousActionModel
from swarmrl.networks.continuous_critic_network import ContinuousCriticModel

__all__ = [FlaxModel.__name__, 
            ContinuousActionModel.__name__,
            ContinuousCriticModel.__name__,
           ]
