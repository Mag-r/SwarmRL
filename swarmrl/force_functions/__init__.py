"""
Force functions module.

These functions talk to the environment and return a force to be applied to the
particle.
"""

from swarmrl.force_functions.force_fn import ForceFunction
from swarmrl.force_functions.global_force_fn import GlobalForceFunction

__all__ = [ForceFunction.__name__, GlobalForceFunction.__name__]
