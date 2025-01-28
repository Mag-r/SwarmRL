"""
Data class for the colloid agent.
"""

import dataclasses

import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Raft:
    """
    Wrapper class for a colloid object.
    """

    pos: np.array
    alpha: float  # in radians
    magnetic_moment: float
    rotational_velocity: float = 0.0
    def get_director(self) -> np.array:
        return np.array([np.cos(self.alpha), np.sin(self.alpha)])

    def __repr__(self):
        """
        Return a string representation of the colloid.
        """
        return (
            f"Colloid(pos={self.pos}, alpha={self.alpha},"
            f" magnetic moment = {self.magnetic_moment})"
        )

    def __eq__(self, other):
        return self.id == other.id

    def tree_flatten(self):
        """
        Flatten the PyTree.
        """
        children = (self.pos, self.director, self.id, self.velocity, self.type)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the PyTree.
        """
        return cls(*children)
