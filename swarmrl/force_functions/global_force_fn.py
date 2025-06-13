"""
Espresso interaction model capable of handling a neural network as a function.
"""

import typing

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.components.colloid import Colloid
import logging

logger = logging.getLogger(__name__)


class GlobalForceFunction:
    """
    Class to bridge agents with an engine. Selects only one action for the whole system. (For Gaurav Sim)
    """

    _kill_switch: bool = False

    def __init__(
        self,
        agents: dict,
    ):
        """
        Constructor for the NNModel.

        Parameters
        ----------
        agents : dict
            Agents used in the simulations.
        """
        super().__init__()
        self.agents = agents

        # Used in the data saving.
        self.particle_types = [type_ for type_ in self.agents]

    @property
    def kill_switch(self):
        """
        If true, kill the simulation.
        """
        return self._kill_switch

    @kill_switch.setter
    def kill_switch(self, value):
        """
        Set the kill switch.
        """
        self._kill_switch = value
        
    def calc_reward(self, colloids: typing.List[Colloid], external_reward: float = 0.0):
        for agent in self.agents:
            self.agents[agent].calc_reward(colloids=colloids, external_reward=external_reward)
            

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Compute the state of the system based on the current colloid position.

        In the case of the ML models, this method undertakes the following steps:

        1. Compute observable
        2. Compute action probabilities
        3. Compute action

        Returns
        -------
        action: Action
                Return the action the colloid should take.
        kill_switch : bool
                Flag capable of ending simulation.
        """
        # Prepare the data storage.
        action = []
        switches = []
        for agent in self.agents:
            action = self.agents[agent].calc_action(colloids=colloids)
            switches.append(self.agents[agent].kill_switch)

        self.kill_switch = any(switches)
        return action

    def set_training_mode(self, training: bool = True):
        """
        Set the training mode for the agents.

        Parameters
        ----------
        training : bool
                If true, set the training mode.
        """
        for agent in self.agents:
            self.agents[agent].train = training

    def save_agents(self, directory: str = "Models"):
        """
        Save the agent network state.

        Parameters
        ----------
        directory : str
                Location to save the models.
        """
        for agent in self.agents:
            self.agents[agent].save_agent(directory=directory)

    def save_camera_image(self):
        """
        Save the camera image.
        """
        for agent in self.agents:
            self.agents[agent].observable.compute_observable(
                colloids=None
            )