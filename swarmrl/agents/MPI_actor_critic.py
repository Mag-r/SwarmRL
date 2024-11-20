"""
Module for the Actor-Critic RL protocol.
"""

import typing

import numpy as np
import logging
import jax

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward
from swarmrl.losses import Loss, GlobalPolicyGradientLoss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import GlobalTrajectoryInformation


logger = logging.getLogger(__name__)

class MPIActorCriticAgent(Agent):
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        network: Network,
        task: Task,
        observable: Observable,
        loss: Loss = GlobalPolicyGradientLoss(),
        train: bool = True,
        intrinsic_reward: IntrinsicReward = None,
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        loss : Loss (default=ProximalPolicyLoss)
                Loss function to use to update the networks.
        train : bool (default=True)
                Flag to indicate if the agent is training.
        intrinsic_reward : IntrinsicReward (default=None)
                Intrinsic reward to use for the agent.
        """
        # Properties of the agent.
        self.network = network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.train = train
        self.loss = loss
        self.intrinsic_reward = intrinsic_reward

        # Trajectory to be updated.
        self.trajectory = GlobalTrajectoryInformation()

    def __name__(self) -> str:
        """
        Give the class a name.

        Return
        ------
        name : str
            Name of the class.
        """
        return "ActorCriticAgent"

    def update_agent(self) -> tuple:
        """
        Update the agents network.

        Returns
        -------
        rewards : float
                Net reward for the agent.
        killed : bool
                Whether or not this agent killed the
                simulation.
        """
        # Collect data for returns.
        rewards = self.trajectory.rewards
        killed = self.trajectory.killed

        # Compute loss for actor and critic.
        logger.debug("Computing loss.")
        self.loss.compute_loss(
            network=self.network,
            episode_data=self.trajectory,
        )
        logger.debug("Loss computed.")
        # Update the intrinsic reward if set.
        if self.intrinsic_reward:
            self.intrinsic_reward.update(self.trajectory)

        # Reset the trajectory storage.
        self.remove_old_data(self.trajectory.features.shape[1] - 150)
        # self.trajectory = GlobalTrajectoryInformation()
        
        return rewards, killed

    def reset_agent(self, colloids: typing.List[Colloid]):
        """
        Reset several properties of the agent.

        Reset the observables and tasks for the agent.

        Parameters
        ----------
        colloids : typing.List[Colloid]
                Colloids to use in the initialization.
        """
        self.observable.initialize(colloids)
        self.task.initialize(colloids)

    def reset_trajectory(self):
        """
        Set all trajectory data to None.
        """
        self.task.kill_switch = False  # Reset here.
        self.trajectory = GlobalTrajectoryInformation()
        
        
    def remove_old_data(self, remove: int):
        """Remove old data from the trajectory.

        Args:
            keep (int): number of elements to remove.
        """
        if remove > 0:
            indices = np.arange(self.trajectory.features.shape[1])
            probabilities = self.trajectory.error_predicted_reward
            probabilities = 1 / probabilities
            probabilities = probabilities.at[-10:].set(0)
            key = jax.random.PRNGKey(0)
            selected_indices = jax.random.choice(key, indices, shape=(remove,), replace=False, p=probabilities)
            selected_indices = np.array(jax.device_get(selected_indices), dtype=int)  # Ensure selected_indices is a NumPy array

            # Get the list of all indices without the selected ones
            all_indices = set(indices)
            selected_indices = set(selected_indices)
            selected_indices = np.array(list(all_indices - selected_indices), dtype=int)

            logger.info(f"Selected indices to keep: {selected_indices} out of total {len(indices)}")
            # Remove elements at the specified indices
            self.trajectory.feature_sequence = [self.trajectory.feature_sequence[i] for i in selected_indices]
            self.trajectory.next_features = [self.trajectory.next_features[i] for i in selected_indices if i < len(indices)-1]
            self.trajectory.features = self.trajectory.features[:, selected_indices]
            self.trajectory.carry = [self.trajectory.carry[i] for i in selected_indices]
            self.trajectory.actions = [self.trajectory.actions[i] for i in selected_indices]
            self.trajectory.log_probs = [self.trajectory.log_probs[i] for i in selected_indices]
            self.trajectory.rewards = [self.trajectory.rewards[i] for i in selected_indices]
            self.trajectory.error_predicted_reward = self.trajectory.error_predicted_reward[selected_indices]
            # Check if the last element is in the selected indices and remove it if necessary

                

        
    def initialize_network(self):
        """
        Initialize all of the models in the gym.
        """
        self.network.reinitialize_network()

    def save_agent(self, directory: str = "Models"):
        """
        Save the agent network state.

        Parameters
        ----------
        directory : str
                Location to save the models.
        """
        self.network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def restore_agent(self, directory: str = "Models"):
        """
        Restore the agent state from a directory.
        """
        self.network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Copmute the new state for the agent.

        Returns the chosen action to the force function which  #check if this is
        talks to the espresso engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        """
        state_description = self.state_description(colloids)
        logger.debug(f"State description shape: {state_description.shape}")
        previous_carry = self.network.carry
        action, log_probs = self.network.compute_action(
            observables=np.array(state_description)
        )

        # Compute extrinsic rewards.
        rewards = self.task(colloids)
        # Compute intrinsic rewards if set.
        if self.intrinsic_reward:
            rewards += self.intrinsic_reward.compute_reward(
                episode_data=self.trajectory
            )

        # Update the trajectory information.
        if self.train:
            self.trajectory.feature_sequence.append(state_description)
            self.trajectory.carry.append(previous_carry)
            self.trajectory.actions.append(action)
            self.trajectory.log_probs.append(log_probs)
            self.trajectory.rewards.append(rewards)
            self.trajectory.killed = self.task.kill_switch
            if len(self.trajectory.feature_sequence)> 1:
                self.trajectory.next_features.append(state_description)
            
        self.kill_switch = self.task.kill_switch

        return action

    def state_description(self, colloids):
        latest_observable = self.observable.compute_observable(colloids)
        latest_observable = np.expand_dims(latest_observable, axis=1)
        if self.trajectory.features.size == 0 and self.train:
            self.trajectory.features = latest_observable
        else:
            self.trajectory.features = np.append(self.trajectory.features, latest_observable,axis=1)
        
        if self.trajectory.features.shape[1] >= self.network.sequence_length:
            state_description = self.trajectory.features[:,-self.network.sequence_length:]      
        else:
            state_description = self.trajectory.features[:]
            state_description = np.concatenate(
                [
                    np.repeat(state_description[:, 0:1], self.network.sequence_length - state_description.shape[1], axis=1),
                    state_description,
                ],
                axis=1,
            )
            
        return state_description
