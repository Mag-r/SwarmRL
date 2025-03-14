"""
Module for the Actor-Critic RL protocol.
"""

import logging
import typing

import jax
import numpy as np
from jax import numpy as jnp

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward
from swarmrl.losses import GlobalPolicyGradientLoss, Loss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import GlobalTrajectoryInformation
import os
import pickle

logger = logging.getLogger(__name__)


class MPIActorCriticAgent(Agent):
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        actor_network: Network,
        critic_network: Network,
        task: Task,
        observable: Observable,
        loss: Loss = GlobalPolicyGradientLoss(),
        train: bool = True,
        intrinsic_reward: IntrinsicReward = None,
        max_samples_in_trajectory: int = 200,
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
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.train = train
        self.loss = loss
        self.intrinsic_reward = intrinsic_reward

        # Trajectory to be updated.
        self.trajectory = GlobalTrajectoryInformation()
        self.max_samples_in_trajectory = max_samples_in_trajectory
        # self.trajectory.actions = np.array([0,0,0.1])
        # self.trajectory.actions = np.expand_dims(self.trajectory.actions, axis=0)

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
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            episode_data=self.trajectory,
        )
        self.actor_network.split_rng_key()
        logger.debug("Loss computed.")
        # Update the intrinsic reward if set.
        if self.intrinsic_reward:
            self.intrinsic_reward.update(self.trajectory)

        # Reset the trajectory storage.
        self.remove_old_data(
            self.trajectory.features.shape[1] - self.max_samples_in_trajectory
        )
        logger.debug(
            f"Shape of all saved properties in trajectory {np.array(self.trajectory.features).shape=}, {np.array(self.trajectory.actions).shape=}, {np.array(self.trajectory.rewards).shape=}, {np.array(self.trajectory.carry).shape=}, {np.array(self.trajectory.next_features).shape=}, {np.array(self.trajectory.next_carry).shape=}, {np.array(self.trajectory.action_sequence).shape=}"
        )
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

    def save_trajectory(
        self, directory: str = "training_data", name: str = "trajectory"
    ):
        """
        Save the trajectory of the agent.

        Parameters
        ----------
        directory : str
                Location to save the trajectory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        trajectory_data = {
            "features": self.trajectory.features,
            "actions": self.trajectory.actions,
            "rewards": self.trajectory.rewards,
            "carry": self.trajectory.carry,
            "next_features": self.trajectory.next_features,
            "next_carry": self.trajectory.next_carry,
            "action_sequence": self.trajectory.action_sequence,
            "killed": self.trajectory.killed,
        }

        filename = os.path.join(directory, f"{name}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(trajectory_data, f)

    def remove_old_data(self, remove: int):
        """Remove old data from the trajectory. The last 10 are always kept.

        Args:
            remove (int): Number of elements to remove.
        """
        if remove > 0:
            indices = np.arange(len(self.loss.error_predicted_reward))
            probabilities = jnp.array(self.loss.error_predicted_reward)
            probabilities = 1 / probabilities
            probabilities = probabilities.at[-10:].set(0)

            key = jax.random.PRNGKey(0)
            selected_indices = jax.random.choice(
                key, indices, shape=(remove,), replace=False, p=probabilities
            )
            selected_indices = np.array(
                jax.device_get(selected_indices), dtype=int
            )  # Ensure selected_indices is a NumPy array

            # Get the list of all indices without the selected ones
            all_indices = set(indices)
            selected_indices = set(selected_indices)
            selected_indices = np.array(list(all_indices - selected_indices), dtype=int)

            # Remove elements at the specified indices
            self.trajectory.feature_sequence = [
                self.trajectory.feature_sequence[i] for i in selected_indices
            ]
            self.trajectory.next_features = [
                self.trajectory.next_features[i]
                for i in selected_indices
                if i < len(self.trajectory.next_features) - 1
            ]
            self.trajectory.features = self.trajectory.features[:, selected_indices]
            self.trajectory.carry = [self.trajectory.carry[i] for i in selected_indices]
            self.trajectory.next_carry = [
                self.trajectory.next_carry[i]
                for i in selected_indices
                if i < len(self.trajectory.next_carry) - 1
            ]
            self.trajectory.actions = self.trajectory.actions[selected_indices]
            self.trajectory.action_sequence = [
                self.trajectory.action_sequence[i] for i in selected_indices
            ]
            self.trajectory.rewards = [
                self.trajectory.rewards[i] for i in selected_indices
            ]

    def initialize_network(self):
        """
        Initialize all of the models in the gym.
        """
        self.actor_network.reinitialize_network()
        self.critic_network.reinitialize_network()

    def save_agent(self, directory: str = "Models"):
        """
        Save the agent network state.

        Parameters
        ----------swarmrl/losses/global_policy_gradient_loss.py
        directory : str
                Location to save the models.
        """
        self.actor_network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}_actor",
            directory=directory,
        )
        self.critic_network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}_critic",
            directory=directory,
        )

    def restore_agent(self, directory: str = "Models"):
        """
        Restore the agent state from a directory.
        """
        self.actor_network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}_actor",
            directory=directory,
        )
        self.critic_network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}_critic",
            directory=directory,
        )

    def calc_reward(
        self, colloids: typing.List[Colloid], external_reward: float = 0.0
    ) -> float:
        """
        Compute the reward for the agent.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        external_reward : float (default=0.0)
                External reward to add to the reward. Neede for Benchmark.

        Returns
        -------
        reward : float
                Reward for the agent.
        """
        if colloids is None:
            colloids = self.observable.compute_observable(None)
        reward = self.task(colloids)
        reward += external_reward
        if self.intrinsic_reward:
            reward += self.intrinsic_reward.compute_reward(episode_data=self.trajectory)
        if self.train:
            self.trajectory.rewards.append(reward)

        return reward

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Copmute the new state for the agent.

        Returns the chosen action to the force function which
        talks to the espresso engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        """
        state_description, latest_observation = self.state_description(colloids)
        if colloids is None:
            colloids = latest_observation  # For experiments without colloids.
        previous_carry = self.actor_network.carry
        previous_actions = self.assemble_previous_actions()

        action = self.actor_network.compute_action(
            observables=np.array(state_description),
            previous_actions=np.array(previous_actions),
        )[np.newaxis, np.shape(state_description)[0] - 1]
        next_carry = self.actor_network.carry
        # Update the trajectory information.
        if self.train:
            self.trajectory.feature_sequence.append(state_description)
            self.trajectory.carry.append(previous_carry)

            previous_actions = np.append(
                previous_actions, np.expand_dims(action, axis=(0)), axis=1
            )
            previous_actions = np.squeeze(previous_actions)
            self.trajectory.action_sequence.append(previous_actions)
            self.trajectory.actions = (
                np.append(self.trajectory.actions, action, axis=0)
                if self.trajectory.actions.size > 0
                else action
            )
            self.trajectory.killed = self.task.kill_switch
            if len(self.trajectory.feature_sequence) > 1:
                self.trajectory.next_features.append(state_description)
                self.trajectory.next_carry.append(next_carry)
        self.kill_switch = self.task.kill_switch
        return np.squeeze(action)

    def assemble_previous_actions(self):
        if self.trajectory.actions.shape[0] >= self.actor_network.sequence_length:
            previous_actions = self.trajectory.actions[
                -self.actor_network.sequence_length :
            ]
        else:
            previous_actions = (
                self.trajectory.actions[:]
                if self.trajectory.actions.shape[0] > 0
                else np.zeros(self.actor_network.action_dimension)[np.newaxis, :]
            )

            previous_actions = np.concatenate(
                [
                    np.repeat(
                        previous_actions[0:1],
                        self.actor_network.sequence_length - previous_actions.shape[0],
                        axis=0,
                    ),
                    previous_actions,
                ],
                axis=0,
            )
        return np.expand_dims(previous_actions, axis=0)  # Add batch dimension

    def state_description(self, colloids):
        latest_observable = self.observable.compute_observable(colloids)
        latest_observable = np.expand_dims(latest_observable, axis=1)
        if self.trajectory.features.size == 0 and self.train:
            self.trajectory.features = latest_observable
        else:
            self.trajectory.features = np.append(
                self.trajectory.features, latest_observable, axis=1
            )

        if self.trajectory.features.shape[1] >= self.actor_network.sequence_length:
            state_description = self.trajectory.features[
                :, -self.actor_network.sequence_length :
            ]
        else:
            state_description = self.trajectory.features[:]
            state_description = np.concatenate(
                [
                    np.repeat(
                        state_description[:, 0:1],
                        self.actor_network.sequence_length - state_description.shape[1],
                        axis=1,
                    ),
                    state_description,
                ],
                axis=1,
            )

        return state_description, latest_observable
