"""
Module for the Actor-Critic RL protocol.
"""

import logging
import os
import pickle
import typing
from threading import Lock

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
        lock: Lock = Lock(),
        resolution_map: int = 64,
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        particle_type : int
                Particle type of the agent.
        actor_network : Network
                Actor network to use.
        critic_network : Network
                Critic network to use (includes the target).
        task : Task
                Task to use for the agent.
        observable : Observable
                Observable to use for the agent.
        loss : Loss (default=GlobalPolicyGradientLoss())
                Loss function to use for the agent.
        train : bool (default=True)
                Whether or not to train the agent.
        intrinsic_reward : IntrinsicReward (default=None)
                Intrinsic reward to use for the agent.
        max_samples_in_trajectory : int (default=200)
                Maximum number of samples saved in the trajectory.
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
        self.lock = lock

        # Trajectory to be updated.
        self.trajectory = GlobalTrajectoryInformation()
        self.max_samples_in_trajectory = max_samples_in_trajectory
        self.resolution_map = resolution_map

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
        The main learning step for the agent. Performs one training step.

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
        # with self.lock:
        #     self.remove_old_data(
        #         self.trajectory.feature_sequence.shape[0] - self.max_samples_in_trajectory
        #     )

        # self.trajectory = GlobalTrajectoryInformation()
        self.actor_network.split_rng_key()
        self.critic_network.split_rng_key()
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
        self, directory: str = "training_data", identifier: str = "trajectory"
    ):
        """
        Save the trajectory of the agent, to be used for later training.
        Saved in /<directory>/trajectory_<identifier>.pkl

        Parameters
        ----------
        directory : str
                Location to save the trajectory.
        identifier : str
                Identifier for the trajectory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        trajectory_data = {
            "features": self.trajectory.features,
            "feature_sequence": self.trajectory.feature_sequence,
            "actions": self.trajectory.actions,
            "rewards": self.trajectory.rewards,
            "carry": self.trajectory.carry,
            "next_features": self.trajectory.next_features,
            "next_carry": self.trajectory.next_carry,
            "action_sequence": self.trajectory.action_sequence,
            "killed": self.trajectory.killed,
        }

        filename = os.path.join(directory, f"trajectory_{identifier}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(trajectory_data, f)

    def restore_trajectory(
        self, directory: str = "training_data", identifier: str = "trajectory"
    ):
        """
        Restore the trajectory of the agent.

        Parameters
        ----------
        directory : str
                Location to restore the trajectory.
        identifier : str
                Identifier for the trajectory.
        """
        filename = os.path.join(directory, f"trajectory_{identifier}.pkl")
        with open(filename, "rb") as f:
            trajectory_data = pickle.load(f)

        self.trajectory.features = trajectory_data["features"]
        self.trajectory.feature_sequence = trajectory_data["feature_sequence"]
        self.trajectory.actions = trajectory_data["actions"]
        self.trajectory.rewards = trajectory_data["rewards"]
        self.trajectory.carry = trajectory_data["carry"]
        self.trajectory.next_features = trajectory_data["next_features"]
        self.trajectory.next_carry = trajectory_data["next_carry"]
        self.trajectory.action_sequence = trajectory_data["action_sequence"]
        self.trajectory.killed = trajectory_data["killed"]
        logger.info(
            f"shape of all features: {np.shape(self.trajectory.features)}, shape of all actions: {np.shape(self.trajectory.actions)}, shape of all rewards: {np.shape(self.trajectory.rewards)}"
        )

    def remove_old_data(self, remove: int = -1):
        """Remove old data from the trajectory. The last 10 are always kept.

        Args:
            remove (int): Number of elements to remove.
        """
        logger.info(
            f"shape of all features: {np.shape(self.trajectory.features)}, shape of all actions: {np.shape(self.trajectory.actions)}, shape of all rewards: {np.shape(self.trajectory.rewards)}, shape of feature sequence{np.shape(self.trajectory.feature_sequence)}, shape of action sequence{np.shape(self.trajectory.action_sequence)}, shape of next features{np.shape(self.trajectory.next_features)}"
        )
        if remove == -1:
            remove = (
                len(self.trajectory.feature_sequence) - self.max_samples_in_trajectory
            )
        if remove > 0 and self.loss.error_predicted_reward is not None:
            indices = np.arange(len(self.loss.error_predicted_reward))
            probabilities = jnp.array(self.loss.error_predicted_reward)
            probabilities = 1 / probabilities
            probabilities = probabilities.at[-10:].set(0)

            key = jax.random.PRNGKey(0)
            remove = np.min([remove, indices.shape[0] - 10])
            selected_indices = jax.random.choice(
                key, indices, shape=(remove,), replace=False, p=probabilities
            )
            selected_indices = np.array(
                jax.device_get(selected_indices), dtype=int
            )  # Ensure selected_indices is a NumPy array
            # Get the list of all indices without the selected ones
            all_indices = set(np.arange(len(self.trajectory.feature_sequence)))
            selected_indices = set(selected_indices)
            selected_indices = np.array(list(all_indices - selected_indices), dtype=int)

            # Remove elements at the specified indices
            self.trajectory.feature_sequence = [
                self.trajectory.feature_sequence[i] for i in selected_indices
            ]
            self.trajectory.next_features = [
                self.trajectory.next_features[i]
                for i in selected_indices
                if i < len(self.trajectory.next_features)
            ]
            self.trajectory.occupancy_map = [
                self.trajectory.occupancy_map[i] for i in selected_indices
            ]
            self.trajectory.next_occupancy_map = [
                self.trajectory.next_occupancy_map[i]
                for i in selected_indices
                if i < len(self.trajectory.next_occupancy_map)
            ]
            self.trajectory.features = self.trajectory.features[:, selected_indices]
            self.trajectory.carry = [self.trajectory.carry[i] for i in selected_indices]
            self.trajectory.next_carry = [
                self.trajectory.next_carry[i]
                for i in selected_indices
                if i < len(self.trajectory.next_carry)
            ]
            self.trajectory.actions = self.trajectory.actions[selected_indices]
            self.trajectory.action_sequence = [
                self.trajectory.action_sequence[i] for i in selected_indices
            ]
            self.trajectory.rewards = [
                self.trajectory.rewards[i] for i in selected_indices
            ]
            logger.info(
                f"AFter removing:: shape of all features: {np.shape(self.trajectory.features)}, shape of all actions: {np.shape(self.trajectory.actions)}, shape of all rewards: {np.shape(self.trajectory.rewards)}, shape of feature sequence{np.shape(self.trajectory.feature_sequence)}, shape of action sequence{np.shape(self.trajectory.action_sequence)}, shape of next features{np.shape(self.trajectory.next_features)}"
            )

    def initialize_network(self):
        """
        Initialize all of the models in the gym.
        """
        self.actor_network.reinitialize_network()
        self.critic_network.reinitialize_network()

    def save_agent(self, directory: str = "Models", identifier: str = "") -> None:
        """
        Save the agent network state.

        Parameters
        ----------
        directory : str
                Location to save the models.
        identifier : str
                Identifier for the models.
        """
        self.actor_network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}_actor_{identifier}",
            directory=directory,
        )
        self.critic_network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}_critic_{identifier}",
            directory=directory,
        )

    def restore_agent(self, directory: str = "Models", identifier: str = "") -> None:
        """
        Restore the agent state from a directory.


        Parameters
        ----------
        directory : str
                Location to restore the models.
        identifier : str
                Identifier for the models.
        """
        self.actor_network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}_actor_{identifier}",
            directory=directory,
        )
        self.critic_network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}_critic_{identifier}",
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
                External reward to add to the reward. Needed for Benchmark.

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
            with self.lock:
                self.trajectory.rewards.append(reward)
        return reward

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Copmute the new state for the agent.

        Returns the chosen action to the force function which
        talks to the engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.

        Returns
        -------
        action : List[Action]
                Action to take.
        """
        state_description, latest_observation = self.state_description(colloids)
        if colloids is None:
            colloids = latest_observation  # For experiments without colloids.
        previous_carry = self.actor_network.carry
        previous_actions = self.assemble_previous_actions()
        occupancy_map = self.add_occupancy_map(colloids)
        action = self.actor_network.compute_action(
            observables=np.array(state_description),
            previous_actions=np.array(previous_actions),
            occupancy_map=occupancy_map,
        )[np.newaxis, np.shape(state_description)[0] - 1]
        next_carry = self.actor_network.carry
        # Update the trajectory information.

        if self.train:
            with self.lock:
                self.trajectory.feature_sequence.append(state_description)
                self.trajectory.occupancy_map.append(occupancy_map)
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
                    self.trajectory.next_occupancy_map.append(occupancy_map)
                    self.trajectory.next_carry.append(next_carry)
        self.kill_switch = self.task.kill_switch
        return np.squeeze(action)

    def assemble_previous_actions(self) -> np.ndarray:
        """Orders the previous actions to be used in the network.
        If the trajectory is smaller than the sequence length, it
        fills the previous actions with zeros.

        Returns:
            np.ndarray: Previous actions to be used in the network.
        """
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

    def state_description(self, colloids) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the observable and concate it into the state description.
        It also updates the trajectory with the new observable.

        Args:
            colloids (_type_): The colloids to be used in the observable.

        Returns:
            tuple[np.ndarray, np.ndarray]: The state description and the latest observable.
        """
        latest_observable = self.observable.compute_observable(colloids)
        latest_observable = np.expand_dims(latest_observable, axis=1)
        if self.trajectory.features.size == 0 and self.train:
            self.trajectory.features = latest_observable
        else:
            with self.lock:
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

    def add_occupancy_map(self, colloids: typing.List[Colloid]):
        """
        Add an occupancy map to the trajectory.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        occupancy_map : jnp.ndarray
                Occupancy map to add.
        """

        with self.lock:
            if self.trajectory.occupancy_map.size == 0:
                x = self.trajectory.features[0, -1, :, 0]
                y = self.trajectory.features[0, -1, :, 1]

                x = (x * self.resolution_map / self.observable.resolution[0]).astype(
                    int
                )
                y = (y * self.resolution_map / self.observable.resolution[1]).astype(
                    int
                )
                occupancy_map = np.zeros(
                    (self.resolution_map, self.resolution_map), dtype=np.int32
                )
                occupancy_map[x, y] += 1

            else:
                x = self.trajectory.features[0, -1, :, 0]
                y = self.trajectory.features[0, -1, :, 1]

                x = (x * self.resolution_map / self.observable.resolution[0]).astype(
                    int
                )
                y = (y * self.resolution_map / self.observable.resolution[1]).astype(
                    int
                )
                occupancy_map = self.trajectory.occupancy_map[-1].copy()
                occupancy_map[x, y] += 1
        return occupancy_map[np.newaxis, ...]
