"""
Jax model for reinforcement learning.
"""

import logging
import os
import pickle
from abc import ABC
from typing import List

import jax
import jax.numpy as np
import numpy as onp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from optax._src.base import GradientTransformation

from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.exploration_policies.random_exploration import RandomExploration
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.continuous_gaussian_distribution import ContinuousGaussianDistribution
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
from swarmrl.actions import MPIAction

logger = logging.getLogger(__name__)


class ContinuousTargetModel(Network, ABC):
    """
    Class for the Flax model in ZnRND.

    Attributes
    ----------
    epoch_count : int
            Current epoch stage. Used in saving the models.
    """

    def __init__(
        self,
        flax_model: nn.Module,
        input_shape: tuple,
        optimizer: GradientTransformation = None,
        exploration_policy: ExplorationPolicy = RandomExploration(probability=0.0),
        sampling_strategy: SamplingStrategy = ContinuousGaussianDistribution(),
        action_dimension: int = 3,
        rng_key: int = None,
        deployment_mode: bool = False,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        flax_model : nn.Module
                Flax model as a neural network.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        rng_key : int
                Key to seed the model with. Default is a randomly generated key but
                the parameter is here for testing purposes.
        deployment_mode : bool
                If true, the model is a shell for the network and nothing else. No
                training can be performed, this is only used in deployment.
        """
        if rng_key is None:
            rng_key = onp.random.randint(0, 1027465782564)
        self.sampling_strategy = sampling_strategy
        self.model = flax_model

        self.apply_fn = self.model.apply #jax.jit(jax.vmap(self.model.apply, in_axes=(None, 0, 0, None))) #erstes argument nicht; zwweites in erster ache, drittes nicht
        self.input_shape = input_shape
        self.model_state = None
        self.action_dimension = action_dimension
        self.sequence_length = self.input_shape[1]
        self.carry = None
        # initialize the model state
        init_rng = jax.random.PRNGKey(rng_key)
        _, subkey = jax.random.split(init_rng)
        self.optimizer = optimizer
        self.model_state = self._create_train_state(subkey)
        self.deployment_mode = deployment_mode
        if not deployment_mode:
            self.exploration_policy = exploration_policy
            self.epoch_count = 0

    def _create_custom_train_state(self, optimizer: dict):
        """
        Deal with the optimizers in case of complex configuration.
        """
        return type("TrainState", (TrainState,), optimizer)

    def _create_train_state(self, init_rng: int) -> TrainState:
        """
        Create a training state of the model.

        Parameters
        ----------
        init_rng : int
                Initial rng for train state that is immediately deleted.

        Returns
        -------
        state : TrainState / CustomTrainState
                initial state of model to then be trained.
                If you have multiple optimizers, this will create a custom train state.
        """
        params = self.model.init(
            init_rng,
            np.ones(list(self.input_shape)),
            np.ones(list([self.input_shape[0], self.action_dimension])),
        )["params"]
        model_summary = self.model.tabulate(
            jax.random.PRNGKey(1),
            np.ones(list(self.input_shape)),
            np.ones(list([self.input_shape[0], self.action_dimension])),
        )
        print(model_summary)
        if isinstance(self.optimizer, dict):
            CustomTrainState = self._create_custom_train_state(self.optimizer)

            return CustomTrainState.create(
                apply_fn=self.model.apply, params=params, tx=self.optimizer
            )
        else:
            return TrainState.create(
                apply_fn=self.model.apply, params=params, tx=self.optimizer
            )

    def reinitialize_network(self):
        """
        Initialize the neural network.
        """
        rng_key = onp.random.randint(0, 1027465782564)
        init_rng = jax.random.PRNGKey(rng_key)
        _, subkey = jax.random.split(init_rng)
        self.model_state = self._create_train_state(subkey)

    def update_model(self, grads):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        # Logging for grads and pre-train model state
        logger.debug(f"{grads=}")
        logger.debug(f"{self.model_state=}")

        if isinstance(self.optimizer, dict):
            pass

        else:
            self.model_state = self.model_state.apply_gradients(grads=grads)

        # Logging for post-train model state
        logger.debug(f"{self.model_state=}")
        logger.debug(f"Model updated")
        self.epoch_count += 1

    def compute_q_values(self, observables: np.ndarray, actions: np.ndarray, carry: np.ndarray):
        """
        Compute the q-value of the network.

        Parameters
        ----------
        observables : np.ndarray (n_agents, observable_dimension)
                Observable for each colloid for which the action should be computed.
        actions : np.ndarray (n_agents, action_dimension)
                Actions taken by the agent.
        carry : np.ndarray
                Carry state of the model.

        Returns
        -------
        tuple : (np.ndarray, np.ndarray)
                The first element is an array of indices corresponding to the action
                taken by the agent. The value is bounded between 0 and the number of
                output neurons. The second element is an array of the corresponding
                log_probs (i.e. the output of the network put through a softmax).
        """
        try:
            first_q_values, second_q_values = self.apply_fn(
                {"params": self.model_state.params}, np.array(observables), np.array(actions), carry
            )
        except AttributeError:
            first_q_values, second_q_values = self.apply_fn(
                {"params": self.model_state["params"]}, np.array(observables), np.array(actions), carry
            )

        return first_q_values, second_q_values

    def export_model(self, filename: str = "model", directory: str = "Models"):
        """
        Export the model state to a directory.

        Parameters
        ----------
        filename : str (default=models)
                Name of the file the models are saved in.
        directory : str (default=Models)
                Directory in which to save the models. If the directory is not
                in the currently directory, it will be created.

        """
        model_params = self.model_state.params
        opt_state = self.model_state.opt_state
        opt_step = self.model_state.step
        epoch = self.epoch_count
        carry = self.carry

        os.makedirs(directory, exist_ok=True)

        with open(directory + "/" + filename + ".pkl", "wb") as f:
            pickle.dump((model_params, opt_state, opt_step, epoch, carry), f)

    def restore_model_state(self, filename: str = "model", directory: str = "Models"):
        """
        Restore the model state from a file.

        Parameters
        ----------
        filename : str
                Name of the model state file
        directory : str
                Path to the model state file.

        Returns
        -------
        Updates the model state.
        """

        with open(directory + "/" + filename + ".pkl", "rb") as f:
            model_params, opt_state, opt_step, epoch, carry = pickle.load(f)

        self.model_state = self.model_state.replace(
            params=model_params, opt_state=opt_state, step=opt_step
        )
        self.epoch_count = epoch
        self.carry = carry
        logger.info(f"Model state restored from {directory}/{filename}.pkl")

    def __call__(self, params: FrozenDict, episode_features, actions, carry):
        """
        vmaped version of the model call function.
        Operates on a batch of episodes.

        Parameters
        ----------
        parmas : dict
                Parameters of the model.
        episode_features: np.ndarray (n_steps, observable_dimension)
                Features of the episode. This contains the features of the global state,
                for all time steps in the episode.


        Returns
        -------
        logits : np.ndarray
                Output of the network.
        """
        raise NotImplementedError
