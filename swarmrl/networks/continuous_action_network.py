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
from optax._src.base import GradientTransformation

from swarmrl.networks.custom_train_state import CustomTrainState
from swarmrl.exploration_policies.exploration_policy import ExplorationPolicy
from swarmrl.exploration_policies.random_exploration import RandomExploration
from swarmrl.networks.network import Network
from swarmrl.sampling_strategies.continuous_gaussian_distribution import (
    ContinuousGaussianDistribution,
)
from swarmrl.sampling_strategies.sampling_strategy import SamplingStrategy
from swarmrl.actions import MPIAction

logger = logging.getLogger(__name__)


class ContinuousActionModel(Network, ABC):
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

        # self.apply_fn = (
        #     self.model.apply
        # )  # jax.jit(jax.vmap(self.model.apply, in_axes=(None, 0, 0, 0)))
        self.input_shape = input_shape
        self.model_state = None
        self.action_dimension = action_dimension
        self.sequence_length = self.input_shape[1]
        self.carry = None
        # initialize the model state
        init_rng = jax.random.PRNGKey(rng_key)
        self.rng_key_sampling_strategy, params_init_rng, self.dropout_key = (
            jax.random.split(init_rng, num=3)
        )
        self.optimizer = optimizer
        self.model_state = self._create_train_state(params_init_rng)
        self.deployment_mode = deployment_mode
        self.iteration = 0
        if not deployment_mode:
            self.exploration_policy = exploration_policy
            self.epoch_count = 0

    def _create_custom_train_state(self, optimizer: dict):
        """
        Deal with the optimizers in case of complex configuration.
        """
        return type("TrainState", (CustomTrainState,), optimizer)

    def _create_train_state(self, init_rng: int) -> CustomTrainState:
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
        variables = self.model.init(
            init_rng,
            np.ones(list(self.input_shape)),
            np.ones(
                list([self.input_shape[0], self.sequence_length, self.action_dimension])
            ),
            train = False,
        )
        params = variables["params"]
        batch_stats = variables["batch_stats"]
        model_summary = self.model.tabulate(
            jax.random.PRNGKey(1),
            np.ones(list(self.input_shape)),
            np.ones(
                list([self.input_shape[0], self.sequence_length, self.action_dimension])
            ),
        )
        print(model_summary)
        *_, self.carry = self.model.apply(
            {"params": params, "batch_stats": batch_stats},
            np.ones(list(self.input_shape)),
            np.ones(
                list([self.input_shape[0], self.sequence_length, self.action_dimension])
            ),
            self.carry,
            train=False,
        )
        self.carry = np.array([self.carry[0][0], self.carry[1][0]])
        self.carry = np.expand_dims(self.carry, axis=1)
        self.carry = tuple(self.carry)
        if isinstance(self.optimizer, dict):
            raise NotImplementedError
            # CustomTrainState = self._create_custom_train_state(self.optimizer)
            # return CustomTrainState.create(
            #     apply_fn=self.model.apply, params=params, tx=self.optimizer
            # )
        else:
            return CustomTrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.optimizer,
                batch_stats=batch_stats,
            )

    def reinitialize_network(self):
        """
        Initialize the neural network.
        """
        rng_key = onp.random.randint(0, 1027465782564)
        init_rng = jax.random.PRNGKey(rng_key)
        _, subkey = jax.random.split(init_rng)
        self.model_state = self._create_train_state(subkey)

    def set_temperature(self, exp_temperature: float):
        """
        Set the exponential temperature of the model.

        Parameters
        ----------
        temperature : float
                Temperature of the model.
        """
        params = self.model_state.params
        logger.info(f"current value and shape of temperature: {params['temperature'], np.shape(params['temperature'])}")
        params["temperature"] = np.array([np.log(exp_temperature)])  
        self.model_state = self.model_state.replace(
            params=params,
        )
        logger.info(f"new value and shape of temperature: {params['temperature'], np.shape(params['temperature'])}")
        logger.info(f"Temperature set to {exp_temperature}")
        
    def update_model(self, grads, updated_batch_stats=None):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        # Logging for grads and pre-train model state
        logger.debug(f"{grads=}")
        logger.debug(f"{self.model_state=}")
        if isinstance(self.optimizer, dict):
            raise NotImplementedError
        else:
            self.model_state = self.model_state.apply_gradients(grads=grads)
        if updated_batch_stats is None:
            updated_batch_stats = self.model_state.batch_stats
        self.model_state = self.model_state.replace(
            batch_stats=updated_batch_stats
        )  # Update batch stats
        # Logging for post-train model state
        logger.debug(f"{self.model_state=}")
        logger.debug("Model updated")
        self.epoch_count += 1

    def split_rng_key(self):
        self.rng_key_sampling_strategy, _ = jax.random.split(
            self.rng_key_sampling_strategy
        )

    def compute_action_training(
        self,
        params: FrozenDict,
        observables: np.ndarray,
        previous_actions: np.ndarray,
        carry: np.ndarray = None,
    ):
        """
        Compute and action from the action space.

        This method computes an global action which acts on all colloids.

        Parameters
        ----------
        observables : List (n_agents, observable_dimension)
                Observable for each colloid for which the action should be computed.

        Returns
        -------
        tuple : (np.ndarray, np.ndarray)
                The first element is an array of indices corresponding to the action
                taken by the agent. The value is bounded between 0 and the number of
                output neurons. The second element is an array of the corresponding
                log_probs (i.e. the output of the network put through a softmax).
        """
        self.iteration += 1
        sampling_subkey = jax.random.fold_in(
            self.rng_key_sampling_strategy, self.iteration
        )
        dropout_subkey = jax.random.fold_in(self.dropout_key, self.iteration)
        try:
            (logits, _), batch_stats_update = self.model_state.apply_fn(
                {"params": params, "batch_stats": self.model_state.batch_stats},
                np.array(observables),
                np.array(previous_actions),
                carry,
                train = not self.deployment_mode,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_subkey},
            )
        except AttributeError:  # We need this for loaded models.
            (logits, _), batch_stats_update = self.model_state.apply_fn(
                {
                    "params": self.model_state["params"],
                    "batch_stats": self.model_state["batch_stats"],
                },
                np.array(observables),
                np.array(previous_actions),
                carry,
                train = not self.deployment_mode,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_subkey},
            )
        # self.model_state = self.model_state.replace(batch_stats=batch_stats_update["batch_stats"])
        action, log_probs = self.sampling_strategy(
            logits, subkey=sampling_subkey, calculate_log_probs=True
        )
        return action, log_probs, batch_stats_update["batch_stats"]

    def compute_action(self, observables, previous_actions):
        self.iteration += 1
        sampling_subkey = jax.random.fold_in(
            self.rng_key_sampling_strategy, self.iteration
        )
        dropout_subkey = jax.random.fold_in(self.dropout_key, self.iteration)
        try:
            logits, self.carry = self.model_state.apply_fn(
                {
                    "params": self.model_state.params,
                    "batch_stats": self.model_state.batch_stats,
                },
                np.array(observables),
                np.array(previous_actions),
                self.carry,
                train = False,
                rngs={"dropout": dropout_subkey},
            )
        except AttributeError:
            logits, self.carry = self.model_state.apply_fn(
                {
                    "params": self.model_state["params"],
                    "batch_stats": self.model_state["batch_stats"],
                },
                np.array(observables),
                np.array(previous_actions),
                self.carry,
                train = False,
                rngs={"dropout": dropout_subkey},
            )

        self.carry = np.array(
            [
                self.carry[0][np.shape(observables)[0] - 1],
                self.carry[1][np.shape(observables)[0] - 1],
            ]
        )

        self.carry = tuple(np.expand_dims(self.carry, axis=1))
        logits = logits.squeeze()
        
        logger.info(
            f"covariance {np.exp(logits[self.action_dimension:]) * (self.sampling_strategy.action_limits[:,1] - self.sampling_strategy.action_limits[:,0])}"
        )
        logger.info(f"mean {logits[:self.action_dimension]}")
        action, _ = self.sampling_strategy(
            logits[np.newaxis, :], subkey=sampling_subkey, calculate_log_probs=False, deployment_mode=self.deployment_mode
        )
        # action = self.exploration_policy(action)
        return action

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
        batch_stats = self.model_state.batch_stats
        os.makedirs(directory, exist_ok=True)

        with open(directory + "/" + filename + ".pkl", "wb") as f:
            pickle.dump(
                (model_params, opt_state, opt_step, epoch, carry, batch_stats), f
            )

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
            model_params, opt_state, opt_step, epoch, carry, batch_stats = pickle.load(
                f
            )

        self.model_state = self.model_state.replace(
            params=model_params,
            opt_state=opt_state,
            step=opt_step,
            batch_stats=batch_stats,
        )
        logger.info(self.model_state.params.keys())
        self.carry = carry
        self.epoch_count = epoch
        # self.carry = carry
        logger.info(f"Model state restored from {directory}/{filename}.pkl")

    def get_exp_temperature(self) -> float:
        """
        Get the temperature of the model.

        Returns
        -------
        float : temperature of the model.
        """
        return jax.lax.stop_gradient(np.exp(self.model_state.params["temperature"]))

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

    def get_exp_temperature(self):
        """
        Get the temperature of the model.

        Returns
        -------
        float : temperature of the model.
        """
        return np.exp(self.model_state.params["temperature"])
