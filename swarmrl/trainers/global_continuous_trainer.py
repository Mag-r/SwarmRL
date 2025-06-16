"""
Module to implement a simple multi-layer perceptron for the colloids.
"""

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn, TimeElapsedColumn
from typing import List, Tuple
import logging
import queue
import threading

from swarmrl.engine.engine import Engine
from swarmrl.trainers.trainer import Trainer
from swarmrl.force_functions.global_force_fn import GlobalForceFunction
from swarmrl.agents.MPI_actor_critic import MPIActorCriticAgent
import psutil
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class GlobalContinuousTrainer(Trainer):
    """
    Class for the simple MLP RL implementation.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    """

    def __init__(
        self,
        agents: List[MPIActorCriticAgent],
        lock: threading.Lock = threading.Lock(),
        deployment_mode: bool = False,
    ):
        super().__init__(agents)
        self.learning_thread = threading.Thread(target=self.async_update_rl)
        self.interaction_model_queue = queue.LifoQueue()
        self.lock = lock
        self.sampling_finished = False
        self.deployment_mode = deployment_mode

    def initialize_training(self) -> GlobalForceFunction:
        return GlobalForceFunction(
            agents=self.agents,
        )

    def async_update_rl(self):
        killed = False
        while not killed and not self.sampling_finished:
            force_fn, current_reward, killed = self.update_rl()
            self.interaction_model_queue.put((force_fn, current_reward, killed))

    def update_rl(self) -> Tuple[GlobalForceFunction, np.ndarray]:
        """
        Update the RL algorithm.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        reward : np.ndarray
                Current mean episode reward. This is returned for nice progress bars.
        killed : bool
                Whether or not the task has ended the training.
        """
        reward = 0.0
        switches = []

        for agent in self.agents.values():
            if isinstance(agent, MPIActorCriticAgent):
                ag_reward, ag_killed = agent.update_agent()
                # logger.info(f"reward: {ag_reward}, sum: {np.sum(ag_reward)}")
                reward += np.mean(ag_reward[-10:])
                switches.append(ag_killed)
            else:
                raise NotImplementedError("Only MPIActorCriticAgent is supported.")

        # Create a new interaction model.
        interaction_model = GlobalForceFunction(agents=self.agents.copy())
        logger.debug("RL updated.")
        return interaction_model, np.array(reward), any(switches)

    def perform_rl_training(
        self,
        system_runner: Engine,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
    ):
        """
        Perform the RL training.

        Parameters
        ----------
        system_runner : Engine
                Engine used to perform steps for each agent.
        n_episodes : int
                Number of episodes to use in the training.
        episode_length : int
                Number of time steps in one episode.
        load_bar : bool (default=True)
                If true, show a progress bar.
        """
        self.engine = system_runner
        rewards = [0.0]
        current_reward = 0.0
        linear_regression = 0.0
        # mp.set_start_method('spawn', force=True)

        force_fn = self.initialize_training()
        self.engine.seperate_rafts()
        # Initialize the tasks and observables.
        for agent in self.agents.values():
            agent.reset_agent(self.engine.colloids)

        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Episode reward: {task.fields[current_reward]} Running Reward:"
            " {task.fields[running_reward]} Total reward: {task.fields[total_reward]} Linear Regression: {task.fields[linear_regression]}",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                "RL Training",
                total=n_episodes,
                Episode=0,
                current_reward=current_reward,
                running_reward=np.mean(rewards),
                total_reward=np.mean(rewards),
                linear_regression=linear_regression,
                visible=load_bar,
            )
            try:
                for episode in range(n_episodes):
                    self.engine.integrate(episode_length, force_fn)
                    with self.lock:
                        for agent in self.agents.values():
                            agent.remove_old_data()
                    if episode == 0 and not self.deployment_mode:
                        self.learning_thread.start()
                    if not self.interaction_model_queue.empty():
                        force_fn, current_reward, killed = (
                            self.interaction_model_queue.get()
                        )
                        with self.lock:  # clear the queue, to not use old models
                            while not self.interaction_model_queue.empty():
                                self.interaction_model_queue.get()
                        logger.info(f"obtained new interaction model")

                        if killed:
                            print(
                                "Reinitializing the engine, as the task has ended the training."
                            )
                            self.engine.seperate_rafts()
                            for agent in self.agents.values():
                                agent.initialize_network()

                        rewards.append(current_reward)
                        logger.info(
                            f"Episode {episode}; mean immediate reward: {current_reward}"
                        )
                        # Perform linear regression on rewards using scipy
                        if len(rewards) > 1:
                            x = np.arange(len(rewards))
                            y = np.array(rewards)
                            slope, _, _, _, _ = linregress(x, y)
                            linear_regression = np.round(slope, 4)

                        progress.update(task, linear_regression=linear_regression)
                        progress.update(
                            task,
                            advance=1,
                            Episode=episode,
                            current_reward=np.round(current_reward, 2),
                            running_reward=np.round(np.mean(rewards[-10:]), 2),
                            total_reward=np.round(np.mean(rewards), 2),
                            linear_regression=linear_regression,
                        )
                        
                    else:
                        logger.info("Sampling is faster than learning.")

                        if not self.learning_thread.is_alive() and not self.deployment_mode:
                            logger.warning("Learning thread has stopped unexpectedly. Try restarting the training.")
                            self.learning_thread = threading.Thread(target=self.async_update_rl)
                            self.learning_thread.start()


                    if episode % 10 == 0 and episode > 0:
                            # Save the agents every 10 episodes.
                            logger.info(
                                "Trying to seperate the rafts and save the agents."
                            )
                            # self.engine.seperate_rafts()
                            
                            with self.lock:
                                for agent in self.agents.values():
                                    agent.actor_network.exploration_policy.reduce_randomness()
                                    agent.save_trajectory(
                                        identifier=f"{agent.task.__class__.__name__}_episode_{int(episode/10.0)}"
                                    )
                                    agent.save_agent(
                                        identifier=agent.task.__class__.__name__
                                    )
            finally:
                self.engine.finalize()
                self.sampling_finished = True
                self.learning_thread.join(100.0)

        return np.array(rewards)
