import jax.numpy as np
from jax import random
import logging

from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)
class ExperimentTask(Task):
    
    def __init__(self):
        super().__init__(particle_type=0)


    
    def __call__(self, colloids: list) -> float:
        return 0