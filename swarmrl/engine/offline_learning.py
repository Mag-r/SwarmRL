import logging
from swarmrl.engine.engine import Engine
from swarmrl.force_functions.global_force_fn import GlobalForceFunction
import time

logger = logging.getLogger(__name__)


class OfflineLearning(Engine):
    """
    Offline learning engine for SwarmRL.
    It is used to perform offline learning of the policy using a dataset of trajectories.
    """

    def __init__(self):
        super().__init__()
        self.colloids = None

    def seperate_rafts(self):
        pass

    def integrate(self, n_slices: int, force_model: GlobalForceFunction):
        """Do nothing"""
        time.sleep(n_slices)
