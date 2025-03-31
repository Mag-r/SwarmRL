"""
__init__ for the tasks module.
"""

from swarmrl.tasks import object_movement, searching
from swarmrl.tasks.multi_tasking import MultiTasking
from swarmrl.tasks.task import Task
from swarmrl.tasks.dummy_task import DummyTask
from swarmrl.tasks.MPI_chain import ChainTask
from swarmrl.tasks.experiment_chain import ExperimentChainTask
from swarmrl.tasks.experiment_hexagon import ExperimentHexagonTask

__all__ = [
    searching.__name__,
    object_movement.__name__,
    Task.__name__,
    MultiTasking.__name__,
    DummyTask.__name__,
    ChainTask.__name__,
    ExperimentChainTask.__name__,
    ExperimentHexagonTask.__name__,
]
