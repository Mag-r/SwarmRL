"""
Package for value functions
"""

from swarmrl.value_functions.expected_returns import ExpectedReturns
from swarmrl.value_functions.generalized_advantage_estimate import GAE
from swarmrl.value_functions.global_expected_returns import GlobalExpectedReturns
from swarmrl.value_functions.td_return import TDReturns

__all__ = [ExpectedReturns.__name__, GAE.__name__, GlobalExpectedReturns.__name__, TDReturns.__name__]
