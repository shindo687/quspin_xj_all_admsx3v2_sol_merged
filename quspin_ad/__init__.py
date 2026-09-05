"""Automatic-differentiation rules for selected QuSpin callables.

This package is a sidecar: QuSpin remains the source of every primal value,
while this module only supplies explicit ChainRules JVP/VJP rules.  Importing
``quspin_ad`` registers rules for the sidecar callables exported here.  When a
QuSpin installation is available, the corresponding upstream callables are
also registered (``register_upstream_rules`` is idempotent).
"""

from .rules import (
    ED_state_vs_time,
    KL_div,
    anti_commutator,
    coherent_state,
    commutator,
    lin_comb_Q_T,
    project_op,
    register_upstream_rules,
)
from .rules import ad as _ad
from .dynamic import (
    dynamic_trajectory,
    fixed_grid_trajectory,
    evolve_fixed_grid,
    fixed_grid_evolve,
    trajectory_jvp,
    trajectory_vjp,
)

ZERO = _ad.ZERO
grad = _ad.grad
jvp = _ad.jvp
value_and_grad = _ad.value_and_grad
vjp = _ad.vjp

__all__ = [
    "ZERO",
    "ED_state_vs_time",
    "KL_div",
    "anti_commutator",
    "coherent_state",
    "commutator",
    "grad",
    "jvp",
    "lin_comb_Q_T",
    "project_op",
    "register_upstream_rules",
    "value_and_grad",
    "vjp",
    "dynamic_trajectory",
    "fixed_grid_trajectory",
    "evolve_fixed_grid",
    "fixed_grid_evolve",
    "trajectory_jvp",
    "trajectory_vjp",
]
