# quspin-ad

`quspin-ad` is a separately installable sidecar for QuSpin 1.0.1.  It registers
analytic first-order rules with the small `chainrules` protocol (or its bundled
compatible fallback when ChainRules is unavailable) while leaving
the upstream QuSpin callable and source tree unchanged.

Install it in a clean environment with:

```bash
python -m pip install .
```

Load the rules explicitly (registration never monkey-patches QuSpin):

```python
import quspin_ad  # registers the rules
import chainrules as ad
from quspin.tools.misc import KL_div

value, tangent = ad.jvp(
    KL_div, p1, p2,
    tangents={"p1": dp1, "p2": ad.ZERO},
)
value, pullback = ad.vjp(KL_div, p1, p2, wrt=("p1", "p2"))
gradients = pullback(1.0)
```

Supported rules and their mathematical domains are specified in [SPEC.md](SPEC.md).
The package currently covers the continuous, array-valued APIs `KL_div`,
`coherent_state`, `commutator`, `anti_commutator`, `ED_state_vs_time`,
`lin_comb_Q_T`, and `project_op` (dense ndarray domain), plus a branch-aware
`floquet_eigensystem` adapter for an already-built one-period propagator.
Floquet JVP/VJP supports `UF`, `T`, and fixed-grid `drive_phase`, `gauge`, and
`momentum` controls; exact control derivative matrices can be supplied for a
driven lattice.  Discrete basis construction, adaptive evolution,
eigensolvers,
I/O, sparse/operator object methods, and non-array workflows are explicitly
reported as deferred or not suitable for AD rather than approximated by finite
differences.


For dynamic control, `dynamic_trajectory(H, psi0, times, params=...)` evaluates
the Schrödinger equation on a caller-provided nondecreasing grid and integrates
analytic state sensitivities for the named callback coefficients. QuSpin
Hamiltonian callbacks must expose `derivative` (one coefficient) or
`derivatives` (a mapping for the differentiated coefficients); plain matrix
callables accept the equivalent `derivatives` mapping. `chainrules.jvp` and
`chainrules.vjp` support `psi0`, `params`, and `controls`, with
`checkpoint_interval` providing bounded checkpoint workspace; the VJP closure
still retains the returned trajectory for general objective pullbacks. An optional
scalar `objective` must provide an analytic `jvp`/`derivative` contract (or
already be a registered ChainRules callable). Missing metadata, non-monotone
grids, discontinuous schedules, mixed states, and iterator or adaptive paths
fail explicitly; no finite-difference fallback is used. Real-valued controls return real
cotangents, while complex-valued controls preserve their complex real-linear
cotangents.

Second-order composition is available through the bundled ChainRules
protocol:

```python
import chainrules as ad

# Compact form: the final positional argument is the direction.
value, gradient, hvp = ad.value_grad_and_hvp(loss, parameters, direction)
# Explicit form for several active inputs:
value, gradient, hvp = ad.value_grad_and_hvp(
    loss, x, y, wrt=("x", "y"), vector={"x": dx, "y": dy}
)

# Forward-over-forward composition, for array-valued primitives as well.
value, tangent, second_tangent = ad.nested_jvp(
    quspin_ad.ED_state_vs_time, psi, energies, eigenvectors, times,
    tangents={"E": dE},
)
```

`value_grad_and_hvp` requires a real scalar loss and returns mappings keyed by
`wrt`; `hvp` returns only the product.  `nested_jvp` (also available as
`jvp2`) returns the value, first directional derivative, and second directional
derivative.  The implementation uses analytic real-linear local rules and a
small NumPy jet tracer for composition.  It does not evaluate perturbed
primal points or use finite differences.  Fixed-shape and boundary restrictions
of the first-order rules remain in force (`iterate=True`, `out=...`, `a=0`,
and unsupported active inputs continue to fail explicitly).


The `upstream/` directory is a byte-for-byte snapshot of the official QuSpin
repository used for API inventory and tests; it is not imported by the wheel.
