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
`lin_comb_Q_T`, and `project_op` (dense ndarray domain).  Discrete basis
construction, eigensolvers, entropy routines,
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

The `upstream/` directory is a byte-for-byte snapshot of the official QuSpin
repository used for API inventory and tests; it is not imported by the wheel.
