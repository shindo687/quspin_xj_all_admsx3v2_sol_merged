"""Analytic sensitivities for fixed-grid Schrödinger trajectories.

The state and parameter tangent equations are integrated together on the
user-supplied output grid.  SciPy's high-accuracy DOP853 integrator is used
when available (matching QuSpin's own forward path); a deterministic fixed
substep RK4 implementation keeps sidecar-only imports usable without SciPy.
Drive derivatives must be supplied through ``derivatives`` (matrix callbacks)
or callback ``derivative`` metadata (QuSpin Hamiltonians).
"""
from __future__ import annotations

from collections.abc import Mapping
import inspect
import numpy as np
# SciPy is a QuSpin runtime dependency, but importing the sidecar should still
# work in an environment where QuSpin has not been installed yet (for example
# while inspecting the API inventory).  Keep the import lazy and provide a
# small deterministic RK4 integrator as a diagnostic fallback.  The fallback
# is not a finite-difference derivative: sensitivities are integrated as part
# of the same augmented ODE.

try:
    import chainrules as ad
except ModuleNotFoundError:  # pragma: no cover
    from . import _chainrules as ad


def _as_grid(times):
    raw = np.asarray(times)
    if np.iscomplexobj(raw):
        raise ValueError("times must be real-valued")
    t = np.asarray(raw, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a finite, one-dimensional fixed grid")
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be nondecreasing")
    return t


def _params(params):
    if params is None:
        return {}, ()
    if isinstance(params, Mapping):
        return dict(params), tuple(params.values())
    a = np.asarray(params)
    if a.ndim == 0:
        return {"param": a.item()}, (a.item(),)
    return {str(i): x for i, x in enumerate(a.flat)}, tuple(a.flat)


def _call_matrix(fn, t, p, positional):
    if p is None:
        try:
            return fn(float(t))
        except TypeError:
            return fn(float(t), *positional)
    try:
        return fn(float(t), p)
    except (TypeError, AttributeError, KeyError, IndexError):
        try:
            return fn(float(t), *positional)
        except (TypeError, AttributeError, KeyError, IndexError):
            # A positional parameter array is another common generic-callback
            # convention (``fn(t, params)[0]``).  The first attempt uses the
            # named mapping so dictionary-style callbacks remain natural.
            values = np.asarray(positional)
            return fn(float(t), values[0] if values.size == 1 else values)


def _call_derivative(fn, t, pmap, positional):
    """Call a matrix-derivative callback across common parameter styles."""
    try:
        return fn(float(t), pmap)
    except (TypeError, AttributeError, KeyError, IndexError):
        try:
            return fn(float(t), *positional)
        except (TypeError, AttributeError, KeyError, IndexError):
            try:
                values = np.asarray(positional)
                return fn(float(t), values[0] if values.size == 1 else values)
            except (TypeError, AttributeError, KeyError, IndexError):
                return fn(float(t))


def _quspin_entries(H):
    dynamic = getattr(H, "dynamic", None)
    if not isinstance(dynamic, Mapping) or not dynamic:
        return None
    entries = []
    for callback, operator in dynamic.items():
        fn = getattr(callback, "_f", callback)
        args = tuple(getattr(callback, "_args", ()))
        try:
            names = tuple(inspect.signature(fn).parameters)[1:]
        except (TypeError, ValueError) as exc:
            raise TypeError("dynamic callback must expose a Python signature") from exc
        if len(names) != len(args):
            raise TypeError("dynamic callback arguments do not match its signature")
        mat = operator.toarray() if hasattr(operator, "toarray") else operator
        entries.append((callback, fn, names, args, np.asarray(mat)))
    return entries


def _controls_for(entries, params, controls):
    supplied = params if params is not None else controls
    pmap, ppos = _params(supplied)
    if entries is None:
        return pmap, ppos
    # Positional control arrays are convenient for generic matrix callbacks,
    # while QuSpin callbacks have named parameters.  Map an array/tuple onto
    # callback names in first-seen order instead of reporting the synthetic
    # ``"0"``/``"1"`` keys as unknown controls.
    if supplied is not None and not isinstance(supplied, Mapping):
        callback_names = []
        for _, _, names, _, _ in entries:
            for name in names:
                if name not in callback_names:
                    callback_names.append(name)
        flat = tuple(np.asarray(supplied).flat)
        if len(flat) != len(callback_names):
            raise ValueError(
                "positional controls must contain one value per callback parameter"
            )
        pmap = dict(zip(callback_names, flat))
    values = {}
    for _, _, names, args, _ in entries:
        for name, value in zip(names, args):
            value = pmap.get(name, value)
            if np.asarray(value).ndim != 0:
                raise TypeError(f"dynamic control {name!r} must be scalar")
            if name in values and not np.allclose(values[name], value):
                raise ValueError(f"control {name!r} has inconsistent values")
            values[name] = value
    unknown = set(pmap) - set(values)
    if unknown:
        raise TypeError(f"unknown dynamic controls: {sorted(unknown)!r}")
    return values, tuple(values[name] for name in values)


def _supplied_controls(params, controls, pmap):
    """Return the parameter names represented by the caller's control value.

    QuSpin callbacks carry default positional arguments internally.  Those
    defaults are needed to evaluate the primal Hamiltonian, but a mapping such
    as ``params={"amplitude": ...}`` only asks AD to expose that one control in
    its nested cotangent.  Keeping this distinction avoids returning gradients
    for values the caller did not supply.
    """
    supplied = params if params is not None else controls
    if isinstance(supplied, Mapping):
        return tuple(supplied)
    return tuple(pmap)


def _is_zero_tangent(value):
    """Recognize nested/array zero tangents without invoking callbacks."""
    if value is ad.ZERO or value is None:
        return True
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return array.size == 0 or bool(np.all(array == 0))


def _parameter_cotangent(value, coefficient):
    """Convert a real-linear coefficient to the primal parameter dtype."""
    if np.iscomplexobj(np.asarray(value)):
        # Re(conj(g) dz) is represented by g = conj(coefficient), where
        # ``coefficient`` is the complex-linear pairing vdot(cotangent, JVP).
        return np.conj(coefficient)
    return float(np.real(coefficient))


def _callback_derivatives(callback, fn, names, t, args, required_names=None):
    """Evaluate callback derivative metadata for the requested coefficients.

    A callback may expose derivatives for several coefficients while a JVP
    activates only a subset.  Requiring metadata for inactive coefficients
    would make an otherwise valid partial JVP fail, so ``required_names`` is
    deliberately separate from the callback's complete signature.
    """
    required = tuple(names if required_names is None else required_names)

    def call_contract(contract):
        if not callable(contract):
            return contract
        try:
            return contract(float(t), *args)
        except (TypeError, AttributeError, KeyError, IndexError):
            try:
                return contract(float(t), dict(zip(names, args)))
            except (TypeError, AttributeError, KeyError, IndexError):
                try:
                    values = np.asarray(args)
                    return contract(
                        float(t), values[0] if values.size == 1 else values
                    )
                except (TypeError, AttributeError, KeyError, IndexError):
                    return contract(float(t))

    contract = getattr(callback, "derivatives", None)
    if contract is None:
        contract = getattr(fn, "derivatives", None)
    contract = call_contract(contract)
    single = getattr(callback, "derivative", None)
    if single is None:
        single = getattr(fn, "derivative", None)
    if contract is None and single is None:
        raise ad.NonDifferentiablePoint(
            f"dynamic callback {getattr(fn, '__name__', fn)!r} has no derivative contract"
        )
    if contract is None:
        contract = call_contract(single)
    if isinstance(contract, Mapping):
        missing = set(required) - set(contract)
        if missing:
            raise ad.NonDifferentiablePoint(
                "dynamic callback derivative contract is missing metadata for "
                f"parameter(s) {sorted(missing)!r}"
            )
        extra = set(contract) - set(names)
        if extra:
            raise TypeError(
                "dynamic callback derivative contract contains unknown parameter(s) "
                f"{sorted(extra)!r}"
            )
        values = {}
        for name in required:
            value = contract[name]
            if callable(value):
                value = call_contract(value)
            values[name] = value
        return values
    if len(names) == 1 and len(required) == 1 and np.asarray(contract).ndim == 0:
        return {required[0]: contract}
    raise TypeError("derivative contract must map every callback parameter")


def _ham_and_derivs(H, t, pmap, ppos, derivatives, n, require_derivatives=False,
                    entries=None, sensitivity_names=None):
    if entries is None:
        entries = _quspin_entries(H)
    if entries is not None:
        # ``hamiltonian.dynamic`` contains only driven terms; include the
        # static matrix as QuSpin's RHS does before evaluating callbacks.
        static = getattr(H, "static", None)
        if static is None:
            static = getattr(H, "_static", None)
        if static is None:
            Ht = np.zeros((n, n), dtype=np.complex128)
        else:
            Ht = static.toarray() if hasattr(static, "toarray") else np.asarray(static)
            if Ht.shape != (n, n):
                raise ValueError("hamiltonian/state dimensions are incompatible")
            Ht = np.asarray(Ht, dtype=np.result_type(Ht, np.complex128)).copy()
        active = set(pmap if sensitivity_names is None else sensitivity_names)
        dm = {name: np.zeros((n, n), dtype=np.complex128) for name in active}
        external = derivatives
        if callable(external):
            external = _call_derivative(external, t, pmap, ppos)
        if external is not None and not isinstance(external, Mapping):
            raise TypeError(
                "QuSpin dynamic derivatives must map callback parameter names to scalars or matrices"
            )
        for cb, fn, names, args, mat in entries:
            aa = tuple(pmap.get(name, value) for name, value in zip(names, args))
            Ht = Ht + mat * fn(float(t), *aa)
            callback_active = tuple(name for name in names if name in active)
            if callback_active and (derivatives is not None or require_derivatives):
                if external is not None:
                    vals = {}
                    for name in callback_active:
                        if name not in external:
                            raise ad.NonDifferentiablePoint(
                                f"missing derivative metadata for dynamic control {name!r}"
                            )
                        raw = external[name]
                        if callable(raw):
                            raw = _call_derivative(
                                raw, t, dict(zip(names, aa)), aa
                            )
                        arr = np.asarray(raw)
                        if arr.shape == (n, n):
                            dm[name] = dm[name] + arr
                            continue
                        if arr.ndim != 0:
                            raise TypeError(
                                f"derivative for dynamic control {name!r} must be scalar or matrix"
                            )
                        vals[name] = arr.item()
                else:
                    vals = _callback_derivatives(
                        cb, fn, names, t, aa, required_names=callback_active
                    )
                for name, scalar in vals.items():
                    if name in dm:
                        dm[name] = dm[name] + mat * scalar
        return Ht, dm
    if callable(H):
        raw = _call_matrix(H, t, pmap if pmap else None, ppos)
    elif hasattr(H, "toarray"):
        try:
            raw = H.toarray(time=float(t))
        except TypeError:
            raw = H.toarray(float(t))
    else:
        raw = H
    Ht = np.asarray(raw)
    if Ht.shape != (n, n):
        raise ValueError("hamiltonian/state dimensions are incompatible")
    dm = {}
    if derivatives is not None:
        rawd = _call_derivative(derivatives, t, pmap, ppos) if callable(derivatives) else derivatives
        if isinstance(rawd, Mapping):
            active_names = tuple(pmap) if sensitivity_names is None else tuple(sensitivity_names)
            for name, val in rawd.items():
                if sensitivity_names is not None and name not in sensitivity_names:
                    continue
                val = _call_derivative(val, t, pmap, ppos) if callable(val) else val
                arr = np.asarray(val)
                if arr.shape != (n, n):
                    raise TypeError("drive derivative matrices must match Hamiltonian shape")
                dm[name] = arr
            missing = set(active_names) - set(dm)
            if missing:
                raise ad.NonDifferentiablePoint(
                    "dynamic_trajectory requires derivative metadata for drive "
                    f"parameter(s) {sorted(missing)!r}"
                )
        else:
            arr = np.asarray(rawd)
            if arr.ndim == 2:
                arr = arr[None, ...]
            if arr.ndim != 3 or arr.shape[1:] != (n, n):
                raise TypeError("drive derivatives must have shape (nparam,n,n)")
            names = tuple(pmap) or tuple(str(i) for i in range(arr.shape[0]))
            if sensitivity_names is not None:
                names = tuple(name for name in names if name in sensitivity_names)
                # ``arr`` is ordered according to the full parameter list;
                # select the corresponding rows before zipping below.
                full_names = tuple(pmap) or tuple(str(i) for i in range(arr.shape[0]))
                arr = arr[[full_names.index(name) for name in names]] if names else arr[:0]
            if len(names) != arr.shape[0]:
                raise ValueError("derivative count does not match parameter count")
            dm = dict(zip(names, arr))
    elif pmap:
        raise ad.NonDifferentiablePoint(
            "dynamic_trajectory requires derivative metadata for drive parameters"
        )
    return Ht, dm


def _parse_call(args, t0, times):
    if times is None:
        if len(args) == 1:
            times = args[0]
        elif len(args) == 2:
            t0, times = args
        else:
            raise TypeError("expected times or (t0, times)")
    elif args:
        raise TypeError("times supplied more than once")
    start = float(0.0 if t0 is None else t0)
    if not np.isfinite(start):
        raise ValueError("t0 must be finite and real-valued")
    return start, _as_grid(times)


def _forward(H, psi0, t, params=None, controls=None, derivatives=None, t0=0.0,
             require_derivatives=False, sensitivity_names=None):
    p = np.asarray(psi0)
    if p.ndim != 1:
        raise TypeError("psi0 must be a one-dimensional state")
    entries = _quspin_entries(H)
    pmap, ppos = _controls_for(entries, params, controls)
    names = tuple(pmap) if sensitivity_names is None else tuple(sensitivity_names)
    unknown_sensitivity = set(names) - set(pmap)
    if unknown_sensitivity:
        raise TypeError(f"unknown dynamic controls: {sorted(unknown_sensitivity)!r}")
    count = 1 + len(names)
    y0 = np.zeros((count, p.size), dtype=np.result_type(p, np.complex128))
    y0[0] = p
    def rhs(x, y):
        yy = y.reshape(count, p.size)
        hm, dm = _ham_and_derivs(
            H, x, pmap, ppos, derivatives, p.size, require_derivatives,
            entries=entries, sensitivity_names=names,
        )
        out = np.empty_like(yy)
        out[0] = -1j * hm @ yy[0]
        for j, name in enumerate(names, 1):
            out[j] = -1j * (hm @ yy[j] + dm.get(name, 0) @ yy[0])
        return out.ravel()
    if np.any(t < float(t0)):
        raise ValueError("times must be greater than or equal to t0")
    if t.size == 1 and float(t[0]) == float(t0):
        sol = y0[:, :, None]
    else:
        span = (float(t0), float(t[-1]))
        if span[0] == span[1]:
            sol = np.repeat(y0[:, :, None], t.size, axis=2)
        else:
            try:
                from scipy.integrate import solve_ivp
            except ImportError:
                # This branch is mainly useful for a sidecar-only install.
                # Use a fixed substep RK4 scheme so the result remains
                # reproducible and no adaptive solver state is differentiated.
                sol = _rk4_grid(rhs, y0.ravel(), t0, t, count, p.size)
            else:
                # QuSpin itself uses DOP853 for hamiltonian.evolve.  We use
                # the same high-accuracy forward integrator and integrate the
                # tangent equations in the augmented state in one call.
                # scipy requires strictly increasing ``t_eval``.  A fixed grid
                # may legitimately contain repeated samples, so integrate unique
                # times and restore the requested indexing afterwards.
                unique_t, inverse = np.unique(t, return_inverse=True)
                result = solve_ivp(rhs, span, y0.ravel(), t_eval=unique_t,
                                   method="DOP853", rtol=2e-11, atol=2e-13)
                if not result.success:
                    raise RuntimeError(f"fixed-grid trajectory integration failed: {result.message}")
                unique_sol = result.y.reshape(count, p.size, unique_t.size)
                sol = unique_sol[:, :, inverse]
    states = sol[0]
    sensitivities = {name: sol[j] for j, name in enumerate(names, 1)}
    return states, sensitivities, pmap


def _rk4_grid(rhs, y0, t0, times, count, n):
    """Integrate an augmented trajectory on a fixed output grid.

    A bounded number of uniform substeps per output interval is used.  This is
    deliberately a forward-only fallback for installations without SciPy;
    the analytic tangent equations are part of ``rhs`` and no finite
    difference is taken.
    """
    y = np.asarray(y0).copy()
    out = np.empty((count * n, times.size), dtype=y.dtype)
    for i, target in enumerate(times):
        start = float(t0 if i == 0 else times[i - 1])
        delta = float(target - start)
        if delta == 0.0:
            out[:, i] = y
            continue
        # Keep the largest RK step small enough for smooth drive callbacks.
        steps = max(1, int(np.ceil(abs(delta) / 2.5e-3)))
        h = delta / steps
        x = start
        for _ in range(steps):
            k1 = rhs(x, y)
            k2 = rhs(x + h / 2.0, y + h * k1 / 2.0)
            k3 = rhs(x + h / 2.0, y + h * k2 / 2.0)
            k4 = rhs(x + h, y + h * k3)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            x += h
        out[:, i] = y
    return out.reshape(count, n, times.size)


def _checkpointed_reverse(H, psi0, t, params, controls, derivatives, g,
                          t0, checkpoint_interval, need_drive,
                          sensitivity_names=None):
    """Reverse a fixed-grid trajectory while retaining only checkpoints.

    The forward state is first evaluated at every ``checkpoint_interval``-th
    output.  During the reverse sweep each checkpoint pair is recomputed with
    dense output, and a continuous adjoint is integrated backwards over that
    segment.  Output cotangents are injected at their grid times.  Thus the
    sidecar retains ``O(Ns * n_checkpoint)`` trajectory state, rather than an
    augmented state for every control and every time sample.
    """
    p = np.asarray(psi0)
    if g.shape != (p.size, t.size):
        raise ValueError("trajectory cotangent must match output shape")
    entries = _quspin_entries(H)
    pmap, ppos = _controls_for(entries, params, controls)
    names = tuple(pmap) if sensitivity_names is None and need_drive else (
        tuple(sensitivity_names) if need_drive else ()
    )
    # Boundaries are output indices; t0 is kept separately as the beginning of
    # the first segment so grids beginning after t0 are handled correctly.
    boundaries = list(range(0, t.size, checkpoint_interval))
    if boundaries[-1] != t.size - 1:
        boundaries.append(t.size - 1)
    checkpoint_times = t[np.asarray(boundaries, dtype=int)]
    checkpoint_states, _, _ = _forward(
        H, p, checkpoint_times, params, controls, derivatives, t0,
        require_derivatives=False, sensitivity_names=(),
    )
    n = p.size
    lam = np.zeros(n, dtype=np.result_type(p, g, np.complex128))
    # Keep the complex coefficient until the final dtype projection.  A
    # complex control needs the conjugate coefficient under the real-linear
    # cotangent convention; reducing to ``real`` here loses its imaginary
    # component and breaks JVP/VJP duality.
    q = np.zeros(len(names), dtype=np.complex128)

    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - QuSpin requires SciPy
        raise RuntimeError("checkpointed reverse mode requires scipy") from exc

    def integrate_state(start, a, b):
        if float(a) == float(b):
            class _Constant:
                def sol(self, x):
                    return np.asarray(start)
            return _Constant()

        def state_rhs(x, y):
            hm, _ = _ham_and_derivs(
                H, x, pmap, ppos, derivatives, n, False, entries=entries,
                sensitivity_names=(),
            )
            return -1j * hm @ y

        result = solve_ivp(
            state_rhs, (float(a), float(b)), np.asarray(start),
            method="DOP853", rtol=2e-11, atol=2e-13, dense_output=True,
        )
        if not result.success:
            raise RuntimeError(f"checkpoint recomputation failed: {result.message}")
        return result.sol

    for seg in range(len(boundaries) - 1, -1, -1):
        end_idx = boundaries[seg]
        if seg == 0:
            start_idx = None
            start_time = float(t0)
            start_state = p
        else:
            start_idx = boundaries[seg - 1]
            start_time = float(t[start_idx])
            start_state = checkpoint_states[:, seg - 1]
        end_time = float(t[end_idx])
        state_sol = integrate_state(start_state, start_time, end_time)
        first_idx = 0 if start_idx is None else start_idx + 1
        for k in range(end_idx, first_idx - 1, -1):
            lam = lam + np.asarray(g[:, k])
            left = start_time if k == 0 else float(t[k - 1])
            right = float(t[k])
            if right == left:
                continue

            def reverse_rhs(x, z):
                lv = z[:n]
                hm, dm = _ham_and_derivs(
                    H, x, pmap, ppos, derivatives, n, need_drive,
                    entries=entries, sensitivity_names=names,
                )
                state = np.asarray(state_sol(x))
                out = np.empty(n + len(names), dtype=np.result_type(z, np.complex128))
                # A=-1j*H, so the backward-time adjoint equation is
                # lambda'=-A^H lambda=-1j*H^H lambda.
                out[:n] = -1j * hm.conj().T @ lv
                for j, name in enumerate(names):
                    # Integrating q' backwards (with a minus sign) yields
                    # the complex-linear pairing.  The final projection
                    # conjugates it for a complex primal control and takes
                    # its real part for a real control.
                    out[n + j] = -np.vdot(lv, -1j * dm.get(name, 0) @ state)
                return out

            z0 = np.concatenate((lam, q.astype(np.complex128)))
            result = solve_ivp(
                reverse_rhs, (right, left), z0,
                method="DOP853", rtol=2e-11, atol=2e-13,
            )
            if not result.success:
                raise RuntimeError(f"checkpoint reverse integration failed: {result.message}")
            lam = result.y[:n, -1]
            if names:
                q = result.y[n:, -1]
    grad_by_name = {
        name: _parameter_cotangent(pmap[name], q[j])
        for j, name in enumerate(names)
    }
    return lam, grad_by_name, pmap


def dynamic_trajectory(hamiltonian, psi0, *args, params=None,
                       derivatives=None, checkpoint_interval=None,
                       objective=None, t0=None, times=None, controls=None):
    """Return a fixed-grid trajectory, optionally evaluated by ``objective``."""
    _validate_checkpoint(checkpoint_interval)
    t0, t = _parse_call(args, t0, times)
    states, *_ = _forward(
        hamiltonian, psi0, t, params, controls, derivatives, t0,
        sensitivity_names=(),
    )
    return objective(states) if objective is not None else states


def _validate_checkpoint(checkpoint_interval):
    if checkpoint_interval is not None and (
        not isinstance(checkpoint_interval, (int, np.integer)) or checkpoint_interval < 1
    ):
        raise ValueError("checkpoint_interval must be a positive integer")


fixed_grid_trajectory = dynamic_trajectory
evolve_fixed_grid = dynamic_trajectory
fixed_grid_evolve = dynamic_trajectory


def _objective_parameter_name(objective):
    """Return the sole state parameter name for an AD-registered objective."""
    try:
        signature = inspect.signature(objective)
        positional = [
            p.name for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        positional = []
    if len(positional) != 1:
        raise ad.NonDifferentiablePoint(
            "trajectory objective must expose a one-argument derivative contract"
        )
    return positional[0]


def _objective_jvp(objective, states, dstates):
    """Evaluate a scalar objective and its tangent without finite differences.

    Objectives may provide a lightweight ``jvp(states, dstates)`` method or a
    ``derivative(states)`` cotangent method.  An objective that is itself a
    registered ChainRules callable is also accepted.  Arbitrary Python
    callables are intentionally rejected rather than approximated numerically.
    """
    value = objective(states)
    if hasattr(objective, "jvp") and callable(objective.jvp):
        result = objective.jvp(states, dstates)
        tangent = result[1] if isinstance(result, tuple) and len(result) == 2 else result
    elif hasattr(objective, "derivative") and callable(objective.derivative):
        cotangent = np.asarray(objective.derivative(states))
        if cotangent.shape != np.asarray(states).shape:
            raise ValueError("trajectory objective derivative must match trajectory shape")
        tangent = np.vdot(cotangent, dstates)
    else:
        name = _objective_parameter_name(objective)
        try:
            _, tangent = ad.jvp(objective, states, tangents={name: dstates})
        except Exception as exc:
            if isinstance(exc, ad.NonDifferentiablePoint):
                raise
            raise ad.NonDifferentiablePoint(
                "trajectory objective has no analytic JVP contract"
            ) from exc
    value_arr = np.asarray(value)
    if value_arr.ndim != 0:
        raise TypeError("trajectory objective must return a scalar")
    if not np.iscomplexobj(value_arr):
        tangent = float(np.real(tangent))
    return value, tangent


def _objective_cotangent(objective, states, cotangent):
    """Pull a scalar objective cotangent back to the trajectory."""
    if hasattr(objective, "vjp") and callable(objective.vjp):
        raw = objective.vjp(states)
        if isinstance(raw, tuple) and len(raw) == 2 and callable(raw[1]):
            pulled = raw[1](cotangent)
        else:
            pulled = raw
    elif hasattr(objective, "derivative") and callable(objective.derivative):
        pulled = np.asarray(objective.derivative(states)) * np.asarray(cotangent)
    else:
        name = _objective_parameter_name(objective)
        try:
            _, pullback = ad.vjp(objective, states, wrt=(name,))
            pulled = pullback(cotangent)[name]
        except Exception as exc:
            if isinstance(exc, ad.NonDifferentiablePoint):
                raise
            raise ad.NonDifferentiablePoint(
                "trajectory objective has no analytic VJP contract"
            ) from exc
    if isinstance(pulled, Mapping):
        if "states" in pulled:
            pulled = pulled["states"]
        elif len(pulled) == 1:
            pulled = next(iter(pulled.values()))
    gradient = np.asarray(pulled)
    if gradient.shape != np.asarray(states).shape:
        raise ValueError("trajectory objective cotangent must match trajectory shape")
    return gradient


def _tangent(H, psi0, t, params, controls, derivatives, tangents, t0=0.0):
    dparams = tangents.get("params", tangents.get("controls", ad.ZERO))
    # Normalize the nested parameter tangent before deciding which callback
    # derivative contracts are needed.  A zero tangent for one coefficient
    # must not force metadata for unrelated coefficients.
    supplied = params if params is not None else controls
    supplied_names_hint = None
    if isinstance(supplied, Mapping):
        supplied_names_hint = tuple(supplied)
    states, sensitivities, pmap = _forward(
        H, psi0, t, params, controls, derivatives, t0,
        require_derivatives=False, sensitivity_names=(),
    )
    p = np.asarray(psi0)
    dpsi = tangents.get("psi0", ad.ZERO)
    dparams = tangents.get("params", tangents.get("controls", ad.ZERO))
    if dparams is ad.ZERO or dparams is None:
        dparams = {}
    if not isinstance(dparams, Mapping):
        arr = np.asarray(dparams)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        target_names = supplied_names_hint if supplied_names_hint is not None else tuple(pmap)
        if arr.size != len(target_names):
            raise ValueError("parameter tangent must contain one value per active control")
        dparams = dict(zip(target_names, arr.flat))
    else:
        unknown = set(dparams) - set(pmap)
        if unknown:
            raise TypeError(f"unknown dynamic control tangents: {sorted(unknown)!r}")
    active_names = tuple(name for name, value in dparams.items() if not _is_zero_tangent(value))
    if supplied_names_hint is not None:
        # ``params`` is allowed to omit callback defaults.  The pmap still
        # contains those defaults for primal evaluation, but they cannot be
        # activated by an omitted nested tangent.
        omitted = set(dparams) - set(supplied_names_hint)
        if omitted:
            raise TypeError(f"dynamic control tangents were not supplied: {sorted(omitted)!r}")
        active_names = tuple(name for name in active_names if name in supplied_names_hint)
    if active_names:
        states, sensitivities, pmap = _forward(
            H, psi0, t, params, controls, derivatives, t0,
            require_derivatives=True, sensitivity_names=active_names,
        )
    out = np.zeros_like(states, dtype=np.result_type(states, dpsi if dpsi is not ad.ZERO else 0))
    if dpsi is not ad.ZERO:
        dpsi = np.asarray(dpsi)
        if dpsi.shape != p.shape: raise ValueError("dpsi0 shape must match psi0")
        # The initial-state tangent obeys the homogeneous Schrödinger equation
        # with the same time-dependent Hamiltonian.  Propagating it through a
        # second analytic solve is exact for the chosen forward integrator and
        # avoids the common (and incorrect) shortcut of assigning it only at
        # column zero.
        propagated, _, _ = _forward(
            H, dpsi, t, params, controls, derivatives=None, t0=t0,
            require_derivatives=False,
        )
        out = out + propagated
    for k, v in dparams.items():
        if k in sensitivities: out = out + sensitivities[k] * v
    return states, out


@ad.rules.jvp_for(dynamic_trajectory)
def _dynamic_jvp(tangents, hamiltonian, psi0, *args, params=None,
                 derivatives=None, checkpoint_interval=None, objective=None,
                 t0=None, times=None, controls=None):
    _validate_checkpoint(checkpoint_interval)
    allowed = {"psi0", "params", "controls"}
    bad = set(tangents) - allowed
    if bad: raise ad.UnsupportedWrt(dynamic_trajectory, bad, supported=allowed)
    t0, t = _parse_call(args, t0, times)
    states, tangent = _tangent(
        hamiltonian, psi0, t, params, controls, derivatives, tangents, t0
    )
    if objective is None:
        return states, tangent
    return _objective_jvp(objective, states, tangent)


@ad.rules.vjp_for(dynamic_trajectory)
def _dynamic_vjp(wrt, hamiltonian, psi0, *args, params=None,
                 derivatives=None, checkpoint_interval=None, objective=None,
                 t0=None, times=None, controls=None):
    _validate_checkpoint(checkpoint_interval)
    allowed = {"psi0", "params", "controls"}
    bad = set(wrt) - allowed
    if bad: raise ad.UnsupportedWrt(dynamic_trajectory, bad, supported=allowed)
    t0, t = _parse_call(args, t0, times)
    checkpoint = checkpoint_interval is not None
    supplied = params if params is not None else controls
    supplied_is_mapping = isinstance(supplied, Mapping)
    supplied_array = None if supplied_is_mapping or supplied is None else np.asarray(supplied)
    # Resolve the callback/default controls once so nested VJP output mirrors
    # the caller's parameter structure rather than exposing omitted defaults.
    _, _, pmap_for_names = _forward(
        hamiltonian, psi0, t[:1], params, controls, derivatives, t0,
        require_derivatives=False, sensitivity_names=(),
    )
    requested_control_names = _supplied_controls(params, controls, pmap_for_names)
    unknown_requested = set(requested_control_names) - set(pmap_for_names)
    if unknown_requested:
        raise TypeError(f"unknown dynamic controls: {sorted(unknown_requested)!r}")
    active_drive_names = tuple(requested_control_names) if any(
        name in wrt for name in ("params", "controls")
    ) else ()
    need_drive = bool(active_drive_names)
    states, sensitivities, pmap = _forward(
        hamiltonian, psi0, t, params, controls, derivatives, t0,
        require_derivatives=need_drive,
        sensitivity_names=() if checkpoint or not need_drive else active_drive_names,
    )
    value = objective(states) if objective is not None else states
    p = np.asarray(psi0)
    def pullback(cotangent):
        if cotangent is ad.ZERO: return dict.fromkeys(wrt, ad.ZERO)
        if objective is not None:
            g = _objective_cotangent(objective, states, cotangent)
        else:
            g = np.asarray(cotangent)
            if g.shape != states.shape:
                raise ValueError("trajectory cotangent must match output shape")
        gradp = None
        if checkpoint:
            # A checkpointed reverse sweep recomputes each segment and
            # propagates one adjoint, so no per-control sensitivity trajectory
            # is retained.  This is also the correct path when only psi0 is
            # active: the same adjoint supplies its real-linear cotangent.
            gradp, grad_by_name, _ = _checkpointed_reverse(
                hamiltonian, psi0, t, params, controls, derivatives, g,
                t0, checkpoint_interval, need_drive,
                sensitivity_names=active_drive_names,
            )
        elif not need_drive:
            grad_by_name = {}
            gradp = None
        else:
            grad_by_name = {
                k: _parameter_cotangent(
                    pmap[k], np.vdot(g, sensitivities[k])
                )
                for k in active_drive_names
            }
        if gradp is None:
            gradp = np.zeros_like(p, dtype=np.result_type(p, g, np.complex128))
            eye = np.eye(p.size, dtype=np.result_type(p, np.complex128))
            for j in range(p.size):
                basis, _, _ = _forward(
                    hamiltonian, eye[:, j], t, params, controls, derivatives, t0,
                    require_derivatives=need_drive,
                    sensitivity_names=() if not need_drive else active_drive_names,
                )
                coefficient = np.vdot(g, basis)
                # For a complex initial state the map is complex-linear but
                # the cotangent pairing is real-linear:
                # Re(vdot(cotangent, JVP(dpsi)))
                # = Re(vdot(gradient, dpsi)).  The returned complex cotangent
                # is therefore the conjugate of the basis coefficient.
                gradp[j] = (
                    coefficient if not np.iscomplexobj(p) else np.conj(coefficient)
                )
        result = {}
        if "psi0" in wrt: result["psi0"] = np.real(gradp) if not np.iscomplexobj(p) else gradp
        if "params" in wrt:
            if supplied_array is not None:
                names = tuple(active_drive_names)
                result["params"] = np.asarray([grad_by_name[n] for n in names]).reshape(supplied_array.shape)
            elif supplied_is_mapping:
                result["params"] = {k: grad_by_name[k] for k in requested_control_names}
            else:
                result["params"] = grad_by_name
        if "controls" in wrt:
            if supplied_array is not None:
                names = tuple(active_drive_names)
                result["controls"] = np.asarray([grad_by_name[n] for n in names]).reshape(supplied_array.shape)
            elif supplied_is_mapping:
                result["controls"] = {k: grad_by_name[k] for k in requested_control_names}
            else:
                result["controls"] = grad_by_name
        return result
    return value, pullback


def trajectory_jvp(hamiltonian, psi0, *args, tangents, **kwargs):
    """Convenience wrapper around ``chainrules.jvp(dynamic_trajectory, ...)``."""
    return ad.jvp(
        dynamic_trajectory, hamiltonian, psi0, *args,
        tangents=tangents, **kwargs,
    )


def trajectory_vjp(hamiltonian, psi0, *args, wrt, **kwargs):
    """Convenience wrapper around ``chainrules.vjp(dynamic_trajectory, ...)``."""
    return ad.vjp(
        dynamic_trajectory, hamiltonian, psi0, *args,
        wrt=wrt, **kwargs,
    )
