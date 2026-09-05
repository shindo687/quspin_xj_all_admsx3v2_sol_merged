"""ChainRules rules for small, continuous QuSpin API functions.

The wrappers below deliberately do not duplicate QuSpin's numerical
implementations.  Each wrapper calls its upstream function for the primal
value, and the registered rules contain only the corresponding linear map or
adjoint map.  Matrix and state dimensions are fixed while differentiating;
basis construction, sparse operator assembly, eigensolver choices and other
discrete operations are outside this module's support domain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

try:  # Prefer the standalone ChainRules package when available.
    import chainrules as ad
except ModuleNotFoundError:  # pragma: no cover - exercised in clean offline venvs
    from . import _chainrules as ad


def _is_jet(value: object) -> bool:
    """Whether ``value`` is the bundled composability tracer."""
    jet_type = getattr(ad, "_Jet", ())
    return bool(jet_type) and isinstance(value, jet_type)


def _has_jet(*values: object) -> bool:
    return any(_is_jet(value) for value in values)


def _zero_like(value: object) -> object:
    return np.zeros_like(value.value if _is_jet(value) else np.asarray(value))


def _native(path: str) -> Callable[..., Any]:
    """Resolve an upstream callable lazily.

    Lazy resolution keeps ``quspin_ad`` importable while a user is preparing a
    fresh environment.  No fallback implementation is provided: calling a
    wrapper without QuSpin installed raises the normal import error.
    """
    module_name, name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


def _unsupported(
    function: Callable[..., Any], names: Iterable[str], supported: Iterable[str]
) -> None:
    bad = set(names) - set(supported)
    if bad:
        raise ad.UnsupportedWrt(function, bad, supported=supported)


def _active(tangents: Mapping[str, object], name: str) -> object:
    return tangents.get(name, ad.ZERO)


def _array(value: object, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy controls the error
        raise TypeError(f"{name} must be array-like") from exc


def _same_shape(value: object, reference: np.ndarray, *, name: str) -> np.ndarray:
    array = _array(value, name=name)
    if array.shape != reference.shape:
        raise ValueError(f"{name} shape {array.shape} does not match {reference.shape}")
    return array


def _input_gradient(value: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Project a real-linear gradient back to the primal input dtype.

    QuSpin accepts real arrays for several callables whose outputs may be
    complex.  Under ChainRules' real inner product, a real input has a real
    cotangent; discard the imaginary component introduced by a complex output
    cotangent in that case.
    """
    return np.real(gradient) if not np.iscomplexobj(value) else gradient


def KL_div(p1: object, p2: object) -> Any:
    """Call :func:`quspin.tools.misc.KL_div` (primal only)."""
    if _has_jet(p1, p2):
        p1_value = np.asarray(p1.value if _is_jet(p1) else p1)
        p2_value = np.asarray(p2.value if _is_jet(p2) else p2)
        if p1_value.ndim != 1 or p2_value.ndim != 1 or p1_value.shape != p2_value.shape:
            raise TypeError("KL_div AD requires same-shaped one-dimensional distributions")
        if np.any(p1_value <= 0.0) or np.any(p2_value <= 0.0):
            raise TypeError("KL_div AD requires strictly positive distributions")
        if abs(np.sum(p1_value) - 1.0) > 1e-13 or abs(np.sum(p2_value) - 1.0) > 1e-13:
            raise ValueError("KL_div AD requires normalized distributions")
        # This is algebraically identical to QuSpin's implementation and is
        # deliberately written with composable NumPy operations for HVPs.
        return np.sum(p1 * (np.log(p1) - np.log(p2)))
    return _native("quspin.tools.misc.KL_div")(p1, p2)


@ad.rules.jvp_for(KL_div)
def _kl_jvp(
    tangents: Mapping[str, object], p1: object, p2: object
) -> tuple[Any, object]:
    if _has_jet(p1, p2):
        value = KL_div(p1, p2)
        dp1, dp2 = _active(tangents, "p1"), _active(tangents, "p2")
        if dp1 is ad.ZERO:
            dp1 = _zero_like(p1)
        if dp2 is ad.ZERO:
            dp2 = _zero_like(p2)
        tangent = (np.log(p1 / p2) + 1.0) * dp1 - (p1 / p2) * dp2
        return value, tangent
    value = KL_div(p1, p2)
    _unsupported(KL_div, tangents, ("p1", "p2"))
    dp1 = _active(tangents, "p1")
    dp2 = _active(tangents, "p2")
    if dp1 is ad.ZERO and dp2 is ad.ZERO:
        return value, ad.ZERO
    x = _array(p1, name="p1")
    y = _array(p2, name="p2")
    tangent = 0.0
    if dp1 is not ad.ZERO:
        tangent = tangent + np.sum(
            (np.log(x / y) + 1.0) * _same_shape(dp1, x, name="dp1")
        )
    if dp2 is not ad.ZERO:
        tangent = tangent - np.sum((x / y) * _same_shape(dp2, y, name="dp2"))
    return value, tangent


@ad.rules.vjp_for(KL_div)
def _kl_vjp(
    wrt: tuple[str, ...], p1: object, p2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(KL_div, wrt, ("p1", "p2"))
    value = KL_div(p1, p2)
    x = _array(p1, name="p1")
    y = _array(p2, name="p2")
    g1 = np.log(x / y) + 1.0
    g2 = -(x / y)

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        result: dict[str, object] = {}
        if "p1" in wrt:
            result["p1"] = _input_gradient(x, np.asarray(cotangent) * g1)
        if "p2" in wrt:
            result["p2"] = _input_gradient(y, np.asarray(cotangent) * g2)
        return result

    return value, pullback


def coherent_state(a: object, n: int, dtype: object = np.float64) -> Any:
    """Call :func:`quspin.basis.coherent_state` (primal only)."""
    if _is_jet(a):
        if np.asarray(a.value).ndim != 0:
            raise TypeError("coherent_state AD currently requires scalar a")
        if np.asarray(a.value) == 0 or not np.all(np.isfinite(np.asarray(a.value))):
            raise ad.NonDifferentiablePoint(
                "coherent_state has no stable rule at a=0 or non-finite amplitude"
            )
        k = np.arange(int(n))
        # exp(-|a|²/2) a**k / sqrt(k!) is the exact upstream construction.
        factorial = np.cumprod(np.arange(1, int(n) + 1, dtype=float))
        factorial = np.concatenate(([1.0], factorial[:-1])) if int(n) else np.array([], dtype=float)
        state = (np.exp(-0.5 * (np.abs(a) ** 2)) * (a ** k)) / np.sqrt(factorial)
        target_dtype = np.result_type(np.asarray(a.value).dtype, dtype)
        return state.astype(target_dtype)
    return _native("quspin.basis.coherent_state")(a, n, dtype=dtype)


def _coherent_linearization(value: np.ndarray, a: object, da: object) -> np.ndarray:
    aa = np.asarray(a)
    if aa.ndim != 0:
        raise TypeError("coherent_state AD currently requires scalar a")
    if aa == 0 or not np.all(np.isfinite(value)):
        raise ad.NonDifferentiablePoint(
            "coherent_state has no stable rule at a=0 or non-finite amplitude"
        )
    k = np.arange(value.size, dtype=np.result_type(value.dtype, np.float64))
    # Real-linear convention: |a|^2 contributes -Re(conj(a) da), while a^k
    # contributes k da/a.  This also specializes correctly to real ``a``.
    daa = np.asarray(da)
    if daa.ndim != 0:
        raise TypeError("coherent_state AD requires a scalar tangent da")
    logarithmic = -np.real(np.conj(aa) * daa) + k * daa / aa
    return value * logarithmic


@ad.rules.jvp_for(coherent_state)
def _coherent_jvp(
    tangents: Mapping[str, object], a: object, n: int, dtype: object = np.float64
) -> tuple[Any, object]:
    if _is_jet(a):
        value = coherent_state(a, n, dtype=dtype)
        da = _active(tangents, "a")
        if da is ad.ZERO:
            da = 0.0
        k = np.arange(np.asarray(value.value).size)
        return value, value * (k * da / a - np.real(np.conj(a) * da))
    value = coherent_state(a, n, dtype=dtype)
    _unsupported(coherent_state, tangents, ("a",))
    da = _active(tangents, "a")
    if da is ad.ZERO:
        return value, ad.ZERO
    return value, _coherent_linearization(np.asarray(value), a, da)


@ad.rules.vjp_for(coherent_state)
def _coherent_vjp(
    wrt: tuple[str, ...], a: object, n: int, dtype: object = np.float64
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(coherent_state, wrt, ("a",))
    value = coherent_state(a, n, dtype=dtype)
    aa = np.asarray(a)
    if aa.ndim != 0:
        raise TypeError("coherent_state AD currently requires scalar a")
    if aa == 0 or not np.all(np.isfinite(value)):
        raise ad.NonDifferentiablePoint(
            "coherent_state has no stable rule at a=0 or non-finite amplitude"
        )
    k = np.arange(np.asarray(value).size, dtype=np.result_type(value, np.float64))

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        c = _array(cotangent, name="cotangent").reshape(-1)
        v = np.asarray(value).reshape(-1)
        if c.shape != v.shape:
            raise ValueError("coherent_state cotangent must match the state shape")
        q = np.sum(np.conj(c) * v)
        r = np.sum(np.conj(c) * v * k)
        # g is defined by Re(conj(g) da) = Re(sum(conj(c) dstate)).
        g = -np.real(q) * aa + np.conj(r / aa)
        g = np.real(g) if not np.iscomplexobj(aa) else g
        return {"a": g}

    return value, pullback


def commutator(H1: object, H2: object) -> Any:
    """Call :func:`quspin.operators.commutator` (primal only)."""
    if _has_jet(H1, H2):
        a = np.asarray(H1.value if _is_jet(H1) else H1)
        b = np.asarray(H2.value if _is_jet(H2) else H2)
        if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
            raise TypeError("commutator AD requires equally shaped square dense matrices")
        return H1 @ H2 - H2 @ H1
    return _native("quspin.operators.commutator")(H1, H2)


def anti_commutator(H1: object, H2: object) -> Any:
    """Call :func:`quspin.operators.anti_commutator` (primal only)."""
    if _has_jet(H1, H2):
        a = np.asarray(H1.value if _is_jet(H1) else H1)
        b = np.asarray(H2.value if _is_jet(H2) else H2)
        if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
            raise TypeError("anti_commutator AD requires equally shaped square dense matrices")
        return H1 @ H2 + H2 @ H1
    return _native("quspin.operators.anti_commutator")(H1, H2)


def _matrix(value: object, *, name: str) -> np.ndarray:
    arr = _array(value, name=name)
    if arr.ndim != 2:
        raise TypeError(f"{name} AD domain is a rank-2 dense ndarray")
    return arr


def _binary_jvp(
    fn: Callable[..., Any],
    tangents: Mapping[str, object],
    H1: object,
    H2: object,
    plus: bool,
) -> tuple[Any, object]:
    if _has_jet(H1, H2):
        value = fn(H1, H2)
        d1, d2 = _active(tangents, "H1"), _active(tangents, "H2")
        terms = []
        if d1 is not ad.ZERO:
            terms.append(d1 @ H2 + (H2 @ d1 if plus else -(H2 @ d1)))
        if d2 is not ad.ZERO:
            terms.append(H1 @ d2 + (d2 @ H1 if plus else -(d2 @ H1)))
        tangent = terms[0] if len(terms) == 1 else terms[0] + terms[1] if terms else np.zeros_like(value)
        return value, tangent
    value = fn(H1, H2)
    _unsupported(fn, tangents, ("H1", "H2"))
    d1 = _active(tangents, "H1")
    d2 = _active(tangents, "H2")
    if d1 is ad.ZERO and d2 is ad.ZERO:
        return value, ad.ZERO
    a = _matrix(H1, name="H1")
    b = _matrix(H2, name="H2")
    tangent_dtype = np.result_type(np.asarray(value), a, b)
    if d1 is not ad.ZERO:
        tangent_dtype = np.result_type(tangent_dtype, d1)
    if d2 is not ad.ZERO:
        tangent_dtype = np.result_type(tangent_dtype, d2)
    tangent = np.zeros_like(np.asarray(value), dtype=tangent_dtype)
    if d1 is not ad.ZERO:
        da = _matrix(d1, name="dH1")
        if da.shape != a.shape:
            raise ValueError("dH1 shape must match H1")
        tangent = tangent + da @ b + (b @ da if plus else -(b @ da))
    if d2 is not ad.ZERO:
        db = _matrix(d2, name="dH2")
        if db.shape != b.shape:
            raise ValueError("dH2 shape must match H2")
        tangent = tangent + a @ db + (db @ a if plus else -(db @ a))
    return value, tangent


@ad.rules.jvp_for(commutator)
def _comm_jvp(
    tangents: Mapping[str, object], H1: object, H2: object
) -> tuple[Any, object]:
    return _binary_jvp(commutator, tangents, H1, H2, False)


@ad.rules.jvp_for(anti_commutator)
def _anti_jvp(
    tangents: Mapping[str, object], H1: object, H2: object
) -> tuple[Any, object]:
    return _binary_jvp(anti_commutator, tangents, H1, H2, True)


def _binary_vjp(
    fn: Callable[..., Any],
    wrt: tuple[str, ...],
    H1: object,
    H2: object,
    plus: bool,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(fn, wrt, ("H1", "H2"))
    value = fn(H1, H2)
    a = _matrix(H1, name="H1")
    b = _matrix(H2, name="H2")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _matrix(cotangent, name="cotangent")
        result: dict[str, object] = {}
        if "H1" in wrt:
            g_h1 = g @ b.conj().T + (b.conj().T @ g if plus else -(b.conj().T @ g))
            result["H1"] = _input_gradient(a, g_h1)
        if "H2" in wrt:
            g_h2 = a.conj().T @ g + (g @ a.conj().T if plus else -(g @ a.conj().T))
            result["H2"] = _input_gradient(b, g_h2)
        return result

    return value, pullback


@ad.rules.vjp_for(commutator)
def _comm_vjp(
    wrt: tuple[str, ...], H1: object, H2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    return _binary_vjp(commutator, wrt, H1, H2, False)


@ad.rules.vjp_for(anti_commutator)
def _anti_vjp(
    wrt: tuple[str, ...], H1: object, H2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    return _binary_vjp(anti_commutator, wrt, H1, H2, True)


def ED_state_vs_time(
    psi: object, E: object, V: object, times: object, iterate: bool = False
) -> Any:
    """Call QuSpin's exact-diagonalization time evolution routine."""
    if _has_jet(psi, E, times):
        if iterate:
            raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
        p_value = np.asarray(psi.value if _is_jet(psi) else psi)
        e_value = np.asarray(E.value if _is_jet(E) else E)
        v_value = np.asarray(V.value if _is_jet(V) else V)
        t_value = np.asarray(times.value if _is_jet(times) else times)
        if (p_value.ndim != 1 or e_value.ndim != 1 or t_value.ndim != 1
                or p_value.size != e_value.size or v_value.shape != (e_value.size, e_value.size)):
            raise TypeError("ED_state_vs_time AD requires 1-D psi, E, times and square V")
        if np.iscomplexobj(e_value) or np.iscomplexobj(t_value):
            raise TypeError("ED_state_vs_time AD requires real E and times")
        # Keep the same phase/eigenvector orientation as QuSpin's pure-state
        # implementation while expressing the smooth path in composable
        # NumPy operations.
        phase = np.exp(-1j * times[:, None] * E[None, :])
        coeff = V.conj().T @ psi
        return V @ (phase * coeff[None, :]).T
    return _native("quspin.tools.evolution.ED_state_vs_time")(
        psi, E, V, times, iterate=iterate
    )


def _ed_forward(
    psi: object, E: object, V: object, times: object
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = ED_state_vs_time(psi, E, V, times, iterate=False)
    p = _array(psi, name="psi")
    e = _array(E, name="E")
    mat = _matrix(V, name="V")
    t = _array(times, name="times")
    if (
        p.ndim != 1
        or e.ndim != 1
        or t.ndim != 1
        or p.size != e.size
        or mat.shape != (e.size, e.size)
    ):
        raise TypeError("ED_state_vs_time AD requires 1-D psi, E, times and square V")
    if np.iscomplexobj(e) or np.iscomplexobj(t):
        raise TypeError("ED_state_vs_time AD requires real E and times")
    phase = np.exp(-1j * t[:, None] * e[None, :])
    coeff = mat.conj().T @ p
    return value, phase, coeff, mat, t


@ad.rules.jvp_for(ED_state_vs_time)
def _ed_jvp(
    tangents: Mapping[str, object],
    psi: object,
    E: object,
    V: object,
    times: object,
    iterate: bool = False,
) -> tuple[Any, object]:
    if _has_jet(psi, E, times):
        if iterate:
            raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
        value = ED_state_vs_time(psi, E, V, times, iterate=False)
        dpsi, dE, dt = _active(tangents, "psi"), _active(tangents, "E"), _active(tangents, "times")
        if dpsi is ad.ZERO:
            dpsi = _zero_like(psi)
        if dE is ad.ZERO:
            dE = _zero_like(E)
        if dt is ad.ZERO:
            dt = _zero_like(times)
        phase = np.exp(-1j * times[:, None] * E[None, :])
        coeff = V.conj().T @ psi
        dphase = phase * (-1j * (dt[:, None] * E[None, :] + times[:, None] * dE[None, :]))
        dc = V.conj().T @ dpsi
        return value, V @ (dphase * coeff[None, :] + phase * dc[None, :]).T
    if iterate:
        raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
    value, phase, coeff, mat, t = _ed_forward(psi, E, V, times)
    _unsupported(ED_state_vs_time, tangents, ("psi", "E", "times"))
    dpsi = _active(tangents, "psi")
    dE = _active(tangents, "E")
    dt = _active(tangents, "times")
    if dpsi is ad.ZERO and dE is ad.ZERO and dt is ad.ZERO:
        return value, ad.ZERO
    p = _array(psi, name="psi")
    e = _array(E, name="E")
    dc = np.zeros_like(coeff, dtype=np.result_type(coeff, np.complex128))
    if dpsi is not ad.ZERO:
        dc = dc + mat.conj().T @ _same_shape(dpsi, p, name="dpsi")
    de = (
        np.zeros_like(np.asarray(E), dtype=np.result_type(E, np.float64))
        if dE is ad.ZERO
        else _same_shape(dE, e, name="dE")
    )
    dtime = (
        np.zeros_like(t, dtype=np.result_type(t, np.float64))
        if dt is ad.ZERO
        else _same_shape(dt, t, name="dtimes")
    )
    dphase = phase * (
        -1j * (dtime[:, None] * _array(E, name="E")[None, :] + t[:, None] * de[None, :])
    )
    # QuSpin returns states in the ``(Hilbert, time)`` orientation for the
    # non-iterator pure-state path (``V.dot(psi_t.T)`` in upstream source).
    # Preserve that exact primal shape here; callers should not need to know
    # that the phase factors are assembled in ``(time, eigenstate)`` order.
    return value, mat @ (dphase * coeff[None, :] + phase * dc[None, :]).T


@ad.rules.vjp_for(ED_state_vs_time)
def _ed_vjp(
    wrt: tuple[str, ...],
    psi: object,
    E: object,
    V: object,
    times: object,
    iterate: bool = False,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    if iterate:
        raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
    _unsupported(ED_state_vs_time, wrt, ("psi", "E", "times"))
    value, phase, coeff, mat, t = _ed_forward(psi, E, V, times)
    e = _array(E, name="E")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _array(cotangent, name="cotangent")
        if g.shape != np.asarray(value).shape:
            raise ValueError("ED_state_vs_time cotangent must match output shape")
        # Y = V @ (phase * coeff).T, so first pull back through the final
        # matrix product and transpose back to phase's (time, eigenstate)
        # orientation.
        g_a = mat.conj().T @ g
        g_s = g_a.T
        g_coeff = np.sum(np.conj(phase) * g_s, axis=0)
        result: dict[str, object] = {}
        if "psi" in wrt:
            result["psi"] = _input_gradient(_array(psi, name="psi"), mat @ g_coeff)
        if "E" in wrt:
            dstate_dE = -1j * t[:, None] * phase * coeff[None, :]
            result["E"] = np.real(np.sum(np.conj(g_s) * dstate_dE, axis=0))
        if "times" in wrt:
            dstate_dt = -1j * phase * e[None, :] * coeff[None, :]
            result["times"] = np.real(np.sum(np.conj(g_s) * dstate_dt, axis=1))
        return result

    return value, pullback


def lin_comb_Q_T(coeff: object, Q_T: object, out: object = None) -> Any:
    """Call :func:`quspin.tools.lanczos.lin_comb_Q_T` (primal only)."""
    if _has_jet(coeff, Q_T):
        if out is not None:
            raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
        c_value = np.asarray(coeff.value if _is_jet(coeff) else coeff)
        q_value = np.asarray(Q_T.value if _is_jet(Q_T) else Q_T)
        if c_value.ndim != 1 or q_value.ndim != 2 or q_value.shape[0] != c_value.size:
            raise TypeError("lin_comb_Q_T AD requires coeff shape (m,) and Q_T shape (m,n)")
        return coeff @ Q_T
    return _native("quspin.tools.lanczos.lin_comb_Q_T")(coeff, Q_T, out=out)


def project_op(Obs: object, proj: object, dtype: object = np.complex128) -> Any:
    """Call QuSpin's observable projection routine (primal only)."""
    if _has_jet(Obs, proj):
        obs_shape = np.asarray(Obs.value if _is_jet(Obs) else Obs).shape
        proj_shape = np.asarray(proj.value if _is_jet(proj) else proj).shape
        if len(obs_shape) != 2 or obs_shape[0] != obs_shape[1] or len(proj_shape) != 2:
            raise TypeError("project_op AD requires dense square observables and rank-2 projectors")
        if proj_shape[0] == obs_shape[0]:
            result = proj.conj().T @ Obs @ proj
        elif proj_shape[1] == obs_shape[0]:
            result = proj @ Obs @ proj.conj().T
        else:
            raise ValueError("project_op observable/projector dimensions are incompatible")
        return {"Proj_Obs": result}
    return _native("quspin.tools.misc.project_op")(Obs, proj, dtype=dtype)


def _projection_inputs(
    Obs: object, proj: object
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate dense projection inputs and identify down/up orientation."""
    observable = _matrix(Obs, name="Obs")
    projector = _matrix(proj, name="proj")
    if observable.shape[0] != observable.shape[1]:
        raise TypeError("project_op AD requires a square observable")
    if projector.shape[0] == observable.shape[0]:
        return observable, projector, True
    if projector.shape[1] == observable.shape[0]:
        return observable, projector, False
    raise ValueError("project_op observable/projector dimensions are incompatible")


def _projection_jvp(
    tangents: Mapping[str, object], Obs: object, proj: object, dtype: object
) -> tuple[Any, object]:
    if _has_jet(Obs, proj):
        value = project_op(Obs, proj, dtype=dtype)
        d_obs, d_proj = _active(tangents, "Obs"), _active(tangents, "proj")
        if d_obs is ad.ZERO:
            d_obs = _zero_like(Obs)
        if d_proj is ad.ZERO:
            d_proj = _zero_like(proj)
        obs_shape = np.asarray(Obs.value if _is_jet(Obs) else Obs).shape
        proj_shape = np.asarray(proj.value if _is_jet(proj) else proj).shape
        if len(obs_shape) != 2 or obs_shape[0] != obs_shape[1] or len(proj_shape) != 2:
            raise TypeError("project_op AD requires dense square observables and rank-2 projectors")
        if proj_shape[0] == obs_shape[0]:
            derivative = d_proj.conj().T @ Obs @ proj + proj.conj().T @ d_obs @ proj + proj.conj().T @ Obs @ d_proj
        else:
            derivative = d_proj @ Obs @ proj.conj().T + proj @ d_obs @ proj.conj().T + proj @ Obs @ d_proj.conj().T
        return value, {"Proj_Obs": derivative}
    value = project_op(Obs, proj, dtype=dtype)
    _unsupported(project_op, tangents, ("Obs", "proj"))
    observable, projector, down = _projection_inputs(Obs, proj)
    d_obs = _active(tangents, "Obs")
    d_proj = _active(tangents, "proj")
    if d_obs is ad.ZERO and d_proj is ad.ZERO:
        return value, ad.ZERO
    derivative_obs = (
        np.zeros_like(observable)
        if d_obs is ad.ZERO
        else _same_shape(d_obs, observable, name="dObs")
    )
    derivative_proj = (
        np.zeros_like(projector)
        if d_proj is ad.ZERO
        else _same_shape(d_proj, projector, name="dproj")
    )
    if down:
        derivative = (
            derivative_proj.conj().T @ observable @ projector
            + projector.conj().T @ derivative_obs @ projector
            + projector.conj().T @ observable @ derivative_proj
        )
    else:
        derivative = (
            derivative_proj @ observable @ projector.conj().T
            + projector @ derivative_obs @ projector.conj().T
            + projector @ observable @ derivative_proj.conj().T
        )
    return value, {"Proj_Obs": derivative}


@ad.rules.jvp_for(project_op)
def _project_jvp(
    tangents: Mapping[str, object],
    Obs: object,
    proj: object,
    dtype: object = np.complex128,
) -> tuple[Any, object]:
    return _projection_jvp(tangents, Obs, proj, dtype)


@ad.rules.vjp_for(project_op)
def _project_vjp(
    wrt: tuple[str, ...],
    Obs: object,
    proj: object,
    dtype: object = np.complex128,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(project_op, wrt, ("Obs", "proj"))
    value = project_op(Obs, proj, dtype=dtype)
    observable, projector, down = _projection_inputs(Obs, proj)

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        if not isinstance(cotangent, Mapping) or set(cotangent) != {"Proj_Obs"}:
            raise TypeError("project_op cotangent must map 'Proj_Obs' to a matrix")
        g = _matrix(cotangent["Proj_Obs"], name="cotangent['Proj_Obs']")
        result: dict[str, object] = {}
        if down:
            if "Obs" in wrt:
                result["Obs"] = _input_gradient(
                    observable, projector @ g @ projector.conj().T
                )
            if "proj" in wrt:
                result["proj"] = _input_gradient(
                    projector,
                    observable @ projector @ g.conj().T
                    + observable.conj().T @ projector @ g,
                )
        else:
            if "Obs" in wrt:
                result["Obs"] = _input_gradient(
                    observable, projector.conj().T @ g @ projector
                )
            if "proj" in wrt:
                result["proj"] = _input_gradient(
                    projector,
                    g @ projector @ observable.conj().T
                    + g.conj().T @ projector @ observable,
                )
        return result

    return value, pullback


@ad.rules.jvp_for(lin_comb_Q_T)
def _lincomb_jvp(
    tangents: Mapping[str, object], coeff: object, Q_T: object, out: object = None
) -> tuple[Any, object]:
    if _has_jet(coeff, Q_T):
        if out is not None:
            raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
        value = lin_comb_Q_T(coeff, Q_T, out=out)
        dc, dq = _active(tangents, "coeff"), _active(tangents, "Q_T")
        if dc is ad.ZERO:
            dc = _zero_like(coeff)
        if dq is ad.ZERO:
            dq = _zero_like(Q_T)
        return value, dc @ Q_T + coeff @ dq
    if out is not None:
        raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
    value = lin_comb_Q_T(coeff, Q_T, out=out)
    _unsupported(lin_comb_Q_T, tangents, ("coeff", "Q_T"))
    dc = _active(tangents, "coeff")
    dq = _active(tangents, "Q_T")
    if dc is ad.ZERO and dq is ad.ZERO:
        return value, ad.ZERO
    c = _array(coeff, name="coeff")
    q = _array(Q_T, name="Q_T")
    if c.ndim != 1 or q.ndim != 2 or q.shape[0] != c.size:
        raise TypeError("lin_comb_Q_T AD requires coeff shape (m,) and Q_T shape (m,n)")
    tangent = np.zeros(q.shape[1], dtype=np.result_type(c, q))
    if dc is not ad.ZERO:
        tangent = tangent + _same_shape(dc, c, name="dcoeff") @ q
    if dq is not ad.ZERO:
        tangent = tangent + c @ _same_shape(dq, q, name="dQ_T")
    return value, tangent


@ad.rules.vjp_for(lin_comb_Q_T)
def _lincomb_vjp(
    wrt: tuple[str, ...], coeff: object, Q_T: object, out: object = None
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    if out is not None:
        raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
    _unsupported(lin_comb_Q_T, wrt, ("coeff", "Q_T"))
    value = lin_comb_Q_T(coeff, Q_T, out=out)
    c = _array(coeff, name="coeff")
    q = _array(Q_T, name="Q_T")
    if c.ndim != 1 or q.ndim != 2 or q.shape[0] != c.size:
        raise TypeError("lin_comb_Q_T AD requires coeff shape (m,) and Q_T shape (m,n)")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _array(cotangent, name="cotangent")
        if g.shape != (q.shape[1],):
            raise ValueError("lin_comb_Q_T cotangent must match output shape")
        result: dict[str, object] = {}
        if "coeff" in wrt:
            result["coeff"] = _input_gradient(c, q.conj() @ g)
        if "Q_T" in wrt:
            result["Q_T"] = _input_gradient(q, np.outer(c.conj(), g))
        return result

    return value, pullback


def register_upstream_rules() -> tuple[str, ...]:
    """Register rules for the actual upstream function identities when present.

    The sidecar wrappers are always registered.  This optional bridge lets a
    caller pass ``quspin.tools.misc.KL_div`` (rather than ``quspin_ad.KL_div``)
    to :func:`chainrules.jvp`/``vjp``.  Registration is best-effort and never
    supplies a non-QuSpin fallback primal.
    """
    registered: list[str] = []
    pairs = (
        ("quspin.tools.misc.KL_div", KL_div),
        ("quspin.basis.coherent_state", coherent_state),
        ("quspin.operators.commutator", commutator),
        ("quspin.operators.anti_commutator", anti_commutator),
        ("quspin.tools.evolution.ED_state_vs_time", ED_state_vs_time),
        ("quspin.tools.lanczos.lin_comb_Q_T", lin_comb_Q_T),
        ("quspin.tools.misc.project_op", project_op),
    )
    # RuleRegistry is identity based and intentionally rejects duplicate
    # registration, so only perform this bridge once per process.
    for path, wrapper in pairs:
        try:
            native = _native(path)
            if native is wrapper:
                continue
            if hasattr(ad.rules, "alias"):
                ad.rules.alias(native, wrapper)
            # Obtain the private rule functions by callable identity.  This is
            # preferable to maintaining a second, divergent implementation.
            dispatch = {
                KL_div: (_kl_jvp, _kl_vjp),
                coherent_state: (_coherent_jvp, _coherent_vjp),
                commutator: (_comm_jvp, _comm_vjp),
                anti_commutator: (_anti_jvp, _anti_vjp),
                ED_state_vs_time: (_ed_jvp, _ed_vjp),
                lin_comb_Q_T: (_lincomb_jvp, _lincomb_vjp),
                project_op: (_project_jvp, _project_vjp),
            }[wrapper]
            # The public registry has no contains operation; duplicate bridge
            # calls are harmlessly ignored based on RuleNotFound probing.
            try:
                ad.rules.get_jvp(native)
            except ad.RuleNotFound:
                ad.rules.jvp_for(native)(dispatch[0])
            try:
                ad.rules.get_vjp(native)
            except ad.RuleNotFound:
                ad.rules.vjp_for(native)(dispatch[1])
            registered.append(path)
        except (ImportError, ModuleNotFoundError):
            continue
    return tuple(registered)


# In normal installations QuSpin is present and explicit import registration
# is useful.  Keep failures silent so users may inspect/install the sidecar
# before installing QuSpin itself; invoking a wrapper still reports the real
# missing dependency.
register_upstream_rules()
