"""Differentiable fixed-dimensional Floquet eigensystems.

The adapter operates on an already-built one-period propagator ``UF`` (or a
QuSpin ``Floquet`` object exposing its ``UF`` and ``T`` attributes).  It does
not differentiate an adaptive construction of that propagator.  Fixed-grid
drive-phase, gauge, and momentum controls are accepted either through exact
``control_derivatives`` matrices or through the documented bare-matrix
diagonal-generator convention.  Eigenvalue branches use the principal phase
``angle(lambda)`` and are therefore rejected at the branch cut or when
eigenvalues are spectrally degenerate.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from . import rules as _r


class FloquetResult(dict):
    """Mapping result with familiar ``EF``, ``thetaF``, ``VF`` and ``PF`` attrs."""

    __getattr__ = dict.__getitem__


_CONTROL_NAMES = ("drive_phase", "gauge", "momentum")


def _control_values(drive_phase=None, gauge=None, momentum=None, controls=None):
    values = {}
    if controls is not None:
        if not isinstance(controls, Mapping):
            raise TypeError("Floquet controls must be a mapping")
        unknown = set(controls) - set(_CONTROL_NAMES)
        if unknown:
            raise TypeError(f"unknown Floquet controls: {sorted(unknown)!r}")
        values.update(controls)
    for name, value in (("drive_phase", drive_phase), ("gauge", gauge),
                        ("momentum", momentum)):
        if value is not None:
            values[name] = value
    for name, value in values.items():
        arr = np.asarray(value)
        if arr.ndim != 0 or not np.isfinite(arr):
            raise TypeError(f"Floquet control {name!r} must be a finite scalar")
        values[name] = float(arr)
    return values


def _unpack(UF, T=None, controls=None):
    source = UF
    if hasattr(UF, "UF"):
        if T is None:
            T = UF.T
        UF = UF.UF
    elif callable(UF):
        # A fixed-grid control wrapper may provide a callable propagator.  It
        # must be evaluable analytically at the requested control values; no
        # finite-difference probing is performed here.
        values = {} if controls is None else dict(controls)
        try:
            UF = UF(**values)
        except TypeError:
            try:
                UF = UF(values)
            except TypeError:
                UF = UF()
    if T is None:
        T = 1.0
    U = np.asarray(UF)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise TypeError("Floquet UF must be square")
    if not np.isscalar(T) or not np.isfinite(T) or float(T) <= 0:
        raise ValueError("Floquet period T must be positive")
    return U, float(T), source


def _control_contract(source, derivatives=None, control_derivatives=None):
    contract = control_derivatives if control_derivatives is not None else derivatives
    if contract is None:
        for attr in ("control_derivatives", "derivatives", "dUF"):
            candidate = getattr(source, attr, None)
            if candidate is not None:
                contract = candidate
                break
    if contract is None:
        return None
    return contract


def _canonical_controls(U, values):
    """Return a deterministic fixed-dimensional phase/gauge/momentum model.

    A bare propagator has no physical lattice metadata from which to infer
    these controls.  We therefore use the standard diagonal generator
    convention below, while callers with a driven lattice can provide exact
    ``control_derivatives`` matrices instead.  The same transformation is
    used by the primal and its JVP, so branches remain consistent.
    """
    n = U.shape[0]
    q = np.arange(n, dtype=float) - (n - 1.0) / 2.0
    Q = np.diag(q)
    phase = values.get("drive_phase", 0.0)
    gauge = values.get("gauge", 0.0)
    momentum = values.get("momentum", 0.0)
    left = np.exp(1j * (phase + momentum * q + gauge * q))
    right = np.exp(-1j * gauge * q)
    controlled = left[:, None] * U * right[None, :]
    derivative = {
        "drive_phase": 1j * controlled,
        "gauge": 1j * (Q @ controlled - controlled @ Q),
        "momentum": 1j * (Q @ controlled),
    }
    return controlled, derivative


def _canonical_factors(n, values):
    q = np.arange(n, dtype=float) - (n - 1.0) / 2.0
    left = np.exp(
        1j * (values.get("drive_phase", 0.0)
              + values.get("momentum", 0.0) * q
              + values.get("gauge", 0.0) * q)
    )
    right = np.exp(-1j * values.get("gauge", 0.0) * q)
    return np.diag(left), np.diag(right)


def _prepare(UF, T=None, drive_phase=None, gauge=None, momentum=None,
             controls=None, derivatives=None, control_derivatives=None):
    values = _control_values(drive_phase, gauge, momentum, controls)
    U, period, source = _unpack(UF, T, values)
    contract = _control_contract(source, derivatives, control_derivatives)
    if contract is None:
        return (*_canonical_controls(U, values), period, values)
    if callable(contract):
        attempts = (
            lambda: contract(U, period, values),
            lambda: contract(U, values),
            lambda: contract(values),
            lambda: contract(),
        )
        for attempt in attempts:
            try:
                contract = attempt()
                break
            except TypeError:
                continue
    if not isinstance(contract, Mapping):
        raise TypeError(
            "Floquet control derivatives must map drive_phase, gauge, and "
            "momentum to UF matrices"
        )
    unknown = set(contract) - set(_CONTROL_NAMES)
    if unknown:
        raise TypeError(f"unknown Floquet control derivatives: {sorted(unknown)!r}")
    dmap = {}
    for name in contract:
        raw = contract[name]
        if callable(raw):
            attempts = (
                lambda: raw(U, period, values),
                lambda: raw(U, values),
                lambda: raw(values),
                lambda: raw(),
            )
            for attempt in attempts:
                try:
                    raw = attempt()
                    break
                except TypeError:
                    continue
        arr = np.asarray(raw)
        if arr.shape != U.shape:
            raise TypeError(
                f"Floquet derivative for {name!r} must match UF shape"
            )
        dmap[name] = arr
    return U, dmap, period, values


def _decomp(U, T, gap_tol=1e-10):
    if not np.isscalar(gap_tol) or float(gap_tol) < 0:
        raise ValueError("gap_tol must be a non-negative scalar")
    eigenvalues, vectors = np.linalg.eig(U)
    phase = np.angle(eigenvalues)
    if np.any(np.abs(np.abs(phase) - np.pi) <= float(gap_tol)):
        raise _r.ad.NonDifferentiablePoint(
            "Floquet eigenphase lies on the quasienergy branch cut"
        )
    gaps = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    np.fill_diagonal(gaps, np.inf)
    if gaps.size and np.min(gaps) <= float(gap_tol):
        raise _r.ad.NonDifferentiablePoint(
            "Floquet eigensystem is degenerate or has a closed spectral gap"
        )

    # Match QuSpin's ordering by ascending quasienergy.  Normalise and choose
    # a deterministic phase so finite-difference comparisons of VF are stable.
    order = np.argsort(-phase / T)
    eigenvalues = eigenvalues[order]
    phase = phase[order]
    vectors = vectors[:, order]
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    for j in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, j])))
        if np.abs(vectors[pivot, j]) > 0:
            vectors[:, j] *= np.exp(-1j * np.angle(vectors[pivot, j]))
    projectors = np.asarray(
        [np.outer(vectors[:, j], vectors[:, j].conj()) for j in range(vectors.shape[1])]
    )
    return FloquetResult(
        EF=-phase / T,
        thetaF=eigenvalues,
        VF=vectors,
        PF=projectors,
    )


def floquet_eigensystem(UF, T=None, gap_tol=1e-10, *, drive_phase=None,
                        gauge=None, momentum=None, controls=None,
                        derivatives=None, control_derivatives=None):
    U, _, period, values = _prepare(
        UF, T, drive_phase, gauge, momentum, controls, derivatives,
        control_derivatives,
    )
    return _decomp(U, period, gap_tol)


def floquet_quasienergies(UF, T=None, gap_tol=1e-10, *, drive_phase=None,
                          gauge=None, momentum=None, controls=None,
                          derivatives=None, control_derivatives=None):
    return floquet_eigensystem(
        UF, T, gap_tol, drive_phase=drive_phase, gauge=gauge,
        momentum=momentum, controls=controls, derivatives=derivatives,
        control_derivatives=control_derivatives,
    )["EF"]


floquet_spectrum = floquet_eigensystem


def _jvp(tangents, UF, T=None, gap_tol=1e-10, *, drive_phase=None,
         gauge=None, momentum=None, controls=None, derivatives=None,
         control_derivatives=None):
    U, control_map, period, values = _prepare(
        UF, T, drive_phase, gauge, momentum, controls, derivatives,
        control_derivatives,
    )
    value = _decomp(U, period, gap_tol)
    dU = tangents.get("UF", _r.ad.ZERO)
    dT = tangents.get("T", _r.ad.ZERO)
    if dU is _r.ad.ZERO:
        dU = np.zeros_like(U)
    else:
        dU = np.asarray(dU)
        if dU.shape != U.shape:
            raise ValueError("UF tangent must match UF shape")
        # A UF tangent is a direction in the bare input propagator.  Under
        # the canonical control convention it must pass through the same
        # left/right phase factors as the primal UF.
        if _control_contract(UF, derivatives, control_derivatives) is None:
            left_factor, right_factor = _canonical_factors(U.shape[0], values)
            dU = left_factor @ dU @ right_factor
    for name in _CONTROL_NAMES:
        tangent = tangents.get(name, _r.ad.ZERO)
        if tangent is _r.ad.ZERO:
            continue
        if name not in control_map:
            raise _r.ad.NonDifferentiablePoint(
                f"Floquet control derivative metadata is missing for {name!r}"
            )
        arr = np.asarray(tangent)
        if arr.ndim != 0 or not np.isfinite(arr):
            raise TypeError(f"Floquet tangent for {name!r} must be a finite scalar")
        dU = dU + float(arr) * control_map[name]
    if dT is _r.ad.ZERO:
        dt = 0.0
    else:
        dt_arr = np.asarray(dT)
        if dt_arr.ndim != 0 or not np.isfinite(dt_arr):
            raise TypeError("T tangent must be a finite scalar")
        dt = float(dt_arr)

    lam = value["thetaF"]
    vectors = value["VF"]
    # The inverse, rather than a conjugate transpose, is the correct left
    # eigenbasis for a general diagonalizable matrix.  For a unitary UF this
    # agrees up to round-off with VF.conj().T.
    left = np.linalg.inv(vectors)
    B = left @ dU @ vectors
    dlam = np.diag(B)
    dphase = np.imag(dlam / lam)
    dEF = -dphase / period + np.angle(lam) * dt / period**2

    # Differentiate right eigenvectors, then differentiate the normalisation
    # and pivot-phase gauge used by _decomp.
    dV = np.zeros_like(vectors, dtype=np.result_type(vectors, dU, np.complex128))
    for i in range(lam.size):
        for j in range(lam.size):
            if i != j:
                dV[:, i] += vectors[:, j] * (B[j, i] / (lam[i] - lam[j]))
        dV[:, i] -= vectors[:, i] * np.real(np.vdot(vectors[:, i], dV[:, i]))
        pivot = int(np.argmax(np.abs(vectors[:, i])))
        pivot_value = vectors[pivot, i]
        if np.abs(pivot_value) > 0:
            darg = np.imag(dV[pivot, i] / pivot_value)
            dV[:, i] -= 1j * darg * vectors[:, i]

    dP = np.asarray(
        [
            np.outer(dV[:, j], vectors[:, j].conj())
            + np.outer(vectors[:, j], dV[:, j].conj())
            for j in range(lam.size)
        ]
    )
    tangent = {"EF": dEF, "thetaF": dlam, "VF": dV, "PF": dP}
    if tangents and all(x is _r.ad.ZERO for x in tangents.values()):
        return value, _r.ad.ZERO
    return value, tangent


def _unsupported(tangents):
    bad = set(tangents) - {"UF", "T", *_CONTROL_NAMES}
    if bad:
        raise _r.ad.UnsupportedWrt(
            floquet_eigensystem, bad,
            supported=("UF", "T", *_CONTROL_NAMES),
        )


def _cotangent_contract(cotangent, tangent):
    if not isinstance(cotangent, Mapping):
        raise TypeError("Floquet eigensystem cotangent must be a mapping")
    total = 0.0
    for name in ("EF", "thetaF", "VF", "PF"):
        if name in cotangent:
            c = np.asarray(cotangent[name])
            d = np.asarray(tangent[name])
            if c.shape != d.shape:
                raise ValueError(f"Floquet cotangent {name!r} shape must match output")
            total += np.real(np.vdot(c, d))
    return float(total)


@_r.ad.rules.jvp_for(floquet_eigensystem)
def _fj(tangents, UF, T=None, gap_tol=1e-10, *, drive_phase=None,
        gauge=None, momentum=None, controls=None, derivatives=None,
        control_derivatives=None):
    _unsupported(tangents)
    return _jvp(
        tangents, UF, T, gap_tol, drive_phase=drive_phase, gauge=gauge,
        momentum=momentum, controls=controls, derivatives=derivatives,
        control_derivatives=control_derivatives,
    )


@_r.ad.rules.vjp_for(floquet_eigensystem)
def _fv(wrt, UF, T=None, gap_tol=1e-10, *, drive_phase=None, gauge=None,
        momentum=None, controls=None, derivatives=None,
        control_derivatives=None):
    _unsupported(dict.fromkeys(wrt))
    U, _, period, values = _prepare(
        UF, T, drive_phase, gauge, momentum, controls, derivatives,
        control_derivatives,
    )
    value = _decomp(U, period, gap_tol)

    def pullback(cotangent):
        if cotangent is _r.ad.ZERO:
            return dict.fromkeys(wrt, _r.ad.ZERO)
        gU = np.zeros_like(U, dtype=np.result_type(U, np.complex128))
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                basis = np.zeros_like(U, dtype=np.result_type(U, np.complex128))
                basis[i, j] = 1
                _, dr = _jvp({"UF": basis}, U, period, gap_tol)
                _, di = _jvp({"UF": 1j * basis}, U, period, gap_tol)
                gU[i, j] = _cotangent_contract(cotangent, dr) + 1j * _cotangent_contract(cotangent, di)
        result = {}
        if "UF" in wrt:
            # Under the bare-matrix convention the controlled propagator is
            # ``L @ UF @ R``.  Pull the Frobenius cotangent through that
            # similarity/global-phase map before returning the UF gradient.
            explicit = _control_contract(UF, derivatives, control_derivatives)
            if explicit is None:
                left_factor, right_factor = _canonical_factors(U.shape[0], values)
                result["UF"] = left_factor.conj().T @ gU @ right_factor.conj().T
            else:
                result["UF"] = gU
        if "T" in wrt:
            _, dt = _jvp({"T": 1.0}, U, period, gap_tol)
            result["T"] = _cotangent_contract(cotangent, dt)
        for name in _CONTROL_NAMES:
            if name in wrt:
                # ``U`` is already the controlled propagator returned by
                # ``_prepare``.  Pass its exact control derivative as a UF
                # direction so the canonical control transform is not applied
                # a second time during the pullback.
                _, control_map, _, _ = _prepare(
                    UF, T, drive_phase, gauge, momentum, controls,
                    derivatives, control_derivatives,
                )
                if name not in control_map:
                    raise _r.ad.NonDifferentiablePoint(
                        f"Floquet control derivative metadata is missing for {name!r}"
                    )
                _, dc = _jvp({"UF": control_map[name]}, U, period, gap_tol)
                result[name] = _cotangent_contract(cotangent, dc)
        return result

    return value, pullback


@_r.ad.rules.jvp_for(floquet_quasienergies)
def _qj(tangents, UF, T=None, gap_tol=1e-10, *, drive_phase=None,
        gauge=None, momentum=None, controls=None, derivatives=None,
        control_derivatives=None):
    _unsupported(tangents)
    value, tangent = _jvp(
        tangents, UF, T, gap_tol, drive_phase=drive_phase, gauge=gauge,
        momentum=momentum, controls=controls, derivatives=derivatives,
        control_derivatives=control_derivatives,
    )
    return value["EF"], _r.ad.ZERO if tangent is _r.ad.ZERO else tangent["EF"]


@_r.ad.rules.vjp_for(floquet_quasienergies)
def _qv(wrt, UF, T=None, gap_tol=1e-10, *, drive_phase=None, gauge=None,
        momentum=None, controls=None, derivatives=None,
        control_derivatives=None):
    _unsupported(dict.fromkeys(wrt))
    U, _, period, _ = _prepare(
        UF, T, drive_phase, gauge, momentum, controls, derivatives,
        control_derivatives,
    )
    value = _decomp(U, period, gap_tol)["EF"]

    def pullback(cotangent):
        if cotangent is _r.ad.ZERO:
            return dict.fromkeys(wrt, _r.ad.ZERO)
        _, pull = _fv(
            wrt, UF, T, gap_tol, drive_phase=drive_phase, gauge=gauge,
            momentum=momentum, controls=controls, derivatives=derivatives,
            control_derivatives=control_derivatives,
        )
        return pull({"EF": np.asarray(cotangent)})

    return value, pullback

