from __future__ import annotations

import numpy as np
import pytest

import quspin_ad
import chainrules as ad


def _unitary(seed: int = 4, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    phases = np.linspace(-0.8, 0.9, n)
    return q @ np.diag(np.exp(1j * phases)) @ q.conj().T


def test_floquet_jvp_vjp_and_projectors_match_central_difference() -> None:
    rng = np.random.default_rng(9)
    U = _unitary()
    dU = rng.normal(size=U.shape) + 1j * rng.normal(size=U.shape)
    period = 2.0
    value, tangent = ad.jvp(
        quspin_ad.floquet_eigensystem, U, period, tangents={"UF": dU}
    )
    eps = 1e-6
    plus = quspin_ad.floquet_eigensystem(U + eps * dU, period)
    minus = quspin_ad.floquet_eigensystem(U - eps * dU, period)
    for name in ("EF", "thetaF", "VF", "PF"):
        oracle = (plus[name] - minus[name]) / (2.0 * eps)
        assert np.allclose(tangent[name], oracle, rtol=4e-5, atol=4e-6)
    assert value.EF.shape == (U.shape[0],)
    assert np.allclose(value.PF, np.swapaxes(value.PF.conj(), -1, -2))
    assert np.allclose(value.PF @ value.PF, value.PF)

    cotangent = {
        name: rng.normal(size=np.shape(value[name]))
        + 1j * rng.normal(size=np.shape(value[name]))
        for name in ("EF", "thetaF", "VF", "PF")
    }
    _, pullback = ad.vjp(
        quspin_ad.floquet_eigensystem, U, period, wrt=("UF",)
    )
    lhs = np.real(np.vdot(pullback(cotangent)["UF"], dU))
    rhs = sum(np.real(np.vdot(cotangent[k], tangent[k])) for k in cotangent)
    assert np.allclose(lhs, rhs, rtol=4e-5, atol=4e-6)


def test_floquet_quasienergy_branch_period_and_control_derivative() -> None:
    phases = np.array([-0.7, 0.35])
    U = np.diag(np.exp(1j * phases))
    period = 1.7
    dphase = 0.21
    value, tangent = ad.jvp(
        quspin_ad.floquet_quasienergies,
        U,
        period,
        drive_phase=0.13,
        tangents={"drive_phase": dphase, "T": 0.2},
    )
    eps = 1e-6
    plus = quspin_ad.floquet_quasienergies(
        U,
        period + eps * 0.2,
        drive_phase=0.13 + eps * dphase,
    )
    minus = quspin_ad.floquet_quasienergies(
        U,
        period - eps * 0.2,
        drive_phase=0.13 - eps * dphase,
    )
    assert np.allclose(value, quspin_ad.floquet_eigensystem(U, period, drive_phase=0.13).EF)
    assert np.allclose(tangent, (plus - minus) / (2.0 * eps), rtol=4e-5, atol=4e-6)

    # A physical fixed-grid adapter can replace the bare-matrix convention
    # with exact control derivatives supplied by the caller.
    control_derivatives = {"momentum": np.diag(1j * np.array([0.4, -0.2])) @ U}
    _, mtangent = ad.jvp(
        quspin_ad.floquet_eigensystem,
        U,
        period,
        control_derivatives=control_derivatives,
        tangents={"momentum": 1.0},
    )
    assert np.allclose(mtangent["EF"], np.array([0.2, -0.4]) / period)


def test_floquet_branch_and_gap_errors_are_explicit() -> None:
    with pytest.raises(ad.NonDifferentiablePoint, match="degenerate"):
        quspin_ad.floquet_eigensystem(np.eye(2), 2.0)
    with pytest.raises(ad.NonDifferentiablePoint, match="branch cut"):
        quspin_ad.floquet_eigensystem(np.diag([np.exp(1j * np.pi), 1.0]), 2.0)
    with pytest.raises((TypeError, ad.UnsupportedWrt)):
        ad.jvp(
            quspin_ad.floquet_eigensystem,
            np.diag(np.exp(1j * np.array([-0.8, 0.2]))),
            2.0,
            tangents={"bad": 1.0},
        )


def test_floquet_accepts_quspin_like_object_and_zero_pullback() -> None:
    class Source:
        UF = np.diag(np.exp(1j * np.array([-0.8, 0.2])))
        T = 2.0

    result = quspin_ad.floquet_spectrum(Source())
    assert np.array_equal(result.EF, quspin_ad.floquet_eigensystem(Source().UF, Source.T).EF)
    _, pullback = ad.vjp(
        quspin_ad.floquet_quasienergies, Source(), wrt=("UF", "T")
    )
    assert pullback(ad.ZERO) == {"UF": ad.ZERO, "T": ad.ZERO}
