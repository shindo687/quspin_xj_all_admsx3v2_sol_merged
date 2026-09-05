from __future__ import annotations

import numpy as np
import pytest

import quspin_ad
import chainrules as ad


def test_nested_jvp_kl_and_bilinear_mixed_terms() -> None:
    p1 = np.array([0.2, 0.3, 0.5])
    p2 = np.array([0.3, 0.3, 0.4])
    d1 = np.array([0.1, -0.04, -0.06])
    d2 = np.array([-0.02, 0.01, 0.01])
    value, tangent, second = ad.nested_jvp(
        quspin_ad.KL_div, p1, p2, tangents={"p1": d1, "p2": d2}
    )
    eps = 1.0e-4
    oracle = (
        quspin_ad.KL_div(p1 + eps * d1, p2 + eps * d2)
        - 2.0 * value
        + quspin_ad.KL_div(p1 - eps * d1, p2 - eps * d2)
    ) / eps**2
    assert np.allclose(second, oracle, rtol=2e-6, atol=2e-6)

    a = np.arange(4, dtype=float).reshape(2, 2)
    b = np.array([[0.3, -0.2], [0.8, 0.1]])
    da = np.ones((2, 2))
    db = np.array([[0.2, 0.4], [-0.1, 0.3]])
    _, _, mixed = ad.nested_jvp(
        quspin_ad.commutator, a, b, tangents={"H1": da, "H2": db}
    )
    assert np.allclose(mixed, 2.0 * (da @ db - db @ da))


def test_value_grad_and_hvp_composes_ed_state_and_matches_oracle() -> None:
    rng = np.random.default_rng(61)
    n, nt = 3, 7
    V, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    energy = np.array([-0.4, 0.7, 1.2])
    times = np.linspace(0.0, 1.8, nt)
    direction = rng.normal(size=n)

    def loss(E):
        state = quspin_ad.ED_state_vs_time(psi, E, V, times)
        amplitude = np.sum(np.conj(psi[:, None]) * state, axis=0)
        return np.mean(np.log(np.abs(amplitude) ** 2 + 0.1))

    value, gradient, product = ad.value_grad_and_hvp(
        loss, energy, direction
    )
    assert np.isfinite(value)
    assert gradient["E"].shape == energy.shape
    eps = 1.0e-4
    oracle = (
        loss(energy + eps * direction)
        - 2.0 * value
        + loss(energy - eps * direction)
    ) / eps**2
    # The scalar contraction of HVP equals the second directional derivative.
    assert np.allclose(np.dot(product["E"], direction), oracle, rtol=2e-5, atol=2e-5)


def test_hvp_complex_real_linear_convention() -> None:
    rng = np.random.default_rng(62)
    x = rng.normal(size=4) + 1j * rng.normal(size=4)
    d = rng.normal(size=4) + 1j * rng.normal(size=4)

    def loss(z):
        return np.sum(np.abs(z) ** 2)

    value, gradient, product = ad.value_grad_and_hvp(loss, x, d)
    assert np.allclose(gradient["z"], 2.0 * x)
    assert np.allclose(product["z"], 2.0 * d)
    assert np.allclose(value, np.sum(np.abs(x) ** 2))


def test_remaining_primitives_second_order_oracles() -> None:
    rng = np.random.default_rng(63)

    a, da = 0.8 + 0.4j, 0.2 - 0.7j
    value, _, second = ad.nested_jvp(
        quspin_ad.coherent_state,
        a,
        6,
        dtype=np.complex128,
        tangents={"a": da},
    )
    eps = 1.0e-4
    oracle = (
        quspin_ad.coherent_state(a + eps * da, 6, dtype=np.complex128)
        - 2.0 * value
        + quspin_ad.coherent_state(a - eps * da, 6, dtype=np.complex128)
    ) / eps**2
    assert np.allclose(second, oracle, rtol=2e-5, atol=2e-7)

    h1 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    h2 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    dh1 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    dh2 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    _, _, second = ad.nested_jvp(
        quspin_ad.anti_commutator,
        h1,
        h2,
        tangents={"H1": dh1, "H2": dh2},
    )
    assert np.allclose(second, 2.0 * (dh1 @ dh2 + dh2 @ dh1))

    coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    q_t = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    dcoeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    dq = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    _, _, second = ad.nested_jvp(
        quspin_ad.lin_comb_Q_T,
        coeff,
        q_t,
        tangents={"coeff": dcoeff, "Q_T": dq},
    )
    assert np.allclose(second, 2.0 * dcoeff @ dq)

    observable = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    projector = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    d_observable = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    d_projector = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    value, _, second = ad.nested_jvp(
        quspin_ad.project_op,
        observable,
        projector,
        tangents={"Obs": d_observable, "proj": d_projector},
    )
    plus = quspin_ad.project_op(
        observable + eps * d_observable, projector + eps * d_projector
    )
    minus = quspin_ad.project_op(
        observable - eps * d_observable, projector - eps * d_projector
    )
    oracle = (
        plus["Proj_Obs"] - 2.0 * value["Proj_Obs"] + minus["Proj_Obs"]
    ) / eps**2
    assert np.allclose(second["Proj_Obs"], oracle, rtol=2e-5, atol=2e-6)


def test_hvp_symmetry_for_complex_state_loss() -> None:
    rng = np.random.default_rng(64)
    n, nt = 3, 5
    V, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    target = rng.normal(size=(n, nt)) + 1j * rng.normal(size=(n, nt))
    energy = np.array([-0.3, 0.4, 1.1])
    times = np.linspace(0.0, 0.8, nt)
    u = rng.normal(size=n) + 1j * rng.normal(size=n)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)

    def loss(state):
        evolved = quspin_ad.ED_state_vs_time(state, energy, V, times)
        return np.sum(np.abs(evolved - target) ** 2)

    hu = ad.hvp(loss, psi, u)["state"]
    hv = ad.hvp(loss, psi, v)["state"]
    assert np.allclose(
        np.real(np.vdot(u, hv)), np.real(np.vdot(v, hu)), rtol=2e-12, atol=2e-12
    )


def test_second_order_boundaries_remain_explicit() -> None:
    with pytest.raises(ad.NonDifferentiablePoint, match="a=0"):
        ad.nested_jvp(quspin_ad.coherent_state, 0.0, 4, tangents={"a": 1.0})

    psi = np.array([1.0, 0.0])
    energy = np.array([0.0, 1.0])
    times = np.array([0.0, 0.5])
    with pytest.raises(ad.NonDifferentiablePoint, match="iterate=False"):
        ad.nested_jvp(
            quspin_ad.ED_state_vs_time,
            psi,
            energy,
            np.eye(2),
            times,
            iterate=True,
            tangents={"E": np.ones(2)},
        )
    with pytest.raises(ad.NonDifferentiablePoint, match="out=None"):
        ad.nested_jvp(
            quspin_ad.lin_comb_Q_T,
            np.ones(2),
            np.ones((2, 2)),
            out=np.empty(2),
            tangents={"coeff": np.ones(2)},
        )
    with pytest.raises(ad.UnsupportedWrt, match="supported inputs"):
        ad.nested_jvp(
            quspin_ad.ED_state_vs_time,
            psi,
            energy,
            np.eye(2),
            times,
            tangents={"V": np.eye(2)},
        )
