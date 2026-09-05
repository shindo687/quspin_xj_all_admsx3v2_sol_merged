from __future__ import annotations

import numpy as np
import pytest

import quspin_ad
import chainrules as ad
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def _sin_drive(t, amplitude):
    return amplitude * np.sin(t)


_sin_drive.derivative = lambda t, amplitude: np.sin(t)


def _cos_drive(t, offset):
    return offset * np.cos(t)


_cos_drive.derivative = lambda t, offset: np.cos(t)


def _fixture():
    basis = spin_basis_1d(L=1)
    H = hamiltonian(
        [],
        [
            ["x", [[1.0, 0]], _sin_drive, (0.7,)],
            ["z", [[1.0, 0]], _cos_drive, (-0.2,)],
        ],
        basis=basis,
        dtype=np.complex128,
    )
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    times = np.linspace(0.0, 1.0, 9)
    return H, psi0, times


def test_dynamic_trajectory_independent_controls_and_checkpointed_vjp():
    H, psi0, times = _fixture()
    params = {"amplitude": 0.7, "offset": -0.2}
    tangent_params = {"amplitude": 1.0, "offset": 0.0}
    value, tangent = ad.jvp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params=params,
        tangents={"params": tangent_params},
    )
    eps = 1e-5
    plus = quspin_ad.dynamic_trajectory(
        H, psi0, times, params={"amplitude": 0.7 + eps, "offset": -0.2}
    )
    minus = quspin_ad.dynamic_trajectory(
        H, psi0, times, params={"amplitude": 0.7 - eps, "offset": -0.2}
    )
    assert np.allclose(tangent, (plus - minus) / (2.0 * eps), rtol=3e-5, atol=3e-6)

    rng = np.random.default_rng(41)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    _, pullback = ad.vjp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params=params,
        checkpoint_interval=3,
        wrt=("params",),
    )
    gradients = pullback(cotangent)["params"]
    assert np.allclose(
        gradients["amplitude"], np.real(np.vdot(cotangent, tangent)), rtol=3e-5, atol=3e-6
    )
    assert abs(gradients["offset"]) < 1e6


class _FinalFidelity:
    def __init__(self, target):
        self.target = np.asarray(target)

    def __call__(self, states):
        overlap = np.vdot(self.target, states[:, -1])
        return float(np.real(np.conj(overlap) * overlap))

    def derivative(self, states):
        overlap = np.vdot(self.target, states[:, -1])
        result = np.zeros_like(states)
        result[:, -1] = 2.0 * overlap * self.target
        return result


def test_dynamic_final_fidelity_objective_jvp_vjp():
    H, psi0, times = _fixture()
    objective = _FinalFidelity([0.0, 1.0])
    params = {"amplitude": 0.7, "offset": -0.2}
    _, tangent = ad.jvp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params=params,
        objective=objective,
        tangents={"params": {"amplitude": 1.0, "offset": 0.0}},
    )
    eps = 1e-5
    plus = quspin_ad.dynamic_trajectory(
        H,
        psi0,
        times,
        params={"amplitude": 0.7 + eps, "offset": -0.2},
        objective=objective,
    )
    minus = quspin_ad.dynamic_trajectory(
        H,
        psi0,
        times,
        params={"amplitude": 0.7 - eps, "offset": -0.2},
        objective=objective,
    )
    assert np.allclose(tangent, (plus - minus) / (2.0 * eps), rtol=3e-5, atol=3e-6)

    _, pullback = ad.vjp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params=params,
        objective=objective,
        checkpoint_interval=2,
        wrt=("params",),
    )
    assert np.allclose(
        pullback(1.0)["params"]["amplitude"], tangent, rtol=3e-5, atol=3e-6
    )


def test_dynamic_plain_matrix_derivative_contract_and_boundaries():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.diag([1.0, -1.0]).astype(np.complex128)

    def matrix_drive(t, params):
        return params["amplitude"] * np.sin(t) * sx + sz

    def matrix_derivatives(t, params):
        return {"amplitude": np.sin(t) * sx}

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    times = np.linspace(0.0, 0.4, 5)
    value, tangent = ad.jvp(
        quspin_ad.dynamic_trajectory,
        matrix_drive,
        psi0,
        times,
        params={"amplitude": 0.3},
        derivatives=matrix_derivatives,
        tangents={"params": {"amplitude": 1.0}},
    )
    assert value.shape == (2, times.size)
    assert np.all(np.isfinite(tangent))
    with pytest.raises(ValueError, match="nondecreasing"):
        quspin_ad.dynamic_trajectory(
            matrix_drive,
            psi0,
            [0.0, 0.2, 0.1],
            params={"amplitude": 0.3},
            derivatives=matrix_derivatives,
        )


def test_dynamic_callback_without_derivative_contract_fails_only_for_sensitivity():
    def no_contract(t, amplitude):
        return amplitude * np.sin(t)

    H = hamiltonian(
        [],
        [["x", [[1.0, 0]], no_contract, (0.7,)]],
        basis=spin_basis_1d(L=1),
        dtype=np.complex128,
    )
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    times = np.linspace(0.0, 0.2, 3)
    # Forward evaluation is still the upstream path and needs no metadata.
    assert quspin_ad.dynamic_trajectory(
        H, psi0, times, params={"amplitude": 0.7}
    ).shape == (2, 3)
    with pytest.raises(ad.NonDifferentiablePoint, match="derivative contract"):
        ad.jvp(
            quspin_ad.dynamic_trajectory,
            H,
            psi0,
            times,
            params={"amplitude": 0.7},
            tangents={"params": {"amplitude": 1.0}},
        )


def test_dynamic_complex_control_real_linear_vjp_and_checkpoint_duality():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def matrix_drive(t, params):
        return params["amplitude"] * np.sin(t) * sx

    def matrix_derivatives(t, params):
        return {"amplitude": np.sin(t) * sx}

    psi0 = np.array([1.0 + 0.2j, 0.4 - 0.3j])
    times = np.linspace(0.0, 0.8, 7)
    params = {"amplitude": 0.7 + 0.2j}
    dparams = {"amplitude": 0.13 - 0.27j}
    _, tangent = ad.jvp(
        quspin_ad.dynamic_trajectory,
        matrix_drive,
        psi0,
        times,
        params=params,
        derivatives=matrix_derivatives,
        tangents={"params": dparams},
    )
    cotangent = np.random.default_rng(44).normal(size=(2, times.size))
    cotangent = cotangent + 1j * np.random.default_rng(45).normal(size=(2, times.size))
    for checkpoint_interval in (None, 2):
        _, pullback = ad.vjp(
            quspin_ad.dynamic_trajectory,
            matrix_drive,
            psi0,
            times,
            params=params,
            derivatives=matrix_derivatives,
            checkpoint_interval=checkpoint_interval,
            wrt=("params",),
        )
        gradient = pullback(cotangent)["params"]["amplitude"]
        assert np.allclose(
            np.real(np.vdot(cotangent, tangent)),
            np.real(np.conj(gradient) * dparams["amplitude"]),
            rtol=3e-5,
            atol=3e-6,
        )
        assert np.iscomplexobj(gradient)


def test_dynamic_partial_control_metadata_and_nested_shape():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.diag([1.0, -1.0]).astype(np.complex128)

    def active(t, amplitude):
        return amplitude * np.sin(t)

    active.derivative = lambda t, amplitude: np.sin(t)

    def inactive(t, offset):
        return offset * np.cos(t)

    H = hamiltonian(
        [],
        [
            ["x", [[1.0, 0]], active, (0.7,)],
            ["z", [[1.0, 0]], inactive, (-0.2,)],
        ],
        basis=spin_basis_1d(L=1),
        dtype=np.complex128,
    )
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    times = np.linspace(0.0, 0.5, 5)
    # Only the active callback is differentiated; missing metadata on the
    # unrelated callback must not block a valid partial JVP.
    _, tangent = ad.jvp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params={"amplitude": 0.7, "offset": -0.2},
        tangents={"params": {"amplitude": 1.0}},
    )
    assert tangent.shape == (2, times.size)
    _, pullback = ad.vjp(
        quspin_ad.dynamic_trajectory,
        H,
        psi0,
        times,
        params={"amplitude": 0.7},
        wrt=("params",),
    )
    gradient = pullback(np.ones_like(tangent))["params"]
    assert tuple(gradient) == ("amplitude",)
