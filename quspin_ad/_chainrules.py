"""Small bundled ChainRules-compatible fallback for offline sidecar installs.

The official ``chainrules`` package is used whenever it is installed.  This
module implements the same narrow v0.1 protocol so that a wheel remains
usable in an isolated environment where that dependency is not mirrored.
It intentionally contains no numerical differentiation fallback.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from numbers import Real

import numpy as np


class _Zero:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ZERO"


ZERO = _Zero()


def _name(function: Callable[..., object]) -> str:
    return getattr(function, "__qualname__", repr(function))


class RuleNotFound(LookupError):
    def __init__(self, function: Callable[..., object], mode: str) -> None:
        super().__init__(f"No {mode.upper()} rule is registered for {_name(function)}")


class UnsupportedWrt(ValueError):
    def __init__(
        self,
        function: Callable[..., object],
        requested: Iterable[str],
        *,
        supported: Iterable[str] | None = None,
    ) -> None:
        self.function = function
        self.requested = tuple(sorted(requested))
        self.supported = None if supported is None else tuple(sorted(supported))
        message = (
            f"{_name(function)} does not support differentiation with respect to "
            f"{self.requested!r}"
        )
        if self.supported is not None:
            message += f"; supported inputs are {self.supported!r}"
        super().__init__(message)


class NonDifferentiablePoint(RuntimeError):
    pass


class _Jet:
    """A small forward-over-reverse real-linear tracing value.

    ``_Jet`` is intentionally private.  It is used by ``value_grad_and_hvp``
    to compose the registered primitive rules through ordinary NumPy code.
    A jet stores a primal value, one directional derivative, and its second
    directional derivative.  Each operation also records an analytic
    pullback and the directional derivative of that pullback; reverse
    accumulation therefore returns both a gradient and its HVP without
    evaluating the function at perturbed points.
    """

    __array_priority__ = 1001

    def __init__(
        self,
        value,
        tangent=0.0,
        second=0.0,
        *,
        parents=(),
        pullback=None,
        pullback_dot=None,
    ):
        self.value = np.asarray(value)
        self.tangent = np.broadcast_to(np.asarray(tangent), self.value.shape)
        self.second = np.broadcast_to(np.asarray(second), self.value.shape)
        self.parents = tuple(parents)
        self.pullback = pullback
        self.pullback_dot = pullback_dot

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return _jet_unary(self, np.real, lambda x: np.ones_like(x), lambda x: np.zeros_like(x), real_output=True)

    @property
    def imag(self):
        return _jet_unary(self, np.imag, lambda x: 1j * np.ones_like(x), lambda x: np.zeros_like(x), real_output=True)

    def __repr__(self):  # pragma: no cover - diagnostic only
        return f"_Jet({self.value!r}, tangent={self.tangent!r})"

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __bool__(self):
        return bool(self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return _jet_index(self, index)

    def __setitem__(self, index, value):
        # Mutation is deliberately not part of the AD contract.  The method
        # exists only so NumPy code can report a clear error instead of
        # silently mutating a traced value.
        raise NonDifferentiablePoint("in-place mutation of a traced value is unsupported")

    def reshape(self, *shape, **kwargs):
        return _jet_reshape(self, self.value.reshape(*shape, **kwargs).shape)

    def transpose(self, *axes):
        return _jet_transpose(self, axes if axes else None)

    def ravel(self, order="C"):
        return _jet_reshape(self, self.value.ravel(order).shape)

    def astype(self, dtype, **kwargs):
        return _jet_cast(self, dtype, **kwargs)

    def conj(self):
        return _jet_conj(self)

    def dot(self, other, out=None):
        if out is not None:
            raise NonDifferentiablePoint("dot(out=...) is unsupported for traced values")
        return _jet_binary(self, other, "matmul")

    def __neg__(self):
        return _jet_unary(self, np.negative, lambda x: -np.ones_like(x), lambda x: np.zeros_like(x))

    def __pos__(self):
        return self

    def __abs__(self):
        return _jet_abs(self)

    def __add__(self, other):
        return _jet_binary(self, other, "add")

    def __radd__(self, other):
        return _jet_binary(other, self, "add")

    def __sub__(self, other):
        return _jet_binary(self, other, "sub")

    def __rsub__(self, other):
        return _jet_binary(other, self, "sub")

    def __mul__(self, other):
        return _jet_binary(self, other, "mul")

    def __rmul__(self, other):
        return _jet_binary(other, self, "mul")

    def __truediv__(self, other):
        return _jet_binary(self, other, "div")

    def __rtruediv__(self, other):
        return _jet_binary(other, self, "div")

    def __pow__(self, other):
        return _jet_binary(self, other, "pow")

    def __rpow__(self, other):
        return _jet_binary(other, self, "pow")

    def __matmul__(self, other):
        return _jet_binary(self, other, "matmul")

    def __rmatmul__(self, other):
        return _jet_binary(other, self, "matmul")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NonDifferentiablePoint(f"NumPy operation {ufunc.__name__}/{method} is unsupported for traced values")
        # Upstream QuSpin uses in-place ufunc output for its phase array
        # (``np.exp(x, x)``).  Returning a fresh traced result is equivalent
        # for the subsequent expression and preserves the derivative tape.
        kwargs = {key: value for key, value in kwargs.items() if key != "out"}
        if ufunc is np.add:
            return _jet_binary(inputs[0], inputs[1], "add")
        if ufunc is np.subtract:
            return _jet_binary(inputs[0], inputs[1], "sub")
        if ufunc is np.multiply:
            return _jet_binary(inputs[0], inputs[1], "mul")
        if ufunc is np.true_divide:
            return _jet_binary(inputs[0], inputs[1], "div")
        if ufunc is np.power:
            return _jet_binary(inputs[0], inputs[1], "pow")
        if ufunc is np.matmul:
            return _jet_binary(inputs[0], inputs[1], "matmul")
        if ufunc is np.negative:
            return _jet_unary(inputs[0], np.negative, lambda x: -np.ones_like(x), lambda x: np.zeros_like(x))
        if ufunc is np.positive:
            return inputs[0]
        if ufunc is np.conjugate:
            return _jet_conj(inputs[0])
        if ufunc is np.exp:
            return _jet_unary(inputs[0], np.exp, np.exp, np.exp)
        if ufunc is np.log:
            return _jet_unary(inputs[0], np.log, lambda x: 1.0 / x, lambda x: -1.0 / (x * x))
        if ufunc is np.sqrt:
            return _jet_unary(inputs[0], np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 / (x * np.sqrt(x)))
        if ufunc is np.square:
            return _jet_unary(inputs[0], np.square, lambda x: 2.0 * x, lambda x: 2.0 * np.ones_like(x))
        if ufunc is np.absolute:
            return _jet_abs(inputs[0])
        if ufunc is np.real:
            return inputs[0].real
        if ufunc is np.imag:
            return inputs[0].imag
        if ufunc is np.equal or ufunc is np.not_equal or ufunc is np.less or ufunc is np.greater:
            return getattr(ufunc, method)(*(np.asarray(x) for x in inputs), **kwargs)
        raise NonDifferentiablePoint(f"NumPy ufunc {ufunc.__name__} is unsupported for traced values")

    def __array_function__(self, func, types, args, kwargs):
        if func is np.sum:
            return _jet_sum(*args, **kwargs)
        if func is np.mean:
            x = args[0]
            axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
            keepdims = kwargs.get("keepdims", False)
            summed = _jet_sum(x, axis=axis, keepdims=keepdims)
            count = np.asarray(x.value if _is_jet(x) else x).size if axis is None else np.asarray(x.value if _is_jet(x) else x).shape[axis]
            return summed / count
        if func is np.dot:
            return _jet_binary(args[0], args[1], "matmul")
        if func is np.trace:
            return _jet_trace(*args, **kwargs)
        if func is np.vdot:
            return _jet_vdot(*args, **kwargs)
        if func is np.outer:
            return _jet_outer(*args, **kwargs)
        if func is np.reshape:
            return _jet_reshape(args[0], args[1])
        if func is np.transpose:
            return _jet_transpose(args[0], kwargs.get("axes", args[1] if len(args) > 1 else None))
        if func is np.conj:
            return _jet_conj(args[0])
        if func is np.real:
            return args[0].real
        if func is np.imag:
            return args[0].imag
        if func is np.abs:
            return _jet_abs(args[0])
        if func is np.linalg.norm:
            return _jet_norm(*args, **kwargs)
        raise NonDifferentiablePoint(f"NumPy function {getattr(func, '__name__', func)!r} is unsupported for traced values")


def _is_jet(value):
    return isinstance(value, _Jet)


def _jet_parts(value):
    if _is_jet(value):
        return value.value, value.tangent, value.second, value
    array = np.asarray(value)
    return array, np.zeros_like(array, dtype=np.result_type(array, float)), np.zeros_like(array, dtype=np.result_type(array, float)), None


def _unbroadcast(value, shape):
    """Sum an elementwise cotangent back to a parent's broadcast shape."""
    out = np.asarray(value)
    while out.ndim > len(shape):
        out = np.sum(out, axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and out.shape[axis] != 1:
            out = np.sum(out, axis=axis, keepdims=True)
    return out.reshape(shape)


def _matmul_pullback(left, right, cotangent):
    """Real-linear adjoints for NumPy's 1-D/2-D matrix products."""
    la, ra = np.asarray(left), np.asarray(right)
    c = np.asarray(cotangent)
    if la.ndim == 1 and ra.ndim == 1:
        return c * np.conj(ra), c * np.conj(la)
    if la.ndim == 1 and ra.ndim == 2:
        return c @ np.conj(ra).T, np.outer(np.conj(la), c)
    if la.ndim == 2 and ra.ndim == 1:
        return np.outer(c, np.conj(ra)), np.conj(la).T @ c
    return c @ np.conj(ra).swapaxes(-1, -2), np.conj(la).swapaxes(-1, -2) @ c


def _jet_binary(left, right, kind):
    lv, lt, ls, lj = _jet_parts(left)
    rv, rt, rs, rj = _jet_parts(right)
    if kind == "add":
        value, tangent, second = lv + rv, lt + rt, ls + rs
        jl, jr = np.ones_like(lv + rv), np.ones_like(lv + rv)
        jld, jrd = np.zeros_like(jl), np.zeros_like(jr)
    elif kind == "sub":
        value, tangent, second = lv - rv, lt - rt, ls - rs
        jl, jr = np.ones_like(value), -np.ones_like(value)
        jld, jrd = np.zeros_like(jl), np.zeros_like(jr)
    elif kind == "mul":
        value = lv * rv
        tangent = lt * rv + lv * rt
        second = ls * rv + lv * rs + 2.0 * lt * rt
        jl, jr = rv, lv
        jld, jrd = rt, lt
    elif kind == "div":
        value = lv / rv
        tangent = (lt * rv - lv * rt) / (rv * rv)
        second = ((ls * rv - lv * rs) / (rv * rv)
                  - 2.0 * (lt * rv - lv * rt) * rt / (rv * rv * rv))
        jl, jr = 1.0 / rv, -lv / (rv * rv)
        jld, jrd = -rt / (rv * rv), -lt / (rv * rv) + 2.0 * lv * rt / (rv * rv * rv)
    elif kind == "pow":
        if _is_jet(right):
            # Variable exponents are uncommon in scientific objectives.  Use
            # the analytic exp(y log(x)) composition so both directions stay
            # exact, while retaining the real-domain error from NumPy.
            return _jet_exp_log_pow(left, right)
        value = lv ** rv
        tangent = rv * (lv ** (rv - 1)) * lt
        second = rv * (lv ** (rv - 1)) * ls + rv * (rv - 1) * (lv ** (rv - 2)) * lt * lt
        jl, jr = rv * lv ** (rv - 1), np.zeros_like(value)
        jld, jrd = rv * (rv - 1) * (lv ** (rv - 2)) * lt, np.zeros_like(value)
    elif kind == "matmul":
        value = lv @ rv
        tangent = lt @ rv + lv @ rt
        second = ls @ rv + lv @ rs + 2.0 * (lt @ rt)
        jl = jr = jld = jrd = None
    else:  # pragma: no cover
        raise AssertionError(kind)

    parents = tuple(x for x in (lj, rj) if x is not None)
    if not parents:
        return value

    def pb(c):
        if kind == "matmul":
            out = []
            left_grad, right_grad = _matmul_pullback(lv, rv, c)
            if lj is not None:
                out.append(left_grad)
            if rj is not None:
                out.append(right_grad)
            return tuple(out)
        out = []
        if lj is not None:
            out.append(_unbroadcast(c * np.conj(jl), lv.shape))
        if rj is not None:
            out.append(_unbroadcast(c * np.conj(jr), rv.shape))
        return tuple(out)

    def pbd(c):
        if kind == "matmul":
            out = []
            if lj is not None:
                # Differentiate the left adjoint with respect to the right
                # operand while holding the cotangent fixed.
                out.append(_matmul_pullback(lv, rt, c)[0])
            if rj is not None:
                out.append(_matmul_pullback(lt, rv, c)[1])
            return tuple(out)
        out = []
        if lj is not None:
            out.append(_unbroadcast(c * np.conj(jld), lv.shape))
        if rj is not None:
            out.append(_unbroadcast(c * np.conj(jrd), rv.shape))
        return tuple(out)

    return _Jet(value, tangent, second, parents=parents, pullback=pb, pullback_dot=pbd)


def _jet_exp_log_pow(left, right):
    # The operation is implemented through the same analytic primitives, so
    # it naturally records a composable tape for variable exponents.
    return _jet_unary(_jet_binary(right, _jet_unary(left, np.log, lambda x: 1.0 / x, lambda x: -1.0 / (x * x)), "mul"), np.exp, np.exp, np.exp)


def _jet_unary(x, fn, jac_fn, jac2_fn, real_output=False):
    xv, xt, xs, xj = _jet_parts(x)
    value = fn(xv)
    jac = jac_fn(xv)
    jac2 = jac2_fn(xv)
    tangent = jac * xt
    second = jac * xs + jac2 * xt * xt
    if real_output:
        tangent = np.real(tangent)
        second = np.real(second)
    if xj is None:
        return value

    def pb(c):
        return (c * np.conj(jac),)

    def pbd(c):
        return (c * np.conj(jac2 * xt),)

    return _Jet(value, tangent, second, parents=(xj,), pullback=pb, pullback_dot=pbd)


def _jet_conj(x):
    xv, xt, xs, xj = _jet_parts(x)
    if xj is None:
        return np.conj(xv)

    def pb(c):
        return (np.conj(c),)

    def pbd(c):
        return (np.conj(c),)

    return _Jet(np.conj(xv), np.conj(xt), np.conj(xs), parents=(xj,), pullback=pb, pullback_dot=pbd)


def _jet_abs(x):
    xv, xt, xs, xj = _jet_parts(x)
    value = np.abs(xv)
    if xj is None:
        return value
    if np.any(value == 0):
        raise NonDifferentiablePoint("absolute value has no stable second-order rule at zero")
    jac = np.conj(xv) / value
    # d|x| = Re(conj(x) dx)/|x|; the real-linear pullback coefficient is x/|x|.
    tangent = np.real(np.conj(xv) * xt) / value
    second = (np.real(np.conj(xv) * xs) / value
              + np.real(np.conj(xt) * xt) / value
              - np.real(np.conj(xv) * xt) ** 2 / value ** 3)

    def pb(c):
        return (c * xv / value,)

    def pbd(c):
        # Directional derivative of x/|x|.
        dv = tangent
        return (c * (xt / value - xv * dv / (value * value)),)

    return _Jet(value, tangent, second, parents=(xj,), pullback=pb, pullback_dot=pbd)


def _jet_sum(x, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if out is not None or where is not True:
        raise NonDifferentiablePoint("sum(out=...) and masked sum are unsupported for traced values")
    xv, xt, xs, xj = _jet_parts(x)
    value = np.sum(xv, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)
    tangent = np.sum(xt, axis=axis, keepdims=keepdims)
    second = np.sum(xs, axis=axis, keepdims=keepdims)
    if xj is None:
        return value
    def pb(c):
        expanded = np.asarray(c)
        if axis is not None and not keepdims:
            axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
            normalized = tuple(a if a >= 0 else xv.ndim + a for a in axes)
            target = list(expanded.shape)
            for a in sorted(normalized):
                target.insert(a, 1)
            expanded = expanded.reshape(target)
        return (np.broadcast_to(expanded, xv.shape),)
    def pbd(c):
        return (np.zeros_like(xv, dtype=np.result_type(c, xv)),)
    return _Jet(value, tangent, second, parents=(xj,), pullback=pb, pullback_dot=pbd)


def _jet_vdot(a, b, **kwargs):
    av, at, ass, aj = _jet_parts(a)
    bv, bt, bss, bj = _jet_parts(b)
    value = np.vdot(av, bv, **kwargs)
    tangent = np.vdot(at, bv) + np.vdot(av, bt)
    second = np.vdot(ass, bv) + np.vdot(av, bss) + 2.0 * np.vdot(at, bt)
    parents = tuple(x for x in (aj, bj) if x is not None)
    if not parents:
        return value
    def pb(c):
        out = []
        if aj is not None:
            out.append(c * np.conj(bv))
        if bj is not None:
            out.append(c * av)
        return tuple(out)
    def pbd(c):
        out = []
        if aj is not None:
            out.append(c * np.conj(bt))
        if bj is not None:
            out.append(c * at)
        return tuple(out)
    return _Jet(value, tangent, second, parents=parents, pullback=pb, pullback_dot=pbd)


def _jet_trace(x, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if out is not None or np.asarray(x.value if _is_jet(x) else x).ndim != 2:
        raise NonDifferentiablePoint("trace(out=...) and non-matrix trace are unsupported for traced values")
    xv = x.value if _is_jet(x) else np.asarray(x)
    rows, cols = xv.shape
    if offset >= 0:
        start = offset
        count = max(0, min(rows, cols - offset))
        idx = (np.arange(count), np.arange(offset, offset + count))
    else:
        start = -offset
        count = max(0, min(rows + offset, cols))
        idx = (np.arange(start, start + count), np.arange(count))
    return _jet_sum(x[idx], dtype=dtype)


def _jet_outer(a, b, out=None):
    if out is not None:
        raise NonDifferentiablePoint("outer(out=...) is unsupported for traced values")
    av, at, ass, aj = _jet_parts(a)
    bv, bt, bss, bj = _jet_parts(b)
    value = np.outer(av, bv)
    tangent = np.outer(at, bv) + np.outer(av, bt)
    second = np.outer(ass, bv) + np.outer(av, bss) + 2.0 * np.outer(at, bt)
    parents = tuple(x for x in (aj, bj) if x is not None)
    if not parents:
        return value
    def pb(c):
        outv = []
        if aj is not None:
            outv.append(np.sum(c * np.conj(bv), axis=1))
        if bj is not None:
            outv.append(np.sum(c * np.conj(av), axis=0))
        return tuple(outv)
    def pbd(c):
        outv = []
        if aj is not None:
            outv.append(np.sum(c * np.conj(bt), axis=1))
        if bj is not None:
            outv.append(np.sum(c * np.conj(at), axis=0))
        return tuple(outv)
    return _Jet(value, tangent, second, parents=parents, pullback=pb, pullback_dot=pbd)


def _jet_reshape(x, shape):
    xv, xt, xs, xj = _jet_parts(x)
    value, tangent, second = xv.reshape(shape), xt.reshape(shape), xs.reshape(shape)
    if xj is None:
        return value
    return _Jet(value, tangent, second, parents=(xj,), pullback=lambda c: (np.asarray(c).reshape(xv.shape),), pullback_dot=lambda c: (np.zeros_like(xv),))


def _jet_transpose(x, axes=None):
    xv, xt, xs, xj = _jet_parts(x)
    value, tangent, second = np.transpose(xv, axes), np.transpose(xt, axes), np.transpose(xs, axes)
    if xj is None:
        return value
    inv = None if axes is None else np.argsort(axes)
    return _Jet(value, tangent, second, parents=(xj,), pullback=lambda c: (np.transpose(c, inv),), pullback_dot=lambda c: (np.zeros_like(xv),))


def _jet_index(x, index):
    xv, xt, xs, xj = _jet_parts(x)
    value, tangent, second = xv[index], xt[index], xs[index]
    if xj is None:
        return value
    def pb(c):
        out = np.zeros_like(xv, dtype=np.result_type(c, xv))
        selected = np.asarray(xv[index])
        reduced = _unbroadcast(c, selected.shape)
        out[index] += reduced
        return (out,)
    return _Jet(value, tangent, second, parents=(xj,), pullback=pb, pullback_dot=lambda c: (np.zeros_like(xv),))


def _jet_cast(x, dtype, **kwargs):
    xv, xt, xs, xj = _jet_parts(x)
    value, tangent, second = xv.astype(dtype, **kwargs), xt.astype(dtype, **kwargs), xs.astype(dtype, **kwargs)
    if xj is None:
        return value
    # Casting is a fixed linear map; preserve real/complex cotangent behavior.
    return _Jet(value, tangent, second, parents=(xj,), pullback=lambda c: (np.asarray(c).astype(xv.dtype),), pullback_dot=lambda c: (np.zeros_like(xv),))


def _jet_norm(x, axis=None, ord=None, keepdims=False, **kwargs):
    if ord not in (None, 2):
        raise NonDifferentiablePoint("only the Euclidean norm is supported for traced values")
    return _jet_unary(_jet_sum(_jet_abs(x) ** 2, axis=axis, keepdims=keepdims), np.sqrt, lambda z: 0.5 / np.sqrt(z), lambda z: -0.25 / (z * np.sqrt(z)))


def _trace_reverse(output):
    """Return (gradient, HVP) maps for a scalar output jet."""
    order = []
    seen = set()
    def visit(node):
        if not _is_jet(node) or id(node) in seen:
            return
        seen.add(id(node))
        for parent in node.parents:
            visit(parent)
        order.append(node)
    visit(output)
    adj, adj_dot = {id(output): np.ones_like(output.value)}, {id(output): np.zeros_like(output.value)}
    for node in reversed(order):
        c = adj.get(id(node))
        cd = adj_dot.get(id(node))
        if node.pullback is None:
            continue
        pbs = node.pullback(c)
        pbs_dot = node.pullback(cd)
        pbd_coeff = node.pullback_dot(c)
        for parent, gp, gp_dot, gp_coeff_dot in zip(node.parents, pbs, pbs_dot, pbd_coeff):
            # Coefficient variation and the directional change in the
            # cotangent are both needed by forward-over-reverse.
            parent_id = id(parent)
            adj[parent_id] = adj.get(parent_id, 0) + gp
            adj_dot[parent_id] = adj_dot.get(parent_id, 0) + gp_dot + gp_coeff_dot
    return adj, adj_dot


class RuleRegistry:
    def __init__(self) -> None:
        self._jvp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
        self._vjp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
        self._aliases: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}

    def alias(self, function, wrapper):
        """Associate an upstream callable with a composable sidecar wrapper."""
        self._aliases[id(function)] = (function, wrapper)

    def primal_for(self, function):
        entry = self._aliases.get(id(function))
        if entry is not None and entry[0] is function:
            return entry[1]
        return function

    def _register(self, table, function):
        key = id(function)

        def decorator(rule):
            if key in table:
                raise RuntimeError(
                    f"A rule is already registered for {_name(function)}"
                )
            table[key] = (function, rule)
            return rule

        return decorator

    def jvp_for(self, function):
        return self._register(self._jvp, function)

    def vjp_for(self, function):
        return self._register(self._vjp, function)

    def _get(self, table, function, mode):
        entry = table.get(id(function))
        if entry is None or entry[0] is not function:
            raise RuleNotFound(function, mode)
        return entry[1]

    def get_jvp(self, function):
        return self._get(self._jvp, function, "JVP")

    def get_vjp(self, function):
        return self._get(self._vjp, function, "VJP")


rules = RuleRegistry()


def _signature_bind(function, args, kwargs):
    signature = inspect.signature(function)
    signature.bind(*args, **kwargs).apply_defaults()
    return signature


def _names(names, signature, label):
    names = (names,) if isinstance(names, str) else tuple(names)
    if not names:
        raise ValueError("wrt must contain at least one parameter name")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every name must be a string parameter name")
    if len(set(names)) != len(names):
        raise ValueError("wrt must contain unique parameter names")
    unknown = set(names) - set(signature.parameters)
    if unknown:
        raise TypeError(f"Unknown {label} parameter names: {sorted(unknown)!r}")
    return names


def jvp(function, /, *args, tangents, second_tangents=None, **kwargs):
    if second_tangents is not None:
        return nested_jvp(function, *args, tangents=tangents, second_tangents=second_tangents, **kwargs)
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    signature = _signature_bind(function, args, kwargs)
    _names(tuple(tangents), signature, "tangent")
    if not tangents or all(value is ZERO for value in tangents.values()):
        return function(*args, **kwargs), ZERO
    try:
        result = rules.get_jvp(function)(dict(tangents), *args, **kwargs)
    except RuleNotFound:
        # A composable Python loss may itself call ``jvp`` on registered
        # primitives.  Trace that loss on the bundled jet so nested JVPs do
        # not require a separate rule for every lambda/closure.
        value, tangent, _ = nested_jvp(function, *args, tangents=tangents, **kwargs)
        return value, tangent
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("A JVP rule must return a two-tuple")
    return result


def _contains_jet(value):
    if _is_jet(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_jet(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_jet(v) for v in value)
    return False


def _unwrap_jets(value, component="value"):
    if _is_jet(value):
        return getattr(value, component)
    if isinstance(value, Mapping):
        return type(value)((k, _unwrap_jets(v, component)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(_unwrap_jets(v, component) for v in value)
    if isinstance(value, list):
        return [_unwrap_jets(v, component) for v in value]
    return value


def nested_jvp(function, /, *args, tangents, second_tangents=None, **kwargs):
    """Evaluate a composable first- and second-directional JVP.

    ``tangents`` and ``second_tangents`` are mappings keyed by function
    parameter name.  The latter describes the second derivative of the input
    path (and defaults to zero), so for a straight line it returns the usual
    second directional derivative.  The result is ``(value, tangent,
    second_tangent)`` and supports scalar/array outputs and mappings returned
    by structured primitives.
    """
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    if second_tangents is None:
        second_tangents = {}
    if not isinstance(second_tangents, Mapping):
        raise TypeError("second_tangents must be a mapping from parameter names to values")
    function = getattr(rules, "primal_for", lambda fn: fn)(function)
    signature = _signature_bind(function, args, kwargs)
    names = _names(tuple(tangents), signature, "tangent")
    supported = {
        "KL_div": {"p1", "p2"},
        "coherent_state": {"a"},
        "commutator": {"H1", "H2"},
        "anti_commutator": {"H1", "H2"},
        "ED_state_vs_time": {"psi", "E", "times"},
        "lin_comb_Q_T": {"coeff", "Q_T"},
        "project_op": {"Obs", "proj"},
    }.get(getattr(function, "__name__", ""))
    if supported is not None and set(names) - supported:
        raise UnsupportedWrt(function, set(names) - supported, supported=supported)
    if second_tangents:
        _names(tuple(second_tangents), signature, "second tangent")
    if set(second_tangents) - set(names):
        raise ValueError("second_tangents may only name parameters present in tangents")
    traced_args = list(args)
    traced_kwargs = dict(kwargs)
    leaves = {}
    parameters = list(signature.parameters)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    for name in names:
        value = bound.arguments[name]
        tangent = tangents[name]
        if tangent is ZERO:
            tangent = np.zeros_like(np.asarray(value))
        second = second_tangents.get(name, 0.0)
        if second is ZERO:
            second = np.zeros_like(np.asarray(value))
        jet = _Jet(value, tangent, second)
        leaves[id(jet)] = (name, np.asarray(value))
        if name in parameters[: len(args)]:
            traced_args[parameters.index(name)] = jet
        else:
            traced_kwargs[name] = jet
    output = function(*traced_args, **traced_kwargs)
    if not _contains_jet(output):
        raise TypeError("nested_jvp requires a composable rule or a function that propagates traced values")
    return (_unwrap_jets(output, "value"), _unwrap_jets(output, "tangent"), _unwrap_jets(output, "second"))


# ``jvp2`` is a concise alias for callers that prefer the name used by some
# forward-mode AD systems.
jvp2 = nested_jvp


def _parse_hvp_call(function, args, wrt, vector, kwargs):
    """Normalize both keyword and ``loss(x, direction)`` call spellings."""
    signature = inspect.signature(function)
    positional = tuple(args)
    try:
        signature.bind(*positional, **kwargs)
    except TypeError:
        if vector is None and positional:
            candidate = positional[-1]
            try:
                signature.bind(*positional[:-1], **kwargs)
            except TypeError:
                raise
            positional = positional[:-1]
            vector = candidate
        else:
            raise
    if wrt is None:
        bound = signature.bind(*positional, **kwargs)
        bound.apply_defaults()
        # A scalar-loss helper normally has one active argument.  Infer that
        # spelling for the compact ``value_grad_and_hvp(loss, x, v)`` API.
        wrt = next(iter(signature.parameters))
    if vector is None:
        raise TypeError("vector is required for a Hessian-vector product")
    return positional, wrt, vector


def value_grad_and_hvp(function, /, *args, wrt=None, vector=None, **kwargs):
    """Return ``(value, gradient, Hessian @ vector)`` for a real scalar loss.

    This is a forward-over-reverse analytic calculation.  The function is
    evaluated once with traced inputs; registered QuSpin primitives and the
    supported NumPy operations record exact local derivatives.  No finite
    differences or perturbed primal evaluations are used.
    """
    function = getattr(rules, "primal_for", lambda fn: fn)(function)
    args, wrt, vector = _parse_hvp_call(function, args, wrt, vector, kwargs)
    signature = _signature_bind(function, args, kwargs)
    names = _names(wrt, signature, "wrt")
    if isinstance(vector, Mapping):
        vectors = vector
        unknown = set(vectors) - set(names)
        if unknown:
            raise TypeError(f"Unknown vector parameter names: {sorted(unknown)!r}")
    elif len(names) == 1:
        vectors = {names[0]: vector}
    else:
        raise TypeError("vector must be a mapping when wrt contains multiple inputs")
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    traced_args = list(args)
    traced_kwargs = dict(kwargs)
    leaves = {}
    parameters = list(signature.parameters)
    for name in names:
        value = bound.arguments[name]
        tangent = vectors.get(name, 0.0)
        array = np.asarray(value)
        tangent_array = np.asarray(tangent)
        if array.shape != tangent_array.shape:
            raise ValueError(f"vector for {name} shape {tangent_array.shape} does not match {array.shape}")
        jet = _Jet(array, tangent_array, np.zeros_like(array, dtype=np.result_type(array, tangent_array, float)))
        leaves[id(jet)] = (name, array)
        if name in parameters[: len(args)]:
            traced_args[parameters.index(name)] = jet
        else:
            traced_kwargs[name] = jet
    output = function(*traced_args, **traced_kwargs)
    if not _is_jet(output) or output.value.ndim != 0 or isinstance(output.value.item(), (bool, np.bool_)):
        raise TypeError("value_grad_and_hvp requires a composable real scalar output")
    if np.iscomplexobj(output.value) and abs(np.imag(output.value)) > 1e-12:
        raise TypeError("value_grad_and_hvp requires a real scalar output")
    adj, adj_dot = _trace_reverse(output)
    gradients, hvps = {}, {}
    for name in names:
        # Locate the leaf jet by walking bound arguments.  Leaves are not in
        # the output's parent list, so retain their IDs while constructing.
        candidate = None
        for value in traced_args:
            if _is_jet(value) and leaves.get(id(value), (None,))[0] == name:
                candidate = value
                break
        if candidate is None:
            for value in traced_kwargs.values():
                if _is_jet(value) and leaves.get(id(value), (None,))[0] == name:
                    candidate = value
                    break
        if candidate is None:
            raise RuntimeError(f"internal tracing error for input {name}")
        grad_value = adj.get(id(candidate), np.zeros_like(candidate.value))
        hvp_value = adj_dot.get(id(candidate), np.zeros_like(candidate.value))
        if not np.iscomplexobj(candidate.value):
            grad_value, hvp_value = np.real(grad_value), np.real(hvp_value)
        gradients[name] = grad_value
        hvps[name] = hvp_value
    return output.value.item() if output.value.shape == () else output.value, gradients, hvps


def hvp(function, /, *args, wrt=None, vector=None, **kwargs):
    """Return only the Hessian-vector-product mapping from
    :func:`value_grad_and_hvp`."""
    return value_grad_and_hvp(function, *args, wrt=wrt, vector=vector, **kwargs)[2]


def value_and_hvp(function, /, *args, wrt=None, vector=None, **kwargs):
    """Return ``(value, Hessian @ vector)`` for a real scalar loss."""
    value, _, product = value_grad_and_hvp(function, *args, wrt=wrt, vector=vector, **kwargs)
    return value, product


# Common spellings used by AD libraries and by early adopters of this sidecar.
value_and_grad_and_hvp = value_grad_and_hvp
value_grad_hvp = value_grad_and_hvp
hessian_vector_product = hvp


def vjp(function, /, *args, wrt, **kwargs):
    signature = _signature_bind(function, args, kwargs)
    names = _names(wrt, signature, "wrt")
    result = rules.get_vjp(function)(names, *args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2 or not callable(result[1]):
        raise TypeError("A VJP rule must return (value, pullback)")
    value, raw = result

    def pullback(cotangent):
        if cotangent is ZERO:
            return dict.fromkeys(names, ZERO)
        output = raw(cotangent)
        if not isinstance(output, Mapping) or set(output) != set(names):
            raise TypeError("Pullback keys must exactly match wrt")
        return {name: output[name] for name in names}

    return value, pullback


def grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("grad requires a single real scalar output")
    return pullback(1.0)


def value_and_grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value_and_grad requires a single real scalar output")
    return value, pullback(1.0)


__version__ = "0.1.0"
sys.modules.setdefault("chainrules", sys.modules[__name__])
