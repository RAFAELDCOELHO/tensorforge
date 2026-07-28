"""
core/tensor.py — tensor autograd engine (M1).

Same graph machinery as engine.py's Value: each Tensor remembers its children
and a _backward closure; backward() walks a reverse topological sort.

The one new thing NumPy forces on us is broadcasting. The forward pass gets it
for free, the backward pass does not: if a (3,) bias was broadcast against a
(2,3) input, its gradient arrives shaped (2,3) and has to be summed back down
to (3,). That is what _unbroadcast does, and every op routes its gradient
through it.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Reduce grad back to `shape`, undoing whatever NumPy broadcast in the forward."""
    # extra leading axes NumPy prepended: sum them away
    for _ in range(grad.ndim - len(shape)):
        grad = grad.sum(axis=0)
    # axes that were size 1 and got stretched: sum but keep the dim
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        # A TUPLE, not a set. Iterating a set of objects follows hashes derived
        # from id(), which move between processes, so the reverse topological
        # order — and therefore the order the += accumulations land in .grad —
        # would change from run to run. Float addition is not associative, so
        # that alone made two identical runs differ by ~1 ulp per parameter.
        # Surfaced by the byte-for-byte test in tests/test_train.py.
        # Duplicates are fine: build_topo's `visited` already guards x + x.
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # ---- ops ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            # d(a/b)/da = 1/b ; d(a/b)/db = -a/b^2 (negative: bigger b, smaller q)
            ga = out.grad / other.data
            gb = -self.data / other.data ** 2 * out.grad
            self.grad += _unbroadcast(ga, self.shape)
            other.grad += _unbroadcast(gb, other.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.ndim >= 2 and other.data.ndim >= 2, \
            "matmul needs 2-D+ operands; reshape 1-D vectors yourself"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # dL/dA = G @ B^T, dL/dB = A^T @ G, transposing only the last two axes
            # so batch dims stay put; _unbroadcast then folds any batch dim that was
            # broadcast (e.g. a shared (K,M) weight against a (B,N,K) input).
            #
            # This holds for ANY number of leading axes, not just one, and that is
            # not an accident of the tests — nothing here counts batch dims:
            #   * swapaxes(-1,-2) indexes from the right, so it touches the last
            #     two axes and leaves however many precede them alone;
            #   * np.matmul already broadcasts all leading axes in the forward;
            #   * _unbroadcast loops over "however many extra axes there are"
            #     (sum them away) and then over every size-1 axis that got
            #     stretched (sum with keepdims). Neither loop assumes one batch dim.
            # Verified at (B,H) — Q@K^T and attn@V — and at (B,H,W).
            ga = out.grad @ other.data.swapaxes(-1, -2)
            gb = self.data.swapaxes(-1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.shape)
            other.grad += _unbroadcast(gb, other.shape)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2, reusing the output so the forward's
            # tanh is not paid for twice. Note 1 - t**2, NOT (1 - t)**2: the two
            # agree at x=0 and diverge by orders of magnitude once t nears 1.
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        """GELU, tanh approximation — GPT-2's gelu_new.

            0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        Pure composition of mul/add/tanh, each already gradchecked, so there
        is no backward of its own here.

        sqrt(2/pi) is computed at runtime rather than pasted as a truncated
        decimal: the literal costs precision for nothing, and this constant
        multiplies the tanh argument, where error is amplified by the
        saturation.

        This is NOT the exact erf GELU. PersonaCore runs
        F.gelu(approximate="tanh"), and the two differ enough to break a
        numeric comparison against its checkpoint.
        """
        c1 = np.sqrt(2.0 / np.pi)
        c2 = 0.044715
        cube = self * self * self
        inner = (self + cube * c2) * c1
        return self * 0.5 * (inner.tanh() + 1.0)

    def sqrt(self):
        s = np.sqrt(self.data)
        out = Tensor(s, (self,), "sqrt")

        def _backward():
            # d/dx sqrt(x) = 1/(2*sqrt(x)) — reuse the output, no second sqrt.
            # No domain guard: callers add eps (LayerNorm does sqrt(v + eps)).
            self.grad += out.grad / (2 * s)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            # no values change, only their arrangement — send the grad back
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes), (self,), "transpose")
        # UNDOING a permutation is its inverse, not the permutation again.
        # argsort(axes) is that inverse. The two coincide for involutive perms
        # — (1,0), (0,2,1,3) — which is exactly why reusing `axes` here passes
        # every swap and head-split test and only breaks on a 3-cycle.
        inv = tuple(np.argsort(axes))

        def _backward():
            self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        # subtract the max before exp: softmax is shift-invariant, and without
        # this exp(1000) overflows to inf and inf/inf is nan
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        y = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(y, (self,), "softmax")

        def _backward():
            # every output in a group depends on every input in it:
            # dL/dx = y * (g - sum_j g_j*y_j), the sum taken over `axis`
            dot = (out.grad * y).sum(axis=axis, keepdims=True)
            self.grad += y * (out.grad - dot)
        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        """log(softmax(x)), computed stably and never as log(softmax(x)).

        The shift is plain NumPy, deliberately outside the graph. For any
        per-row constant c the result is unchanged —

            (x-c) - log sum exp(x-c) = x - c - (-c + log sum exp x)

        — so d(out)/dc is exactly zero and the two routes c would take cancel
        term for term. Detaching it therefore changes no gradient, and it
        avoids putting a node in the graph whose local derivative is a
        tie-breaking convention (max is not differentiable at ties), whose two
        cancelling halves would each carry rounding noise, and which would
        allocate buffers in the hottest path of training to add zero.
        """
        shift = self.data.max(axis=axis, keepdims=True)
        z = self.data - shift
        y = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
        s = np.exp(y)  # == softmax(x), recovered from the stable log-probs
        out = Tensor(y, (self,), "log_softmax")

        def _backward():
            # dL/dx = g - softmax(x) * sum(g); no factor of y, which is why
            # cross-entropy is written on top of this rather than on softmax
            self.grad += out.grad - s * out.grad.sum(axis=axis, keepdims=True)
        out._backward = _backward
        return out

    def pick(self, idx):
        """out[i] = x[i, idx[i]] — one entry per row, chosen by a column index.

        Indices are addresses, not numbers: no gradient flows to them, so they
        stay a plain integer array rather than a Tensor.

        The backward writes to (i, idx[i]) for each row i, and those cells are
        pairwise distinct no matter what idx contains: two of them are equal
        only if both coordinates match, and the first coordinate is i itself.
        So a repeated target class — which is the common case, not the corner
        one — collides with nothing, and NumPy's buffered fancy-index += is
        enough. This is NOT the Embedding situation, where the row is chosen by
        the data and repeats do collide (hence np.add.at there).

        The += is still a += rather than an =: another consumer of x may have
        written to self.grad already, and assigning would drop it.
        """
        idx = np.asarray(idx)
        rows = np.arange(self.data.shape[0])
        out = Tensor(self.data[rows, idx], (self,), "pick")

        def _backward():
            self.grad[rows, idx] += out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.broadcast_to(out.grad, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        n = self.data.size // max(out.data.size, 1)

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                # the reduced axis is gone from g, and broadcasting aligns from
                # the right — put the axis back before spreading, or the grad
                # lands on the wrong dimension
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape) / n
        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False):
        """Biased variance (ddof=0) — the convention LayerNorm uses.

        Built from ops that already have a backward instead of a hand-written
        closure. The mean is a node every element of x feeds into AND that the
        subtraction reads back, so the graph holds a shared node; gradient
        accumulation walks both paths for us. Writing dv/dx = (2/N)(x - m) by
        hand gets the same number only because the correction term vanishes
        (see the derivation at the top of tests/test_tensor.py) — no reason to
        lean on that coincidence.
        """
        centered = self - self.mean(axis=axis, keepdims=True)
        return (centered * centered).mean(axis=axis, keepdims=keepdims)

    # ---- convenience via ops already defined above ----
    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    # ---- backward pass ----
    def backward(self):
        assert self.data.ndim == 0, \
            f"backward() starts from a scalar; got shape {self.shape} (call .sum() first)"
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()
