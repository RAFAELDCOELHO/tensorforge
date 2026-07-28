"""
Gradcheck tests for the scalar autograd engine (M0).

Numerical gradient (central difference) vs analytical gradient (.backward()).
Every op must pass both: an isolated single-op check and a chained-graph check.
"""
import math
import pytest
from core.engine import Value

EPS = 1e-6
TOL = 1e-4


def numerical_grad(f, x0):
    """Central difference numerical gradient of f (float -> float) at x0."""
    return (f(x0 + EPS) - f(x0 - EPS)) / (2 * EPS)


def analytical_grad(build_graph, x0):
    """Build a Value graph from x0, run backward, return d(out)/d(x)."""
    x = Value(x0)
    out = build_graph(x)
    out.backward()
    return x.grad


@pytest.mark.parametrize("x0", [-2.5, -0.5, 0.3, 1.7, 3.0])
def test_add_gradient(x0):
    f = lambda x: (x + 3.0) + x  # d/dx = 2
    build = lambda x: (x + Value(3.0)) + x
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [-2.5, -0.5, 0.3, 1.7, 3.0])
def test_mul_gradient(x0):
    f = lambda x: x * 2.0 * x  # d/dx = 4x
    build = lambda x: x * Value(2.0) * x
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [-2.5, -0.5, 0.3, 1.7, 3.0])
def test_pow_gradient(x0):
    f = lambda x: x ** 3
    build = lambda x: x ** 3
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 4.5])
def test_relu_gradient(x0):
    f = lambda x: max(x, 0.0) * 3.0
    build = lambda x: x.relu() * Value(3.0)
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [-1.5, -0.2, 0.4, 1.1, 2.3])
def test_tanh_gradient(x0):
    f = lambda x: math.tanh(x)
    build = lambda x: x.tanh()
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [-1.5, -0.2, 0.4, 1.1, 2.3])
def test_exp_gradient(x0):
    f = lambda x: math.exp(x)
    build = lambda x: x.exp()
    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


@pytest.mark.parametrize("x0", [-1.5, -0.2, 0.4, 1.1, 2.3])
def test_chained_graph_gradient(x0):
    """A composite function exercising every op together, like a real forward pass."""
    def f(x):
        return math.tanh((x * 2.0 - 1.0) ** 2 + math.exp(x) - max(x, 0.0))

    def build(x):
        return (((x * Value(2.0)) - Value(1.0)) ** 2 + x.exp() - x.relu()).tanh()

    assert analytical_grad(build, x0) == pytest.approx(numerical_grad(f, x0), abs=TOL)


def test_backward_accumulates_grad_when_variable_used_twice():
    """Classic autograd bug: x used twice must SUM gradients, not overwrite."""
    x = Value(3.0)
    y = x + x  # dy/dx = 2, not 1 (would happen if backward overwrites instead of +=)
    y.backward()
    assert x.grad == pytest.approx(2.0)


# ---- determinismo do backward ----
#
# _prev era um set. Iterar um set de objetos segue hashes derivados de id(),
# que mudam entre processos, então a ordem topológica reversa — e com ela a
# ordem em que os += caem no .grad — mudava a cada execução; soma de ponto
# flutuante não é associativa, então execuções idênticas divergiam nos últimos
# bits. Achado pelo teste byte-a-byte de tests/test_train.py no Tensor do M1;
# o mesmo defeito estava aqui, e o mesmo conserto (tupla) vale.

def test_the_same_graph_gives_bit_identical_gradients_every_time():
    """Igualdade exata, não approx: é o último bit que estava se mexendo."""
    def run():
        x, y = Value(1.7), Value(-0.4)
        z = ((x * y + x).tanh() + (x + y).exp() * x).relu() + (x * x + y * y)
        z.backward()
        return x.grad, y.grad

    first = run()
    assert all(run() == first for _ in range(5))


def test_the_child_list_is_ordered_not_a_set():
    """A causa raiz, fixada diretamente."""
    a, b = Value(1.0), Value(2.0)
    assert not isinstance((a + b)._prev, set)
    assert list((a + b)._prev) == [a, b]


def test_a_tuple_of_children_still_accumulates_a_repeated_node():
    """A tupla admite a duplicata que o set colapsava — x + x continua dando 2."""
    x = Value(3.0)
    assert list((x + x)._prev) == [x, x]
    (x + x).backward()
    assert x.grad == pytest.approx(2.0)
