"""
Tests for the optimizers (M2) — SGD and Adam.

No gradcheck here: an optimizer is not a node in the graph. It reads .grad and
writes .data in place, so what has to be pinned down is arithmetic against a
hand-computed number, per-parameter state isolation, and — once — that the
whole stack actually learns something.

Adam convention (see the comment in core/optim.py): eps goes OUTSIDE the
square root, matching PyTorch:

    p -= lr * m_hat / (sqrt(v_hat) + eps)

not lr * m_hat / sqrt(v_hat + eps). test_adam_eps_is_outside_the_sqrt pins the
choice with an eps big enough that the two variants disagree measurably.
"""
import os

import numpy as np
import pytest

from core.nn import Linear
from core.optim import SGD, Adam, AdamW, clip_grad_norm_, lr_at_step
from core.tensor import Tensor


def params_with_grads():
    """Two independent parameters of different shapes, with known gradients."""
    a, b = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])), Tensor(np.array([10.0, 20.0]))
    a.grad = np.array([[0.5, -1.0], [2.0, 0.0]])
    b.grad = np.array([-3.0, 4.0])
    return a, b


def mse(pred, target):
    d = pred - target
    return (d * d).mean()


# ============================ SGD ============================

def test_sgd_step_subtracts_lr_times_grad_elementwise():
    a, b = params_with_grads()
    a0, b0, ga, gb = a.data.copy(), b.data.copy(), a.grad.copy(), b.grad.copy()

    SGD([a, b], lr=0.1).step()

    assert np.allclose(a.data, a0 - 0.1 * ga)
    assert np.allclose(b.data, b0 - 0.1 * gb)


def test_sgd_step_is_exact_not_approximate():
    p = Tensor(np.array([1.0]))
    p.grad = np.array([0.25])
    SGD([p], lr=0.5).step()
    assert p.data[0] == 1.0 - 0.5 * 0.25


def test_sgd_respects_lr():
    a, _ = params_with_grads()
    a0, ga = a.data.copy(), a.grad.copy()
    SGD([a], lr=2.0).step()
    assert np.allclose(a.data, a0 - 2.0 * ga)


def test_sgd_leaves_grad_untouched():
    a, b = params_with_grads()
    ga = a.grad.copy()
    SGD([a, b], lr=0.1).step()
    assert np.allclose(a.grad, ga)


def test_sgd_two_steps_with_different_grads():
    p = Tensor(np.array([5.0]))
    opt = SGD([p], lr=0.1)
    p.grad = np.array([1.0])
    opt.step()
    p.grad = np.array([-2.0])
    opt.step()
    assert np.allclose(p.data, 5.0 - 0.1 * 1.0 - 0.1 * -2.0)


def test_sgd_parameters_do_not_affect_each_other():
    a, b = params_with_grads()
    b0 = b.data.copy()
    a.grad = np.array([[100.0, 100.0], [100.0, 100.0]])
    SGD([a, b], lr=0.1).step()
    assert np.allclose(b.data, b0 - 0.1 * np.array([-3.0, 4.0]))


def test_sgd_zero_grad_clears_every_param():
    a, b = params_with_grads()
    opt = SGD([a, b], lr=0.1)
    opt.zero_grad()
    assert np.all(a.grad == 0) and np.all(b.grad == 0)


def test_sgd_takes_a_generator_of_params():
    lin = Linear(2, 3)
    opt = SGD(lin.parameters(), lr=0.1)
    assert len(opt.params) == 2


# ============================ Adam ============================

def adam_hand_step(p_data, m, v, g, t, lr, b1, b2, eps):
    """The reference update, written out longhand — eps outside the sqrt."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p_data - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def test_adam_state_starts_at_zero():
    a, b = params_with_grads()
    opt = Adam([a, b], lr=0.1)
    assert opt.t == 0
    assert all(np.all(x == 0) for x in opt.m) and all(np.all(x == 0) for x in opt.v)
    assert [x.shape for x in opt.m] == [(2, 2), (2,)]


def test_adam_first_step_matches_hand_computed():
    """t=1, hand-computed via the longhand reference — not 'it converges'."""
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    p = Tensor(np.array([1.0, -2.0, 0.5]))
    p.grad = np.array([0.3, -0.7, 2.0])
    expected, _, _ = adam_hand_step(p.data.copy(), 0.0, 0.0, p.grad.copy(),
                                    1, lr, b1, b2, eps)

    Adam([p], lr=lr, betas=(b1, b2), eps=eps).step()

    assert np.allclose(p.data, expected)


def test_adam_first_step_is_lr_times_sign_of_grad():
    """At t=1 the bias correction cancels: m_hat = g, sqrt(v_hat) = |g|."""
    p = Tensor(np.array([1.0, 1.0, 1.0]))
    p.grad = np.array([0.3, -0.7, 2.0])
    Adam([p], lr=0.01, eps=1e-12).step()
    assert np.allclose(p.data, 1.0 - 0.01 * np.sign([0.3, -0.7, 2.0]), atol=1e-9)


def test_adam_second_step_matches_hand_computed():
    """t=2 with a different grad — this is where bias correction stops cancelling."""
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    p = Tensor(np.array([1.0, -2.0]))
    opt = Adam([p], lr=lr, betas=(b1, b2), eps=eps)

    g1, g2 = np.array([0.3, -0.7]), np.array([-0.1, 0.9])
    d, m, v = adam_hand_step(p.data.copy(), 0.0, 0.0, g1, 1, lr, b1, b2, eps)
    d, m, v = adam_hand_step(d, m, v, g2, 2, lr, b1, b2, eps)

    p.grad = g1.copy()
    opt.step()
    p.grad = g2.copy()
    opt.step()

    assert opt.t == 2
    assert np.allclose(p.data, d)


def test_adam_eps_is_outside_the_sqrt():
    """PyTorch convention. With a big eps the two variants disagree loudly."""
    lr, eps = 0.1, 0.5
    p = Tensor(np.array([1.0]))
    p.grad = np.array([4.0])
    Adam([p], lr=lr, eps=eps).step()

    outside = 1.0 - lr * 4.0 / (np.sqrt(16.0) + eps)
    inside = 1.0 - lr * 4.0 / np.sqrt(16.0 + eps)

    assert np.allclose(p.data, outside)
    assert not np.allclose(p.data, inside)


def test_adam_respects_betas():
    """Non-default betas must actually reach the update, not be ignored."""
    lr = 0.01
    g1, g2 = np.array([0.3, -0.7]), np.array([0.9, 0.2])

    def run(betas):
        p = Tensor(np.array([1.0, -2.0]))
        opt = Adam([p], lr=lr, betas=betas)
        p.grad = g1.copy()
        opt.step()
        p.grad = g2.copy()
        opt.step()
        return p.data.copy()

    assert not np.allclose(run((0.9, 0.999)), run((0.5, 0.5)))


def test_adam_leaves_grad_untouched():
    p = Tensor(np.array([1.0, 2.0]))
    p.grad = np.array([0.3, -0.7])
    g = p.grad.copy()
    Adam([p], lr=0.01).step()
    assert np.allclose(p.grad, g)


def test_adam_zero_grad_clears_every_param():
    a, b = params_with_grads()
    Adam([a, b], lr=0.01).zero_grad()
    assert np.all(a.grad == 0) and np.all(b.grad == 0)


# ---- (3) per-parameter state, two Linears in sequence ----

def test_adam_state_is_not_shared_between_parameters():
    lin1, lin2 = Linear(1, 1), Linear(1, 1)
    opt = Adam(lin1.parameters() + lin2.parameters(), lr=0.01)

    for p, g in zip(opt.params, [1.0, 2.0, 3.0, 4.0]):
        p.grad = np.full_like(p.data, g)
    opt.step()

    # distinct arrays holding distinct values — not one buffer reused
    assert len({id(x) for x in opt.m}) == len(opt.m)
    firsts = [x.flat[0] for x in opt.m]
    assert len(set(firsts)) == len(firsts)
    for x, g in zip(opt.m, [1.0, 2.0, 3.0, 4.0]):
        assert np.allclose(x, 0.1 * g)


def test_adam_state_does_not_leak_between_optimizers():
    """Two Linears, one optimizer each, stepped a different number of times.

    lin2 is stepped once and must land exactly on the t=1 update — proof that
    lin1's three steps left nothing behind in lin2's state.
    """
    lr = 0.01
    lin1, lin2 = Linear(1, 1), Linear(1, 1)
    opt1, opt2 = Adam(lin1.parameters(), lr=lr), Adam(lin2.parameters(), lr=lr)

    w2_before = lin2.weight.data.copy()
    g = np.array([[0.5]])

    for _ in range(3):
        for p in lin1.parameters():
            p.grad = np.full_like(p.data, 0.5)
        opt1.step()

    lin2.weight.grad = g.copy()
    lin2.bias.grad = np.zeros_like(lin2.bias.data)
    opt2.step()

    expected, _, _ = adam_hand_step(w2_before, 0.0, 0.0, g, 1, lr, 0.9, 0.999, 1e-8)
    assert opt1.t == 3 and opt2.t == 1
    assert np.allclose(lin2.weight.data, expected)


# ---- (4) end-to-end: optimizer + autograd actually learn ----

def regression_data(n=64):
    """y = 3x + 2, no noise."""
    x = np.random.default_rng(0).uniform(-1.0, 1.0, (n, 1))
    return Tensor(x), Tensor(3.0 * x + 2.0)


@pytest.mark.parametrize("make_opt,lr,steps", [
    (SGD, 0.2, 500),
    (Adam, 0.1, 500),
])
def test_optimizer_learns_y_equals_3x_plus_2(make_opt, lr, steps):
    X, Y = regression_data()
    lin = Linear(1, 1, rng=np.random.default_rng(0))
    opt = make_opt(lin.parameters(), lr=lr)

    first = mse(lin(X), Y).data.item()
    for _ in range(steps):
        loss = mse(lin(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = mse(lin(X), Y).data.item()

    assert last < first * 1e-4, f"loss barely moved: {first} -> {last}"
    assert lin.weight.data.item() == pytest.approx(3.0, abs=1e-2)
    assert lin.bias.data.item() == pytest.approx(2.0, abs=1e-2)


def test_training_without_zero_grad_diverges_from_the_clean_run():
    """Guards the loop itself: stale grads accumulate and change the result."""
    X, Y = regression_data()

    def train(zero):
        lin = Linear(1, 1, rng=np.random.default_rng(0))
        opt = SGD(lin.parameters(), lr=0.05)
        for _ in range(20):
            loss = mse(lin(X), Y)
            if zero:
                opt.zero_grad()
            loss.backward()
            opt.step()
        return lin.weight.data.item()

    assert train(zero=True) != pytest.approx(train(zero=False), abs=1e-6)


# ============================ AdamW ============================
#
# AdamW = Adam + weight decay DESACOPLADO. Uma linha de diferença, mas a linha
# muda o que o decay é.
#
#   acoplado (Adam clássico com weight_decay):  g <- g + wd*p, e daí tudo segue
#   desacoplado (AdamW):                        p <- p*(1 - lr*wd), depois Adam
#
# No acoplado o decay entra no gradiente e portanto passa por m e v: fica
# normalizado pelo sqrt(v_hat), então parâmetros com gradiente historicamente
# grande são decaídos MENOS que os com gradiente pequeno — o encolhimento
# depende da escala do gradiente, que não é o que "regularização L2" deveria
# significar. E ele contamina o estado: m e v carregam o termo de decay para os
# passos seguintes.
#
# No desacoplado o encolhimento é multiplicativo e uniforme, aplicado direto em
# p, sem tocar m nem v. É por isso que a ordem importa e o teste (5) muta ela:
# aplicar o decay DEPOIS da atualização de Adam ainda encolhe, mas encolhe o
# valor já atualizado, dando um número diferente.
#
# Ordem exata do torch (torch/optim/adamw.py), replicada aqui:
#     p  <- p * (1 - lr*wd)         PRIMEIRO
#     m  <- b1*m + (1-b1)*g
#     v  <- b2*v + (1-b2)*g²
#     p  <- p - lr * m_hat/(sqrt(v_hat) + eps)
#
# O lr usado no decay é o MESMO do passo, ou seja o já escalado pelo scheduler —
# não o lr base. Num schedule com cosine decay o encolhimento diminui junto com
# o learning rate.
#
# O decay é aplicado a TODO parâmetro, sem filtrar por ndim
# ------------------------------------------------------------
# A convenção do nanoGPT é montar dois param groups e excluir do decay tudo que
# é 1-D — vieses e ganhos de LayerNorm — com o argumento de que encolher um
# ganho de normalização em direção a zero não é regularização, é sabotagem da
# própria normalização.
#
# O PersonaCore NÃO faz isso. training/loop.py:258 passa model.parameters()
# direto para o AdamW, um único param group. Conferido no checkpoint: o
# param_groups gravado tem UM grupo com os 100 parâmetros e weight_decay=0.1,
# aplicado a ln_1.bias e ln_f.weight igual a qualquer matriz.
#
# Então filtrar por ndim aqui seria "mais correto" pela literatura e ERRADO
# para o objetivo deste projeto, que é reproduzir o PersonaCore real. O teste
# (3) trava isso: um parâmetro 1-D tem que ser decaído. Se algum dia o
# PersonaCore ganhar param groups, é este teste que deve mudar primeiro.

def adamw_hand_step(p_data, m, v, g, t, lr, b1, b2, eps, wd):
    """Um passo de AdamW à mão, decay antes — a referência independente."""
    p = p_data * (1 - lr * wd)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return p, m, v


# ---- (1) invariante gratuito: wd=0 tem que ser o Adam existente ----

def test_adamw_with_zero_decay_is_bit_identical_to_adam():
    """wd=0 anula o único termo novo: p*(1 - lr*0) = p, exatamente.

    Byte a byte, não "próximo": a multiplicação por 1.0 é exata em ponto
    flutuante, então qualquer divergência aqui significa que o delta mexeu em
    algo além do decay.
    """
    a1, b1_ = params_with_grads()
    a2, b2_ = params_with_grads()
    adam, adamw = Adam([a1, b1_], lr=0.1), AdamW([a2, b2_], lr=0.1, weight_decay=0.0)

    for _ in range(3):
        adam.step()
        adamw.step()
        assert np.array_equal(a1.data, a2.data), "params divergiram com wd=0"
        assert np.array_equal(b1_.data, b2_.data)
    assert np.array_equal(adam.m[0], adamw.m[0]) and np.array_equal(adam.v[0], adamw.v[0])


def test_adamw_default_weight_decay_is_not_zero():
    """O default não pode ser 0.0 — isso faria o teste acima passar por acidente."""
    p = Tensor(np.array([1.0])); p.grad = np.array([1.0])
    assert AdamW([p]).weight_decay > 0


# ---- (2) discriminante: desacoplado != acoplado ----

def test_decoupled_decay_differs_from_coupled_decay():
    """Mesmos dados, duas convenções, números diferentes.

    O acoplado é simulado somando wd*p ao gradiente e rodando o Adam existente
    — que é literalmente o que torch.optim.Adam(weight_decay=) faz. Se os dois
    batessem, o teste (4) contra o torch não teria como discriminar nada.
    """
    lr, wd = 0.1, 0.5
    dec, cpl = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])), Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    g = np.array([[0.5, -1.0], [2.0, 0.0]])

    dec.grad = g.copy()
    AdamW([dec], lr=lr, weight_decay=wd).step()

    cpl.grad = g + wd * cpl.data          # acoplado: o decay entra pelo gradiente
    Adam([cpl], lr=lr).step()

    assert not np.allclose(dec.data, cpl.data), "as duas convenções coincidiram"


def test_coupled_decay_contaminates_the_moment_estimates():
    """O porquê da diferença: no acoplado, m e v carregam o termo de decay."""
    lr, wd = 0.1, 0.5
    p, q = Tensor(np.array([2.0])), Tensor(np.array([2.0]))
    g = np.array([1.0])

    p.grad = g.copy()
    w = AdamW([p], lr=lr, weight_decay=wd); w.step()

    q.grad = g + wd * q.data
    a = Adam([q], lr=lr); a.step()

    assert np.allclose(w.m[0], (1 - 0.9) * g), "desacoplado: m vê só o gradiente"
    assert not np.allclose(a.m[0], w.m[0]), "acoplado: m viu o decay também"


# ---- (3) o decay NÃO filtra por ndim ----

def test_decay_is_applied_to_one_dimensional_parameters_too():
    """Vieses e ganhos de LayerNorm são decaídos — a convenção do PersonaCore.

    Sem gradiente nenhum, para isolar o decay: o único movimento possível é o
    encolhimento. Se a implementação filtrasse por ndim (a convenção nanoGPT),
    este parâmetro ficaria parado e o teste cai.
    """
    bias = Tensor(np.array([1.0, -2.0, 3.0]))       # 1-D, formato de bias/LayerNorm
    bias.grad = np.zeros(3)
    lr, wd = 0.1, 0.5

    AdamW([bias], lr=lr, weight_decay=wd).step()

    assert np.allclose(bias.data, np.array([1.0, -2.0, 3.0]) * (1 - lr * wd))


def test_one_dimensional_and_two_dimensional_decay_by_the_same_factor():
    """Mesmo fator para as duas — nenhum tratamento especial por forma."""
    v, m = Tensor(np.array([2.0, 4.0])), Tensor(np.array([[2.0, 4.0]]))
    v.grad, m.grad = np.zeros(2), np.zeros((1, 2))
    lr, wd = 0.1, 0.5

    AdamW([v, m], lr=lr, weight_decay=wd).step()

    assert np.allclose(v.data, m.data[0])
    assert np.allclose(v.data / np.array([2.0, 4.0]), 1 - lr * wd)


def test_adamw_matches_the_hand_computed_step():
    """Contra a fórmula escrita à mão, não contra si mesmo."""
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    p.grad = np.array([[0.5, -1.0], [2.0, 0.0]])
    lr, wd, b1, b2, eps = 0.1, 0.3, 0.9, 0.999, 1e-8

    opt = AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    expected, m, v = adamw_hand_step(p.data.copy(), 0, 0, p.grad, 1, lr, b1, b2, eps, wd)
    opt.step()

    assert np.allclose(p.data, expected, atol=1e-15)
    assert np.allclose(opt.m[0], m) and np.allclose(opt.v[0], v)


# ---- (4) contra o estado real do checkpoint ----
#
# Um passo de AdamW a partir do estado de treino REAL: exp_avg / exp_avg_sq /
# step do best.pt no passo 49000, e os gradientes já congelados e validados em
# test_parity.py. Os hiperparâmetros são os gravados no param_group, não
# escolhidos: lr = 3.0267460081032337e-05 (o valor que o schedule produziu em
# 49000), betas=(0.9,0.999), eps=1e-8, wd=0.1.
#
# O que este teste prova e o que NÃO prova. Os gradientes vêm de UMA janela de
# 256 tokens em float64, não do micro-batch de 32 janelas que o passo 49000 de
# verdade usou. Então isto não reproduz aquele passo histórico — é estado real
# + gradientes reais combinados para exercitar a FÓRMULA. É o alvo aprovado:
# o passo, não a trajetória. Reproduzir a trajetória é impossível por
# princípio, não por bug: o treino foi fp32 no MPS, este engine é float64, a
# diferença no primeiro passo é ~1e-7 relativo, e o treino amplifica isso
# exponencialmente.
#
# Tolerância derivada. Os dois lados fazem a MESMA sequência de operações em
# float64 sobre os MESMOS bits de entrada: multiplicação, duas atualizações de
# momento, uma raiz, uma divisão. São ~8 operações elementares, sem contração
# nenhuma (o optimizer é ponto a ponto — não há soma de K termos, que é o que
# gerava o sqrt(K)·eps da paridade). O erro é então de poucos ulps: ~8·eps ≈
# 1.8e-15 relativo.
#
# Uma diferença conhecida de formulação empurra isso um pouco: o torch escreve
# a atualização do momento como lerp_(grad, 1-b1), ou seja m + (1-b1)*(g - m),
# enquanto a forma escrita aqui é b1*m + (1-b1)*g. São algebricamente iguais e
# diferem nos últimos bits. O torch também agrupa a correção de viés como
# sqrt(v)/sqrt(bc2) + eps em vez de sqrt(v/bc2) + eps — de novo, mesma álgebra,
# arredondamento diferente.
#
# rtol=1e-13 então: o limite de ~2e-15 com margem para essas reassociações.
# O erro medido é impresso e asserido contra um limite apertado à parte.

ADAMW_FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                             "fixtures", "personacore_adamw.npz")

ADAMW_KEYS = ["blocks.0.attn.q_proj.weight", "blocks.0.ln_1.bias",
              "blocks.5.mlp.fc_out.weight", "ln_f.weight"]
ADAMW_RTOL = 1e-13


@pytest.fixture(scope="module")
def adamw_ref():
    if not os.path.exists(ADAMW_FIXTURE):
        pytest.skip("fixtures/personacore_adamw.npz ausente")
    return np.load(ADAMW_FIXTURE)


def _stepped(ref, key):
    """Roda um passo do AdamW daqui a partir do estado real gravado."""
    p = Tensor(ref[f"p0::{key}"])
    p.grad = ref[f"g::{key}"].copy()
    opt = AdamW([p], lr=float(ref["lr"]), betas=tuple(ref["betas"]),
                eps=float(ref["eps"]), weight_decay=float(ref["wd"]))
    opt.m[0] = ref[f"m0::{key}"].copy()
    opt.v[0] = ref[f"v0::{key}"].copy()
    opt.t = int(ref[f"t0::{key}"])
    opt.step()
    return p, opt


@pytest.mark.parametrize("key", ADAMW_KEYS)
def test_one_step_matches_torch_adamw_from_real_state(adamw_ref, key):
    p, opt = _stepped(adamw_ref, key)

    for name, got, expected in [("param", p.data, adamw_ref[f"p1::{key}"]),
                                ("exp_avg", opt.m[0], adamw_ref[f"m1::{key}"]),
                                ("exp_avg_sq", opt.v[0], adamw_ref[f"v1::{key}"])]:
        err = np.abs(got - expected).max() / np.abs(expected).max()
        print(f"[adamw] {key} {name}: {err:.3e}")
        assert err < ADAMW_RTOL, f"{key}/{name}: {err:.3e}"


def test_the_real_step_actually_moves_the_parameters(adamw_ref):
    """O passo não é um no-op: sem isto, um step() vazio passaria em tudo acima."""
    for key in ADAMW_KEYS:
        moved = np.abs(adamw_ref[f"p1::{key}"] - adamw_ref[f"p0::{key}"]).max()
        assert moved > 1e-6, f"{key}: referência não se moveu ({moved:.3e})"


def test_the_fixture_covers_both_one_and_two_dimensional_parameters(adamw_ref):
    """A amostra tem que incluir 1-D, senão o teste (3) não é exercitado no real."""
    ndims = {adamw_ref[f"p0::{k}"].ndim for k in ADAMW_KEYS}
    assert ndims == {1, 2}


def test_coupled_decay_does_not_match_torch_on_real_state(adamw_ref):
    """O discriminante no dado real: a convenção acoplada erra por muito.

    A referência acoplada foi gerada com o mesmo estado e os mesmos números,
    só mudando onde o decay entra. Se a implementação daqui fosse acoplada, o
    teste acima falharia — e a margem não é numérica, é grosseira.
    """
    key = ADAMW_KEYS[0]
    p, _ = _stepped(adamw_ref, key)
    coupled = adamw_ref[f"coupled_p1::{key}"]

    err = np.abs(p.data - coupled).max() / np.abs(coupled).max()
    print(f"[adamw] desacoplado vs acoplado no estado real: {err:.3e}")
    assert err > 1e-6, "as duas convenções ficaram indistinguíveis"


# ============================ LR schedule ============================
#
# Warmup linear -> cosine decay -> piso, a fórmula de schedule.py:30-37 do
# PersonaCore:
#
#     step < warmup:  mult = (step + 1) / warmup
#     step >= max:    mult = min_ratio
#     senão:          progress = (step - warmup) / (max - warmup)
#                     mult = min_ratio + (1 - min_ratio)·½·(1 + cos(π·progress))
#
#     lr = base_lr · mult
#
# Função pura, sem classe e sem state_dict. O PersonaCore embrulha isso num
# LambdaLR, mas por um motivo que não existe aqui: checkpoint.py chama
# scheduler.state_dict() no save e load_state_dict() no resume, e o LambdaLR
# serializa só o contador last_epoch — a lambda em si NÃO é picklada, então o
# harness tem que reconstruir build_scheduler(...) identicamente antes de
# carregar. Ou seja o wrapper existe para serializar um inteiro.
#
# Aqui o inteiro é o argumento. Não há estado a serializar, nem ordem de
# reconstrução a acertar no resume, nem a classe de bug em que o schedule
# reconstruído difere do que salvou. lr_at_step(step, ...) é referencialmente
# transparente: mesmo step, mesmo lr, sempre.
#
# Três propriedades da fórmula que os testes fixam por cálculo, não por fé:
#
# 1. O warmup usa (step+1)/warmup, não step/warmup. Consequência: o passo 0
#    NÃO sai com lr zero (sairia com 0.01·base_lr aqui), e o multiplicador
#    chega a 1.0 já no passo warmup-1, não em warmup.
#
# 2. A transição warmup->cosine é contínua, e isso é uma coincidência
#    aritmética das duas expressões, não algo imposto: em step=warmup-1 o ramo
#    linear dá (warmup)/warmup = 1.0, e em step=warmup o ramo cosine dá
#    progress=0 -> cos(0)=1 -> min_ratio + (1-min_ratio)·1 = 1.0. Dois ramos
#    diferentes, mesmo valor. Por isso os dois lados são testados SEPARADAMENTE:
#    um teste só de um lado não distingue "contínuo" de "descontínuo com sorte".
#
# 3. O piso é min_ratio·base_lr, não zero, e é plano depois de max_steps — não
#    continua caindo (o cosine sozinho passaria de 1 e voltaria a SUBIR) nem
#    dispara.
#
#    Mas o piso NÃO é um mínimo global, e eu escrevi isso errado antes do teste
#    refutar: o warmup começa em 0.01·base_lr, seis vezes ABAIXO do piso, e sobe
#    atravessando ele. min_ratio limita só o ramo cosine. Quem confundir os dois
#    "conserta" o warmup com um clamp e mata o ramp.
#
# O lr que sai daqui alimenta os DOIS termos do AdamW: a atualização e o decay
# desacoplado (p *= 1 - lr·wd). Um erro de schedule não muda só a velocidade do
# passo, muda também quanto os pesos encolhem.

SCHED = dict(warmup_steps=100, max_steps=50000, min_ratio=0.1)
BASE_LR = 3e-4

# Do param_groups gravado em checkpoints/best.pt, last_epoch=49000. Não é um
# número que eu escolhi: é o que o schedule do PersonaCore produziu na corrida
# real, lido do checkpoint.
LR_AT_49000 = 3.0267460081032337e-05


# ---- (1) o valor real do checkpoint ----

def test_schedule_reproduces_the_checkpoints_recorded_lr():
    """A fórmula reimplementada bate com o lr gravado no passo 49000."""
    got = lr_at_step(49000, BASE_LR, **SCHED)
    assert got == pytest.approx(LR_AT_49000, rel=1e-15, abs=0.0)


def test_the_checkpoint_lr_is_not_reproduced_one_step_off():
    """Fixa o off-by-one: 48999 dá outro número, então o alinhamento é real."""
    assert lr_at_step(48999, BASE_LR, **SCHED) != pytest.approx(LR_AT_49000, rel=1e-12)


# ---- (2) o passo 0 não é zero ----

def test_step_zero_is_one_percent_of_base_lr_not_zero():
    """(0+1)/100 = 0.01. Com step/warmup seria 0.0 e o primeiro passo seria nulo."""
    assert lr_at_step(0, BASE_LR, **SCHED) == pytest.approx(0.01 * BASE_LR, rel=1e-15)
    assert lr_at_step(0, BASE_LR, **SCHED) > 0.0


def test_warmup_ramps_linearly():
    """Passos igualmente espaçados no warmup sobem por incrementos iguais."""
    lrs = [lr_at_step(s, BASE_LR, **SCHED) for s in (0, 10, 20, 30)]
    deltas = np.diff(lrs)
    assert np.allclose(deltas, deltas[0], rtol=1e-15)


# ---- (3) fronteira warmup -> cosine, os dois lados separados ----

def test_last_warmup_step_evaluates_to_multiplier_one():
    """Ramo LINEAR: (99+1)/100 = 1.0 exato."""
    assert lr_at_step(99, BASE_LR, **SCHED) == pytest.approx(BASE_LR, rel=1e-15)


def test_first_cosine_step_evaluates_to_multiplier_one():
    """Ramo COSINE: progress=0 -> min_ratio + (1-min_ratio)·1 = 1.0.

    Teste separado do anterior de propósito. Os dois avaliam para o mesmo
    valor por ramos diferentes da fórmula; testar só um lado provaria que
    aquele lado está certo, não que a transição é contínua.
    """
    assert lr_at_step(100, BASE_LR, **SCHED) == pytest.approx(BASE_LR, rel=1e-15)


def test_the_warmup_boundary_is_continuous():
    """E, juntos: os dois lados coincidem — a transição não tem degrau."""
    assert lr_at_step(99, BASE_LR, **SCHED) == pytest.approx(
        lr_at_step(100, BASE_LR, **SCHED), rel=1e-15)


def test_the_cosine_branch_starts_decaying_immediately_after_the_boundary():
    """Em 101 já caiu — senão o 'contínuo' acima poderia ser um platô achatado."""
    assert lr_at_step(101, BASE_LR, **SCHED) < lr_at_step(100, BASE_LR, **SCHED)


# ---- (4) fronteira cosine -> piso ----

FLOOR = 0.1 * BASE_LR


def test_step_just_before_max_is_near_the_floor_but_not_on_it():
    """max-1 ainda está no ramo cosine: acima do piso, mas por muito pouco."""
    got = lr_at_step(49999, BASE_LR, **SCHED)
    assert got > FLOOR
    assert got == pytest.approx(FLOOR, rel=1e-7)


def test_step_at_max_is_exactly_the_floor():
    """max entra no ramo do piso — igualdade exata, não aproximação."""
    assert lr_at_step(50000, BASE_LR, **SCHED) == FLOOR


def test_the_floor_is_flat_after_max():
    """Depois de max fica plano: não continua caindo nem volta a subir.

    O ramo cosine sozinho, com progress > 1, voltaria a SUBIR (cos passa do
    mínimo e cresce). É esse o bug que o piso previne, não só um lr pequeno.
    """
    beyond = [lr_at_step(s, BASE_LR, **SCHED) for s in (50000, 50001, 60000, 200000)]
    assert all(x == FLOOR for x in beyond)


def test_the_floor_binds_only_after_warmup_not_during_it():
    """O piso NÃO é um mínimo global — o warmup passa por baixo dele.

    Escrevi primeiro que o multiplicador vive em [min_ratio, 1] e o teste
    refutou: no passo 0 o lr é 0.01·base_lr = 3e-6, seis vezes ABAIXO do piso
    de 3e-5. O warmup começa embaixo e sobe atravessando o piso; min_ratio só
    limita o ramo cosine. Confundir os dois faria alguém "consertar" o warmup
    com um clamp e destruir o ramp.
    """
    assert lr_at_step(0, BASE_LR, **SCHED) < FLOOR
    crossing = [s for s in range(100) if lr_at_step(s, BASE_LR, **SCHED) >= FLOOR][0]
    assert 0 < crossing < 100, "o warmup cruza o piso em algum ponto do ramp"

    after = [lr_at_step(s, BASE_LR, **SCHED) for s in range(100, 60000, 137)]
    assert min(after) >= FLOOR - 1e-18, "do fim do warmup em diante, o piso segura"
    assert max(after) <= BASE_LR + 1e-18


def test_the_cosine_branch_is_monotonically_decreasing():
    """Sem oscilação entre a fronteira do warmup e max."""
    lrs = [lr_at_step(s, BASE_LR, **SCHED) for s in range(100, 50000, 331)]
    assert all(b < a for a, b in zip(lrs, lrs[1:]))


def test_midpoint_of_the_cosine_is_the_average_of_the_endpoints():
    """Em progress=0.5, cos(π/2)=0 -> mult = min_ratio + (1-min_ratio)/2.

    Um ponto interior calculado à mão, não só as fronteiras.
    """
    mid = 100 + (50000 - 100) // 2
    expected = (0.1 + 0.9 * 0.5) * BASE_LR
    assert lr_at_step(mid, BASE_LR, **SCHED) == pytest.approx(expected, rel=1e-6)


# ============================ clip_grad_norm_ ============================
#
# Fonte lida em torch/nn/utils/clip_grad.py do venv do PersonaCore (torch
# 2.7.1), não de memória. clip_grad_norm_ é _get_total_norm seguido de
# _clip_grads_with_norm_:
#
#     total_norm = ||[ ||g_1||, ||g_2||, ..., ||g_N|| ]||        (duas etapas)
#     clip_coef  = max_norm / (total_norm + 1e-6)
#     coef       = min(clip_coef, 1.0)
#     g_i       *= coef        para TODO i, com o MESMO coef
#     return total_norm                                          (PRÉ-clip)
#
# Quatro coisas que só se sabem lendo a fonte, e cada uma vira teste:
#
# 1. GLOBAL, não por tensor. O coeficiente sai da norma conjunta e multiplica
#    todo mundo igual. Clipar cada tensor pela sua própria norma é outra
#    operação: preserva a direção de cada tensor isoladamente mas muda a
#    direção do gradiente CONJUNTO, que é o vetor que o passo de otimização
#    anda. O teste (1) constrói o caso onde as duas divergem: nenhuma norma
#    individual passa do limite, mas a conjunta passa — clipar por tensor seria
#    um no-op ali.
#
# 2. A norma é calculada em DUAS ETAPAS (norma de cada tensor, depois norma do
#    vetor de normas), não achatando tudo num vetor só. Algebricamente idêntico
#    — sqrt(soma dos quadrados das normas) = sqrt(soma de todos os quadrados) —
#    mas a ordem de soma difere, e é a forma de duas etapas que é replicada
#    aqui para o item (3) bater apertado.
#
# 3. eps = 1e-6 no DENOMINADOR, somado à norma. Não é configurável, é literal.
#    Consequência que decide o item (4) por cálculo e não por gosto: com a
#    norma EXATAMENTE igual a max_norm, coef = max_norm/(max_norm + 1e-6) < 1,
#    então CLIPA — por um fator de ~1e-6, mas clipa. A fronteira real entre
#    "intocado" e "escalado" está em max_norm - 1e-6, não em max_norm.
#
# 4. O clamp em 1.0 existe, e o torch multiplica MESMO ASSIM quando o coef foi
#    clampado. O comentário na fonte diz o porquê: evitar um `if coef < 1:`,
#    que forçaria sincronização CPU<->GPU. Como multiplicar por 1.0 é exato em
#    ponto flutuante, os grads saem bit a bit idênticos — é por isso que o
#    teste (2) pode exigir array_equal e não allclose.
#
# O retorno é a norma PRÉ-clip. O loop.py do PersonaCore descarta, mas é o
# número que um log de treino quer (ver se o clipping está mordendo ou não), e
# replicar sai de graça.

def grads_of(*arrays):
    """Tensores descartáveis carregando os gradientes dados."""
    ps = []
    for a in arrays:
        t = Tensor(np.zeros_like(a))
        t.grad = a.copy()
        ps.append(t)
    return ps


def two_stage_norm(*arrays):
    """A referência independente: norma das normas."""
    return np.linalg.norm([np.linalg.norm(a) for a in arrays])


# ---- (1) o caso que só a norma GLOBAL pega ----

def test_all_parameters_are_scaled_by_one_globally_derived_coefficient():
    """Cinco tensores de formas diferentes, cada um bem abaixo do limite.

    Normas individuais ~0.6 contra max_norm=1.0 — clipar por tensor não tocaria
    em nada. A conjunta é sqrt(5·0.36) ≈ 1.34 e passa. Confirma que o fator
    aplicado é o MESMO em todos e vem da norma conjunta.
    """
    shapes = [(4,), (2, 3), (5,), (3, 3), (2, 2, 2)]
    arrays = [np.full(s, 0.6 / np.sqrt(np.prod(s))) for s in shapes]
    for a in arrays:
        assert np.linalg.norm(a) < 1.0, "cada um sozinho está abaixo do limite"

    ps = grads_of(*arrays)
    before = [p.grad.copy() for p in ps]
    total = clip_grad_norm_(ps, 1.0)

    assert total > 1.0, "a norma conjunta passa do limite"
    expected_coef = 1.0 / (total + 1e-6)
    ratios = [(p.grad / b)[np.abs(b) > 0] for p, b in zip(ps, before)]
    for r in ratios:
        assert np.allclose(r, expected_coef, rtol=1e-15)
    assert np.allclose(np.concatenate([r.ravel() for r in ratios]), ratios[0].ravel()[0])


def test_per_tensor_clipping_would_be_a_noop_on_that_case():
    """Fixa a premissa: por tensor, nada mudaria — então (1) discrimina mesmo."""
    shapes = [(4,), (2, 3), (5,), (3, 3), (2, 2, 2)]
    for s in shapes:
        a = np.full(s, 0.6 / np.sqrt(np.prod(s)))
        assert min(1.0, 1.0 / (np.linalg.norm(a) + 1e-6)) == 1.0


def test_the_clipped_global_norm_lands_at_max_norm():
    """Depois do clip a norma conjunta é max_norm (a menos do eps)."""
    arrays = [np.array([3.0, 4.0]), np.array([[6.0, 8.0]])]
    ps = grads_of(*arrays)
    clip_grad_norm_(ps, 1.0)
    assert two_stage_norm(*[p.grad for p in ps]) == pytest.approx(1.0, rel=1e-5)


# ---- (2) abaixo do limite: intocado, exatamente ----

def test_gradients_below_the_limit_come_out_bit_identical():
    """array_equal, não allclose. O torch multiplica por 1.0 mesmo assim, e
    multiplicar por 1.0 é exato — então 'não mexeu' é literal."""
    arrays = [np.array([0.1, -0.2]), np.array([[0.3, 0.05], [-0.01, 0.2]])]
    ps = grads_of(*arrays)
    before = [p.grad.copy() for p in ps]

    total = clip_grad_norm_(ps, 100.0)

    assert total < 100.0
    for p, b in zip(ps, before):
        assert np.array_equal(p.grad, b), "gradiente abaixo do limite foi alterado"


def test_a_small_gradient_is_never_amplified():
    """Sem o clamp, coef = max_norm/(norma+eps) >> 1 e o grad seria INFLADO."""
    tiny = np.array([1e-8, -1e-8])
    p = grads_of(tiny)[0]
    clip_grad_norm_([p], 1.0)
    assert np.array_equal(p.grad, tiny)


def test_zero_gradients_stay_zero_and_do_not_produce_nan():
    """Norma zero: o eps no denominador é o que evita 0/0."""
    ps = grads_of(np.zeros(3), np.zeros((2, 2)))
    total = clip_grad_norm_(ps, 1.0)
    assert total == 0.0
    assert all(np.array_equal(p.grad, np.zeros_like(p.grad)) for p in ps)


# ---- (3) contra o clip_grad_norm_ real, em gradientes reais ----
#
# Tolerância derivada. Os dois lados fazem a mesma sequência sobre os mesmos
# bits: N normas L2 (contrações de comprimento até 590k), uma norma dessas N,
# uma divisão, uma multiplicação por tensor. A única contração longa é a norma
# de cada tensor — soma em pares dá ~sqrt(K)·eps, e com K = 384·1536 = 589824,
# sqrt(K) ≈ 768, logo ~1.7e-13 relativo na norma. O coeficiente herda isso, e a
# multiplicação final não acrescenta mais que 1 ulp. rtol=1e-12, o limite com
# ~6x de margem. O medido é impresso.

CLIP_FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                            "fixtures", "personacore_clip.npz")
CLIP_RTOL = 1e-12


@pytest.fixture(scope="module")
def clip_ref():
    if not os.path.exists(CLIP_FIXTURE):
        pytest.skip("fixtures/personacore_clip.npz ausente")
    return np.load(CLIP_FIXTURE)


@pytest.mark.parametrize("tag", ["clips", "noop", "tight"])
def test_matches_torch_clip_grad_norm_on_real_gradients(clip_ref, tag):
    keys = list(clip_ref["keys"])
    ps = grads_of(*[clip_ref[f"g::{k}"] for k in keys])
    max_norm = float(clip_ref[f"maxnorm::{tag}"])

    total = clip_grad_norm_(ps, max_norm)

    ref_total = float(clip_ref[f"totalnorm::{tag}"])
    err_n = abs(total - ref_total) / ref_total
    print(f"\n[clip:{tag}] max_norm={max_norm} norma total: erro {err_n:.3e}")
    assert err_n < CLIP_RTOL

    for k, p in zip(keys, ps):
        expected = clip_ref[f"clipped::{tag}::{k}"]
        err = np.abs(p.grad - expected).max() / np.abs(expected).max()
        print(f"[clip:{tag}] {k}: {err:.3e}")
        assert err < CLIP_RTOL


def test_the_real_fixture_exercises_both_branches(clip_ref):
    """Sem isto, três casos que por acaso não clipassem passariam por 'validado'."""
    keys = list(clip_ref["keys"])
    changed = {}
    for tag in ("clips", "noop", "tight"):
        changed[tag] = any(
            not np.array_equal(clip_ref[f"clipped::{tag}::{k}"], clip_ref[f"g::{k}"])
            for k in keys)
    assert changed["clips"] and changed["tight"], "casos de corte não cortaram"
    assert not changed["noop"], "o caso no-op mexeu nos gradientes"


# ---- (4) fronteira: norma EXATAMENTE igual a max_norm ----

def test_norm_exactly_at_the_limit_still_clips_because_of_the_eps():
    """Construído, não sorteado: [3,4] tem norma 5 exata, max_norm=5.0.

    coef = 5/(5 + 1e-6) < 1, então CLIPA. Decidido pela fórmula lida na fonte,
    não por preferência: com eps no denominador não existe ponto em que a norma
    esteja no limite e o gradiente saia intocado.
    """
    g = np.array([3.0, 4.0])
    p = grads_of(g)[0]

    total = clip_grad_norm_([p], 5.0)

    assert total == 5.0, "norma exata, sem erro de arredondamento"
    assert not np.array_equal(p.grad, g), "o eps faz clipar mesmo no limite"
    assert np.allclose(p.grad, g * (5.0 / (5.0 + 1e-6)), rtol=1e-15)


def test_the_boundary_case_matches_torch_exactly(clip_ref):
    """O mesmo caso, contra o número que o torch produziu."""
    p = grads_of(clip_ref["boundary_grad_in"])[0]
    total = clip_grad_norm_([p], 5.0)
    assert total == pytest.approx(float(clip_ref["boundary_total"]), rel=1e-15)
    assert np.allclose(p.grad, clip_ref["boundary_grad_out"], rtol=1e-15)


def test_the_untouched_boundary_is_max_norm_minus_eps_not_max_norm():
    """Onde o clamp realmente morde: norma + 1e-6 == max_norm.

    Logo abaixo disso o coeficiente passa de 1 e é clampado (intocado); logo
    acima, clipa. A fronteira do comportamento não é max_norm.
    """
    g = np.array([3.0, 4.0])                     # norma 5
    below = grads_of(g)[0]
    clip_grad_norm_([below], 5.0 + 2e-6)         # coef > 1 -> clampado
    assert np.array_equal(below.grad, g)

    above = grads_of(g)[0]
    clip_grad_norm_([above], 5.0 + 0.5e-6)       # coef < 1 -> clipa
    assert not np.array_equal(above.grad, g)


# ---- retorno ----

def test_returns_the_pre_clip_norm_not_the_post_clip_one():
    """O torch devolve a norma ANTES do corte — o número que diz se mordeu."""
    ps = grads_of(np.array([3.0, 4.0]), np.array([0.0]))
    total = clip_grad_norm_(ps, 1.0)
    assert total == pytest.approx(5.0, rel=1e-15), "pré-clip"
    assert two_stage_norm(*[p.grad for p in ps]) == pytest.approx(1.0, rel=1e-5)
