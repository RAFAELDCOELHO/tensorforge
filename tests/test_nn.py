"""
Gradcheck tests for the nn modules (M2) — Linear only for now.

Reuses the M1 gradcheck harness: numerical (central difference) vs analytical.
Linear adds nothing to the math — it is x @ W + b, both already gradchecked in
test_tensor.py — so what these tests actually pin down is the *module*: that
the parameters it owns are the tensors the gradient lands on, that they keep
their shape under a batched input (dW summed over the batch axis, NOT left at
(B, in, out)), and that parameters()/zero_grad() see them.


LayerNorm — que op existente cada linha do forward usa
======================================================

Nenhuma op nova, nenhuma derivada nova. Cada linha abaixo já é coberta por
gradcheck próprio em tests/test_tensor.py; o que sobra pra testar aqui é a
composição e os parâmetros.

    m = x.mean(axis=-1, keepdims=True)
        └─ Tensor.mean          (op própria, backward = broadcast(G)/N)

    v = x.var(axis=-1, keepdims=True)
        └─ Tensor.var           (composta: __sub__, __mul__, Tensor.mean)

    xhat = (x - m) / (v + eps).sqrt()
        ├─ __sub__              (= __add__ + __neg__, __neg__ = __mul__ por -1.0)
        │                        m é (…,1) contra x (…,N) → _unbroadcast no lado de m
        ├─ __add__              (v + eps: float vira Tensor 0-d, broadcast)
        ├─ Tensor.sqrt          (backward = G/(2s); o eps é o que impede 1/0 aqui)
        └─ __truediv__          (denominador (…,1) contra numerador (…,N)
                                 → _unbroadcast no lado do denominador)

    out = xhat * gamma + beta
        ├─ __mul__              (gamma é (N,) contra xhat (…,N) → dgamma soma
        │                        sobre TODOS os eixos de batch via _unbroadcast)
        └─ __add__              (beta idem — mesmo caminho que o bias do Linear)

keepdims=True em mean e var é o que faz os shapes alinharem contra o x
original: (…,1) broadcasta contra (…,N) pela direita. Com keepdims=False o
eixo some, o broadcast alinha errado e o gradiente cai no eixo errado.
"""
import numpy as np
import pytest
from core.nn import (GPT, MLP, Block, CausalSelfAttention, Embedding, LayerNorm,
                     Linear, Module, cross_entropy, scaled_dot_product_attention)
from core.tensor import Tensor
from tests.test_tensor import (gelu_np, gradcheck, log_softmax_np, rand,
                               softmax_np)


def linear_build(in_f, out_f, bias=True):
    """Return a build fn that stuffs the given W/b Tensors into a fresh Linear."""
    def build(x, W, b=None):
        lin = Linear(in_f, out_f, bias=bias)
        lin.weight = W
        if bias:
            lin.bias = b
        return lin(x).sum()
    return build


# ---- forward ----

def test_linear_forward_matches_manual():
    x, W, b = rand(4, 3), rand(3, 5), rand(5)
    lin = Linear(3, 5)
    lin.weight, lin.bias = Tensor(W), Tensor(b)
    assert np.allclose(lin(Tensor(x)).data, x @ W + b)


def test_linear_init_shapes():
    lin = Linear(3, 5)
    assert lin.weight.shape == (3, 5)
    assert lin.bias.shape == (5,)


def test_linear_init_is_reproducible_with_seed():
    a = Linear(3, 5, rng=np.random.default_rng(0))
    b = Linear(3, 5, rng=np.random.default_rng(0))
    assert np.allclose(a.weight.data, b.weight.data)


def test_linear_init_differs_without_shared_seed():
    a = Linear(3, 5, rng=np.random.default_rng(0))
    b = Linear(3, 5, rng=np.random.default_rng(1))
    assert not np.allclose(a.weight.data, b.weight.data)


# ---- gradcheck ----

def test_linear_gradcheck_2d():
    x, W, b = rand(4, 3), rand(3, 5), rand(5)
    gradcheck(lambda x, W, b: (x @ W + b).sum(), linear_build(3, 5), [x, W, b])


def test_linear_gradcheck_batched():
    """(B,N,in) input — dW must be summed back to (in,out), not left as (B,in,out)."""
    x, W, b = rand(6, 4, 3), rand(3, 5), rand(5)
    gradcheck(lambda x, W, b: (x @ W + b).sum(), linear_build(3, 5), [x, W, b])


def test_linear_batched_dW_has_parameter_shape():
    """The assertion above lives inside gradcheck; state it directly too."""
    lin = Linear(3, 5)
    lin.weight, lin.bias = Tensor(rand(3, 5)), Tensor(rand(5))
    lin(Tensor(rand(6, 4, 3))).sum().backward()
    assert lin.weight.grad.shape == (3, 5)
    assert lin.bias.grad.shape == (5,)


def test_linear_batched_dW_equals_sum_over_batch():
    """dW for a batched input is exactly the sum of the per-sample dWs."""
    W, b = Tensor(rand(3, 5)), Tensor(rand(5))
    x = rand(6, 4, 3)

    lin = Linear(3, 5)
    lin.weight, lin.bias = W, b
    lin(Tensor(x)).sum().backward()

    per_sample = np.zeros((3, 5))
    for i in range(x.shape[0]):
        W_i, b_i = Tensor(W.data), Tensor(b.data)
        lin_i = Linear(3, 5)
        lin_i.weight, lin_i.bias = W_i, b_i
        lin_i(Tensor(x[i])).sum().backward()
        per_sample += W_i.grad

    assert np.allclose(W.grad, per_sample)


def test_linear_gradcheck_no_bias():
    x, W = rand(4, 3), rand(3, 5)
    build = linear_build(3, 5, bias=False)
    gradcheck(lambda x, W: (x @ W).sum(), lambda x, W: build(x, W), [x, W])


def test_linear_stacked_gradcheck():
    """Two layers — gradient has to flow through the first layer's output."""
    x, W1, b1, W2, b2 = rand(4, 3), rand(3, 6), rand(6), rand(6, 2), rand(2)

    def f(x, W1, b1, W2, b2):
        return ((x @ W1 + b1) @ W2 + b2).sum()

    def build(x, W1, b1, W2, b2):
        l1, l2 = Linear(3, 6), Linear(6, 2)
        l1.weight, l1.bias = W1, b1
        l2.weight, l2.bias = W2, b2
        return l2(l1(x)).sum()

    gradcheck(f, build, [x, W1, b1, W2, b2])


# ---- module plumbing ----

def test_parameters_returns_weight_and_bias():
    lin = Linear(3, 5)
    assert [id(p) for p in lin.parameters()] == [id(lin.weight), id(lin.bias)]


def test_parameters_omits_bias_when_disabled():
    lin = Linear(3, 5, bias=False)
    assert [id(p) for p in lin.parameters()] == [id(lin.weight)]
    assert lin.bias is None


def test_zero_grad_clears_accumulated_gradients():
    lin = Linear(3, 5)
    lin(Tensor(rand(4, 3))).sum().backward()
    assert np.any(lin.weight.grad != 0)
    lin.zero_grad()
    assert np.all(lin.weight.grad == 0)
    assert np.all(lin.bias.grad == 0)


def test_linear_is_a_module():
    assert isinstance(Linear(3, 5), Module)


def test_wrong_input_width_is_rejected():
    with pytest.raises(Exception):
        Linear(3, 5)(Tensor(rand(4, 7))).sum().backward()


# ============================ LayerNorm ============================

def layernorm_np(x, gamma, beta, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * gamma + beta


def layernorm_build(dim, eps=1e-5):
    """Build fn that stuffs the given gamma/beta Tensors into a fresh LayerNorm."""
    def build(x, gamma, beta):
        ln = LayerNorm(dim, eps=eps)
        ln.gamma, ln.beta = gamma, beta
        return ln(x).sum()
    return build


# ---- init: ones and zeros, NOT the Linear init ----

def test_layernorm_gamma_starts_at_ones():
    assert np.all(LayerNorm(8).gamma.data == 1.0)


def test_layernorm_beta_starts_at_zeros():
    assert np.all(LayerNorm(8).beta.data == 0.0)


def test_layernorm_param_shapes():
    ln = LayerNorm(8)
    assert ln.gamma.shape == (8,)
    assert ln.beta.shape == (8,)


def test_layernorm_init_is_deterministic():
    """No rng involved at all — unlike Linear, two instances are identical."""
    assert np.all(LayerNorm(8).gamma.data == LayerNorm(8).gamma.data)


# ---- forward ----

def test_layernorm_forward_matches_numpy():
    x, g, b = rand(4, 6, 8), rand(8), rand(8, seed=1)
    ln = LayerNorm(8)
    ln.gamma, ln.beta = Tensor(g), Tensor(b)
    assert np.allclose(ln(Tensor(x)).data, layernorm_np(x, g, b))


def test_layernorm_preserves_shape():
    assert LayerNorm(8)(Tensor(rand(4, 6, 8))).shape == (4, 6, 8)


def test_layernorm_is_a_module():
    assert isinstance(LayerNorm(8), Module)


def test_layernorm_parameters():
    ln = LayerNorm(8)
    assert [id(p) for p in ln.parameters()] == [id(ln.gamma), id(ln.beta)]


# ---- (3) sanity, no gradient: normalized output is mean 0 / var 1 per group ----

def test_layernorm_output_is_standardized_per_group():
    """gamma=ones, beta=zeros (the default init) => mean 0, var 1 along the last axis.

    eps is tiny here on purpose: it biases the output variance by
    eps/(v+eps), which at the default eps=1e-5 and v~1 is ~1e-5 — bigger than
    the atol=1e-6 asked for. That offset is eps doing its job, not a bug; the
    test right below pins it.
    """
    x = rand(4, 6, 8)
    out = LayerNorm(8, eps=1e-12)(Tensor(x)).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-6)


def test_layernorm_default_eps_shrinks_variance_by_a_known_amount():
    """Documents the offset the strict test above dodges: var_out = v/(v+eps)."""
    x = rand(4, 6, 8)
    out = LayerNorm(8, eps=1e-5)(Tensor(x)).data
    v = x.var(axis=-1)
    assert np.allclose(out.var(axis=-1), v / (v + 1e-5), atol=1e-9)


def test_layernorm_constant_row_does_not_blow_up():
    """v=0 on a constant row — eps is the only thing between this and 0/0."""
    out = LayerNorm(8)(Tensor(np.full((2, 8), 3.0))).data
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


# ---- (1) full gradcheck ----

def test_layernorm_gradcheck_4_6_8():
    """Full gradcheck over the last axis, x/gamma/beta all at once."""
    x, g, b = rand(4, 6, 8), rand(8), rand(8, seed=1)
    gradcheck(lambda x, g, b: layernorm_np(x, g, b).sum(),
              layernorm_build(8), [x, g, b])


def test_layernorm_gradcheck_2d():
    x, g, b = rand(5, 8), rand(8), rand(8, seed=1)
    gradcheck(lambda x, g, b: layernorm_np(x, g, b).sum(),
              layernorm_build(8), [x, g, b])


def test_layernorm_gradcheck_nonuniform_upstream():
    """Square the output so the upstream grad is not all-ones."""
    x, g, b = rand(4, 6, 8), rand(8), rand(8, seed=1)

    def f(x, g, b):
        y = layernorm_np(x, g, b)
        return (y * y).sum()

    def build(x, g, b):
        ln = LayerNorm(8)
        ln.gamma, ln.beta = g, b
        y = ln(x)
        return (y * y).sum()

    gradcheck(f, build, [x, g, b])


def test_layernorm_grad_shapes_survive_batch():
    ln = LayerNorm(8)
    ln(Tensor(rand(4, 6, 8))).sum().backward()
    assert ln.gamma.grad.shape == (8,)
    assert ln.beta.grad.shape == (8,)


# ---- (2) dgamma/dbeta sum over the batch axes ----

def test_layernorm_batched_dgamma_dbeta_equal_sum_over_batch():
    """Same shape of proof as test_linear_batched_dW_equals_sum_over_batch."""
    g_np, b_np = rand(8), rand(8, seed=1)
    x = rand(4, 6, 8)

    gamma, beta = Tensor(g_np), Tensor(b_np)
    ln = LayerNorm(8)
    ln.gamma, ln.beta = gamma, beta
    ln(Tensor(x)).sum().backward()

    per_slice_g, per_slice_b = np.zeros(8), np.zeros(8)
    for i in range(x.shape[0]):
        g_i, b_i = Tensor(g_np), Tensor(b_np)
        ln_i = LayerNorm(8)
        ln_i.gamma, ln_i.beta = g_i, b_i
        ln_i(Tensor(x[i])).sum().backward()
        per_slice_g += g_i.grad
        per_slice_b += b_i.grad

    assert np.allclose(gamma.grad, per_slice_g)
    assert np.allclose(beta.grad, per_slice_b)


def test_layernorm_dbeta_is_count_of_normalized_groups():
    """beta is added to every element, so dbeta = number of rows, per feature."""
    ln = LayerNorm(8)
    ln(Tensor(rand(4, 6, 8))).sum().backward()
    assert np.allclose(ln.beta.grad, 4 * 6)


def test_layernorm_zero_grad_clears_both_params():
    ln = LayerNorm(8)
    ln(Tensor(rand(4, 6, 8))).sum().backward()
    assert np.any(ln.beta.grad != 0)
    ln.zero_grad()
    assert np.all(ln.gamma.grad == 0)
    assert np.all(ln.beta.grad == 0)


# ---- composes with Linear ----

# ================= scaled dot-product attention (no mask yet) =================
#
# Pure composition, one line per existing op:
#
#     scores = (q @ k^T) / sqrt(dk)
#         ├─ Tensor.transpose  (swap of the last two axes only; the op itself
#         │                     was gradchecked on non-involutive perms too)
#         ├─ __matmul__        ((B,H,T,dk) @ (B,H,dk,T) — two batch dims, the
#         │                     case closed in test_tensor.py)
#         └─ __truediv__       (by the python float sqrt(dk))
#
#     out = softmax(scores, axis=-1) @ v
#         ├─ Tensor.softmax    (over the key axis)
#         └─ __matmul__        ((B,H,T,T) @ (B,H,T,dk))
#
# ---------------------------------------------------------------------------
# Máscara causal: por que ADITIVA, ANTES do softmax, e com -1e9 e não -inf
# ---------------------------------------------------------------------------
#
# ANTES, nunca depois
# -------------------
# O softmax normaliza: cada linha dos pesos soma 1. Zerar as posições futuras
# DEPOIS do softmax remove massa de uma distribuição já normalizada, e a linha
# passa a somar (algo < 1) — diferente em cada linha, porque cada uma esconde
# uma quantidade diferente de posições. A saída vira uma combinação convexa
# encolhida por um fator que varia com a posição: não é mais média ponderada,
# é média ponderada multiplicada por um número arbitrário.
#
# Mascarando antes, os scores proibidos entram no exp já desprezíveis, o
# denominador só soma os permitidos, e a linha volta a somar exatamente 1.
#
# O incômodo: mascarar depois é causal em v, mas NÃO em k. Verificado por
# mutação, não por intuição:
#
#   w'_ij = softmax_j(s_i)_j · 1[j<=i]
#
# Os pesos que sobrevivem já foram divididos por Σ_j exp(s_ij) somado sobre a
# linha INTEIRA, futuro incluso. Então out_i não depende de v_j futuro (o peso
# é zero), mas depende de k_j futuro através do denominador. Vaza informação
# do futuro por um caminho que ninguém olha.
#
# Consequência prática pros testes: os testes de causalidade em v — que são os
# que a maioria escreve — passam com o bug. Quem pega são o teste de "linha
# soma 1" (com v=ones a saída tem que ser exatamente ones) e os testes de
# causalidade em k.
#
# ADITIVA, não multiplicativa
# ---------------------------
# scores + mask, com mask = 0 no permitido e um negativo grande no proibido.
# A alternativa scores * mask com mask 0/1 zera o score, e score 0 vira
# exp(0)=1, que é um peso ALTO, não baixo. Multiplicar mascara errado.
#
# -1e9 e não float('-inf')
# ------------------------
# Com máscara causal os dois dão o mesmo número: a diagonal é sempre
# permitida, então nenhuma linha fica inteiramente mascarada, o max da linha é
# finito, e exp(-inf - finito) = 0 sem nan. -inf funciona aqui.
#
# A escolha é por causa do que vem depois. Nosso softmax subtrai o max do eixo
# para estabilidade; se uma linha inteira for mascarada — o que uma máscara de
# padding produz o tempo todo — o max daquela linha é -inf, e -inf - (-inf) =
# nan. O nan então contamina a linha toda e, pelo backward, o batch inteiro.
# Com -1e9 a mesma linha degenera para uma distribuição uniforme: errada, mas
# finita, localizada e depurável. Trocar um nan silencioso por um número
# obviamente errado é a troca certa num framework de treino.
#
# Consequência: -1e9 não é detectável pelos testes deste ciclo (nenhum mascara
# uma linha inteira). É decisão de design documentada, não bug — e
# test_why_not_literal_negative_infinity mostra a diferença no nível do
# softmax, onde ela é observável.

# ---------------------------------------------------------------------------
# Embedding: gather no forward, scatter-add no backward
# ---------------------------------------------------------------------------
#
# Forward — gather
# ----------------
#     out[b, t] = weight[idx[b, t]]
#
# Nenhuma conta, só leitura de linhas. weight tem shape (vocab, dim), idx tem
# shape qualquer (b, t) de inteiros, e a saída sai com idx.shape + (dim,).
# Os índices NÃO são diferenciáveis — são endereços, não números. O único
# parâmetro é weight.
#
# A mesma classe serve token embedding e positional embedding: muda a tabela e
# muda o que você passa como idx (tokens num caso, arange(T) no outro).
#
# Backward — scatter-add
# ----------------------
# Se a linha v foi lida em várias posições, ela é um NÓ COMPARTILHADO: cada
# leitura é um caminho pelo qual o gradiente volta, e todos somam. Mesma
# categoria do x + x do M0, onde x usado duas vezes tem que somar 2 e não 1.
#
#     grad_weight[v] = Σ_{(b,t) : idx[b,t] == v} grad_out[b, t]
#
# Por que grad_weight[idx] += grad_out é o bug clássico
# -----------------------------------------------------
# Indexação avançada do NumPy com += NÃO acumula em índices repetidos. O
#     grad_weight[idx] += grad_out
# não é in-place de verdade: o NumPy lê grad_weight[idx] pra um buffer
# temporário, soma, e escreve o buffer de volta. Com idx = [2, 2] as duas
# posições leem o MESMO valor original, e a segunda escrita sobrescreve a
# primeira. Sobrevive só a última contribuição — a linha 2 recebe grad_out[1]
# em vez de grad_out[0] + grad_out[1].
#
#     np.add.at(grad_weight, idx, grad_out)
#
# é a versão sem buffer: acumula elemento por elemento, na ordem, e índice
# repetido soma de verdade.
#
# Por que passa despercebido: com índices únicos as duas formas dão
# exatamente o mesmo resultado. Repetição é rara em teste pequeno escrito à
# mão e é a regra absoluta em texto real — qualquer token comum aparece
# dezenas de vezes por batch. O bug fica invisível no teste e degrada o treino
# em produção.

# ---------------------------------------------------------------------------
# CausalSelfAttention: por que o split de heads tem que passar por (B,T,H,dk)
# ---------------------------------------------------------------------------
#
# Projeção q/k/v: TRÊS Linears separadas (q_proj/k_proj/v_proj), não uma
# c_attn fundida — conferido em PersonaCore gpt.py:71-74, onde os comentários
# dizem explicitamente "NOT a fused c_attn" (a separação mantém a costura do
# LoRA aberta por nome). Mais a c_proj de saída. Todas com bias.
#
# O caminho certo
# ---------------
#     (B, T, C) --reshape--> (B, T, H, dk) --transpose(0,2,1,3)--> (B, H, T, dk)
#
# O bug clássico
# --------------
#     (B, T, C) --reshape--> (B, H, T, dk)          <-- direto, sem transpose
#
# Os dois dão EXATAMENTE o mesmo shape. É por isso que assert de shape não
# distingue, e por isso que "os testes de atenção continuam passando" não
# prova nada: com pesos aleatórios a saída embaralhada é tão plausível quanto
# a correta.
#
# Row-major: C é o eixo que varia mais rápido. Pra um b fixo, a memória é
#
#     [t=0: c=0..C-1][t=1: c=0..C-1] ... [t=T-1: c=0..C-1]
#
# O elemento certo pra head h, tempo t, canal j é flat[t*C + h*dk + j].
#
#   * reshape (T,H,dk) lê [t,h,j] = flat[t*(H*dk) + h*dk + j] = flat[t*C + h*dk + j]
#     -> bate. Depois o transpose só reordena os eixos, sem mexer no conteúdo.
#
#   * reshape (H,T,dk) lê [h,t,j] = flat[h*(T*dk) + t*dk + j]
#     -> a head 0 fica com o PREFIXO do buffer, ou seja os primeiros T/H
#        instantes de tempo com TODOS os canais. Não é uma fatia de canais: é
#        uma fatia de tempo. Mistura canal com tempo, e cada head passa a ver
#        um pedaço diferente da sequência em vez do seu próprio subespaço.
#
# Daí a forma dos testes (1) e (2): igualdade DIRETA entre
# split[:, h] e a fatia de entrada x[:, :, h*dk:(h+1)*dk], head por head.
# É a única asserção que separa embaralhado de correto.

def sdpa_np(q, k, v, causal=False):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if causal:
        scores = scores + np.triu(
            np.full(scores.shape[-2:], -1e9), 1)
    return softmax_np_last(scores) @ v


def softmax_np_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_sdpa_forward_matches_numpy():
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    got = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v)).data
    assert np.allclose(got, sdpa_np(q, k, v))


def test_sdpa_output_shape():
    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    assert scaled_dot_product_attention(q, k, v).shape == (2, 3, 5, 8)


def test_sdpa_output_shape_follows_v_not_q():
    """dk of the output comes from v; q and k only decide the T x T weights."""
    q, k, v = Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 4))
    assert scaled_dot_product_attention(q, k, v).shape == (2, 3, 5, 4)


def test_sdpa_is_scaled_by_sqrt_dk():
    """Directly pins the 1/sqrt(dk): unscaled scores give a different result."""
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    got = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v)).data
    unscaled = softmax_np_last(q @ np.swapaxes(k, -1, -2)) @ v
    assert np.allclose(got, sdpa_np(q, k, v))
    assert not np.allclose(got, unscaled)


def test_sdpa_with_zero_scores_averages_v():
    """q=0 makes every score 0, so softmax is uniform and out is the mean of v."""
    q = Tensor(np.zeros((1, 1, 4, 8)))
    k, v = Tensor(rand(1, 1, 4, 8)), Tensor(rand(1, 1, 4, 8))
    out = scaled_dot_product_attention(q, k, v).data
    assert np.allclose(out, v.data.mean(axis=-2, keepdims=True))


def test_sdpa_gradcheck_q_k_v():
    """(2,3,5,8): batch=2, heads=3, seq=5, dk=8. Non-uniform upstream."""
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    w = rand(2, 3, 5, 8) + 2.0

    gradcheck(lambda q, k, v: (w * sdpa_np(q, k, v)).sum(),
              lambda q, k, v: (scaled_dot_product_attention(q, k, v) * Tensor(w)).sum(),
              [q, k, v])


def test_sdpa_gradcheck_smaller_shape():
    q, k, v = rand(2, 3, 4, 2), rand(2, 3, 4, 2, seed=1), rand(2, 3, 4, 2, seed=2)
    w = rand(2, 3, 4, 2) + 2.0

    gradcheck(lambda q, k, v: (w * sdpa_np(q, k, v)).sum(),
              lambda q, k, v: (scaled_dot_product_attention(q, k, v) * Tensor(w)).sum(),
              [q, k, v])


def test_sdpa_grad_shapes():
    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    (scaled_dot_product_attention(q, k, v) * Tensor(rand(2, 3, 5, 8) + 2.0)).sum().backward()
    assert q.grad.shape == (2, 3, 5, 8)
    assert k.grad.shape == (2, 3, 5, 8)
    assert v.grad.shape == (2, 3, 5, 8)


# ---- (4) causal=False changes nothing ----

def test_causal_false_is_identical_to_the_default():
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    a = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v)).data
    b = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), causal=False).data
    assert np.array_equal(a, b)


# ---- (1) the test that sees masking-after-softmax ----

def test_causal_rows_still_sum_to_one():
    """With v = ones, the output is exactly the sum of each attention row.

    Masking BEFORE the softmax leaves every row normalized, so the output is
    exactly 1 everywhere. Masking AFTER would strip mass from an already
    normalized row, and row i would come out at less than 1 — by a different
    amount per row. This is the only causality test that can see that bug.
    """
    q, k = Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1))
    v = Tensor(np.ones((2, 3, 5, 8)))
    out = scaled_dot_product_attention(q, k, v, causal=True).data
    assert np.allclose(out, 1.0, atol=1e-12)


def test_causal_first_row_attends_only_to_itself():
    """Row 0 has exactly one allowed position, so its output must equal v[0]."""
    q, k = Tensor(rand(1, 1, 4, 8)), Tensor(rand(1, 1, 4, 8, seed=1))
    v_np = rand(1, 1, 4, 8)
    out = scaled_dot_product_attention(q, k, Tensor(v_np), causal=True).data
    assert np.allclose(out[:, :, 0, :], v_np[:, :, 0, :])


def test_why_not_literal_negative_infinity():
    """A fully masked row: -1e9 degrades to uniform, -inf would produce nan.

    Causal masking never hits this (the diagonal is always allowed), which is
    exactly why no other test here can justify the choice. A padding mask does
    hit it, so the reasoning is recorded where it is observable.
    """
    with np.errstate(invalid="ignore"):
        big_neg = Tensor(np.full((1, 4), -1e9)).softmax(axis=-1).data
        literal_inf = Tensor(np.full((1, 4), -np.inf)).softmax(axis=-1).data

    assert np.all(np.isfinite(big_neg)) and np.allclose(big_neg, 0.25)
    assert np.isnan(literal_inf).all()


# ---- (2) the test that actually proves causality ----

def test_causal_output_ignores_future_v():
    """Perturb v at a future position; earlier outputs must not move at all."""
    q, k = Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1))
    v_np = rand(2, 3, 5, 8)

    base = scaled_dot_product_attention(q, k, Tensor(v_np), causal=True).data
    bumped_np = v_np.copy()
    bumped_np[:, :, 4, :] += 100.0
    bumped = scaled_dot_product_attention(q, k, Tensor(bumped_np), causal=True).data

    assert np.array_equal(base[:, :, :4, :], bumped[:, :, :4, :])
    assert not np.allclose(base[:, :, 4, :], bumped[:, :, 4, :])


def test_causal_output_ignores_future_k():
    """Same for k: row i's scores against future keys are masked out."""
    q, v = Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=2))
    k_np = rand(2, 3, 5, 8)

    base = scaled_dot_product_attention(q, Tensor(k_np), v, causal=True).data
    bumped_np = k_np.copy()
    bumped_np[:, :, 4, :] += 100.0
    bumped = scaled_dot_product_attention(q, Tensor(bumped_np), v, causal=True).data

    assert np.allclose(base[:, :, :4, :], bumped[:, :, :4, :], atol=1e-12)


def test_causal_gradient_to_future_v_is_exactly_zero():
    """dL/dv[j] must be 0 for j > i when only row i enters the loss.

    No indexing op exists, so 'only row i' is expressed as a constant weight
    that is zero on every other row.
    """
    i = 2
    w = np.zeros((2, 3, 5, 8))
    w[:, :, i, :] = 1.0

    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    out = scaled_dot_product_attention(q, k, v, causal=True)
    (out * Tensor(w)).sum().backward()

    assert np.all(v.grad[:, :, i + 1:, :] == 0.0), "future v received gradient"
    assert np.all(v.grad[:, :, :i + 1, :] != 0.0), "past v got no gradient (vacuous)"


def test_causal_gradient_to_future_k_is_exactly_zero():
    i = 2
    w = np.zeros((2, 3, 5, 8))
    w[:, :, i, :] = 1.0

    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    (scaled_dot_product_attention(q, k, v, causal=True) * Tensor(w)).sum().backward()

    assert np.all(k.grad[:, :, i + 1:, :] == 0.0)


def test_noncausal_does_leak_to_the_future():
    """Control: without the mask the same probe DOES reach future positions,
    so the two tests above are measuring the mask and not something else."""
    i = 2
    w = np.zeros((2, 3, 5, 8))
    w[:, :, i, :] = 1.0

    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    (scaled_dot_product_attention(q, k, v, causal=False) * Tensor(w)).sum().backward()

    assert np.any(v.grad[:, :, i + 1:, :] != 0.0)


# ---- (3) gradcheck with the mask on ----

def test_sdpa_causal_gradcheck():
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    w = rand(2, 3, 5, 8) + 2.0

    gradcheck(
        lambda q, k, v: (w * sdpa_np(q, k, v, causal=True)).sum(),
        lambda q, k, v: (
            scaled_dot_product_attention(q, k, v, causal=True) * Tensor(w)).sum(),
        [q, k, v])


def test_sdpa_causal_gradcheck_smaller_shape():
    q, k, v = rand(2, 3, 4, 2), rand(2, 3, 4, 2, seed=1), rand(2, 3, 4, 2, seed=2)
    w = rand(2, 3, 4, 2) + 2.0

    gradcheck(
        lambda q, k, v: (w * sdpa_np(q, k, v, causal=True)).sum(),
        lambda q, k, v: (
            scaled_dot_product_attention(q, k, v, causal=True) * Tensor(w)).sum(),
        [q, k, v])


def test_sdpa_causal_forward_matches_numpy():
    q, k, v = rand(2, 3, 5, 8), rand(2, 3, 5, 8, seed=1), rand(2, 3, 5, 8, seed=2)
    got = scaled_dot_product_attention(
        Tensor(q), Tensor(k), Tensor(v), causal=True).data
    assert np.allclose(got, sdpa_np(q, k, v, causal=True))


def test_sdpa_causal_output_shape():
    q, k, v = (Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 5, 8, seed=1)),
               Tensor(rand(2, 3, 5, 8, seed=2)))
    assert scaled_dot_product_attention(q, k, v, causal=True).shape == (2, 3, 5, 8)


# ============================ Embedding ============================

def embedding_build(vocab, dim, idx, upstream):
    """Build fn that stuffs the given weight Tensor into a fresh Embedding."""
    def build(W):
        emb = Embedding(vocab, dim)
        emb.weight = W
        return (emb(idx) * Tensor(upstream)).sum()
    return build


# ---- forward / plumbing ----

def test_embedding_forward_is_a_gather():
    W = rand(6, 4)
    idx = np.array([[3, 0, 5], [1, 4, 2]])
    emb = Embedding(6, 4)
    emb.weight = Tensor(W)
    assert np.array_equal(emb(idx).data, W[idx])


def test_embedding_output_shape_is_idx_shape_plus_dim():
    emb = Embedding(6, 4)
    assert emb(np.array([[3, 0, 5], [1, 4, 2]])).shape == (2, 3, 4)


def test_embedding_works_as_a_positional_table():
    """Same class, different table: idx = arange(T) gives one row per position."""
    emb = Embedding(8, 4)
    assert emb(np.arange(5)).shape == (5, 4)


def test_embedding_param_shape_and_parameters():
    emb = Embedding(6, 4)
    assert emb.weight.shape == (6, 4)
    assert [id(p) for p in emb.parameters()] == [id(emb.weight)]


def test_embedding_is_a_module():
    assert isinstance(Embedding(6, 4), Module)


def test_embedding_init_is_reproducible_with_seed():
    a = Embedding(6, 4, rng=np.random.default_rng(0))
    b = Embedding(6, 4, rng=np.random.default_rng(0))
    assert np.array_equal(a.weight.data, b.weight.data)


# ---- (1) gradcheck, no repeated index ----

def test_embedding_gradcheck_distinct_indices():
    """idx is a permutation of the vocab, so every row is read exactly once.

    With no repetition, the buggy fancy-index += and the correct np.add.at are
    indistinguishable — this test cannot see that bug, by construction.
    """
    W = rand(6, 4)
    idx = np.array([[3, 0, 5], [1, 4, 2]])
    up = rand(2, 3, 4) + 2.0

    gradcheck(lambda W: (up * W[idx]).sum(),
              embedding_build(6, 4, idx, up), [W])


def test_embedding_gradcheck_1d_indices():
    W = rand(6, 4)
    idx = np.array([4, 1, 0])
    up = rand(3, 4) + 2.0

    gradcheck(lambda W: (up * W[idx]).sum(),
              embedding_build(6, 4, idx, up), [W])


# ---- (2) the test that proves the scatter-add ----

def test_embedding_repeated_index_accumulates_both_reads():
    """idx = [[2, 2]]: row 2 must receive the SUM of both positions' gradients.

    Compared by exact equality against two separate single-read backwards
    summed by hand — a fancy-index += keeps only the last write and lands on
    the second term alone.
    """
    W = rand(6, 4)
    up = rand(1, 2, 4) + 2.0

    emb = Embedding(6, 4)
    emb.weight = Tensor(W)
    (emb(np.array([[2, 2]])) * Tensor(up)).sum().backward()
    together = emb.weight.grad

    apart = np.zeros((6, 4))
    for pos in range(2):
        e = Embedding(6, 4)
        e.weight = Tensor(W)
        (e(np.array([[2]])) * Tensor(up[:, pos:pos + 1, :])).sum().backward()
        apart += e.weight.grad

    assert np.array_equal(together, apart)
    assert np.array_equal(together[2], up[0, 0] + up[0, 1])


def test_embedding_index_repeated_many_times_accumulates_all():
    """Five reads of the same row — the sum must include every one of them."""
    W = rand(6, 4)
    up = rand(1, 5, 4) + 2.0

    emb = Embedding(6, 4)
    emb.weight = Tensor(W)
    (emb(np.array([[1, 1, 1, 1, 1]])) * Tensor(up)).sum().backward()

    assert np.array_equal(emb.weight.grad[1], up[0].sum(axis=0))


def test_embedding_repeats_across_rows_of_the_batch():
    """The same token in different batch rows also has to accumulate."""
    W = rand(6, 4)
    up = rand(2, 2, 4) + 2.0

    emb = Embedding(6, 4)
    emb.weight = Tensor(W)
    (emb(np.array([[3, 0], [3, 5]])) * Tensor(up)).sum().backward()

    assert np.array_equal(emb.weight.grad[3], up[0, 0] + up[1, 0])


def test_embedding_gradcheck_with_repeated_indices():
    """Numeric confirmation of the same thing, via the gradcheck harness."""
    W = rand(6, 4)
    idx = np.array([[2, 2, 5], [2, 0, 5]])
    up = rand(2, 3, 4) + 2.0

    gradcheck(lambda W: (up * W[idx]).sum(),
              embedding_build(6, 4, idx, up), [W])


# ---- (3) untouched rows stay at exactly zero ----

def test_embedding_unused_rows_get_exactly_zero_gradient():
    W = rand(6, 4)
    up = rand(1, 2, 4) + 2.0

    emb = Embedding(6, 4)
    emb.weight = Tensor(W)
    (emb(np.array([[1, 4]])) * Tensor(up)).sum().backward()

    for row in (0, 2, 3, 5):
        assert np.all(emb.weight.grad[row] == 0.0), f"row {row} leaked gradient"
    assert np.all(emb.weight.grad[1] != 0.0)
    assert np.all(emb.weight.grad[4] != 0.0)


def test_embedding_gradient_flows_through_a_computed_weight():
    """weight is not required to be a leaf, and that is what the graph edge buys.

    Added after a mutation check: dropping `(self.weight,)` from the node's
    children survived the whole suite, because in every other test weight is a
    leaf and the backward writes into weight.grad directly. The moment weight
    is computed — weight tying, where the embedding table is shared with the
    output projection — the missing edge means its producer never runs and the
    gradient silently stops here.
    """
    base = Tensor(rand(6, 4))
    emb = Embedding(6, 4)
    emb.weight = base * 2.0  # a computed node, not a leaf

    (emb(np.array([[1, 4]])) * Tensor(np.ones((1, 2, 4)))).sum().backward()

    assert np.any(base.grad != 0.0), "gradient never reached the weight's producer"
    assert np.allclose(base.grad, emb.weight.grad * 2.0)


def test_embedding_grad_has_the_weight_shape():
    emb = Embedding(6, 4)
    (emb(np.array([[1, 4]])) * Tensor(rand(1, 2, 4) + 2.0)).sum().backward()
    assert emb.weight.grad.shape == (6, 4)


def test_embedding_zero_grad_clears_the_table():
    emb = Embedding(6, 4)
    (emb(np.array([[1, 4]])) * Tensor(rand(1, 2, 4) + 2.0)).sum().backward()
    assert np.any(emb.weight.grad != 0)
    emb.zero_grad()
    assert np.all(emb.weight.grad == 0)


def test_embedding_accumulates_across_two_backward_passes():
    """No zero_grad in between: the second pass adds to the first."""
    emb = Embedding(6, 4)
    up = Tensor(np.ones((1, 1, 4)))
    (emb(np.array([[2]])) * up).sum().backward()
    first = emb.weight.grad.copy()
    (emb(np.array([[2]])) * up).sum().backward()
    assert np.array_equal(emb.weight.grad, first * 2)


# ==================== CausalSelfAttention ====================

def split_heads_np(z, H):
    B, T, C = z.shape
    return z.reshape(B, T, H, C // H).transpose(0, 2, 1, 3)


def merge_heads_np(y):
    B, H, T, dk = y.shape
    return y.transpose(0, 2, 1, 3).reshape(B, T, H * dk)


def csa_np(x, Wq, bq, Wk, bk, Wv, bv, Wc, bc, H):
    q = split_heads_np(x @ Wq + bq, H)
    k = split_heads_np(x @ Wk + bk, H)
    v = split_heads_np(x @ Wv + bv, H)
    return merge_heads_np(sdpa_np(q, k, v, causal=True)) @ Wc + bc


def csa_build(n_embd, n_head, upstream):
    def build(x, Wq, bq, Wk, bk, Wv, bv, Wc, bc):
        att = CausalSelfAttention(n_embd, n_head)
        att.q_proj.weight, att.q_proj.bias = Wq, bq
        att.k_proj.weight, att.k_proj.bias = Wk, bk
        att.v_proj.weight, att.v_proj.bias = Wv, bv
        att.c_proj.weight, att.c_proj.bias = Wc, bc
        return (att(x) * Tensor(upstream)).sum()
    return build


# ---- (1) the head split, proved by direct equality ----

def test_head_split_gives_each_head_its_own_channel_slice():
    """split[:, h] must be exactly the input channels [h*dk : (h+1)*dk]."""
    B, T, H, dk = 2, 5, 3, 8
    x = rand(B, T, H * dk)

    split = Tensor(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3).data

    assert split.shape == (B, H, T, dk)
    for h in range(H):
        assert np.array_equal(split[:, h], x[:, :, h * dk:(h + 1) * dk]), f"head {h}"


def test_direct_reshape_to_B_H_T_dk_scrambles_the_heads():
    """The classic bug, pinned as a fact: same shape, wrong contents.

    Reshaping straight to (B,H,T,dk) hands head 0 the first T/H timesteps
    across all channels instead of the first dk channels across all time.
    """
    B, T, H, dk = 2, 6, 3, 4
    x = rand(B, T, H * dk)

    wrong = Tensor(x).reshape(B, H, T, dk).data
    right = Tensor(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3).data

    assert wrong.shape == right.shape, "the shapes agree — that is the whole problem"
    assert not np.array_equal(wrong, right)
    assert not np.array_equal(wrong[:, 0], x[:, :, 0:dk])
    # what the buggy reshape actually returns for head 0: a slice of TIME
    assert np.array_equal(wrong[:, 0], x[:, :T // H, :].reshape(B, T, dk))


# ---- (2) the merge back, same proof in reverse ----

def test_head_merge_puts_each_head_back_in_its_channel_slice():
    B, T, H, dk = 2, 5, 3, 8
    y = rand(B, H, T, dk)

    merged = Tensor(y).transpose(0, 2, 1, 3).reshape(B, T, H * dk).data

    assert merged.shape == (B, T, H * dk)
    for h in range(H):
        assert np.array_equal(merged[:, :, h * dk:(h + 1) * dk], y[:, h]), f"head {h}"


def test_direct_reshape_on_the_merge_path_also_scrambles():
    B, T, H, dk = 2, 6, 3, 4
    y = rand(B, H, T, dk)

    wrong = Tensor(y).reshape(B, T, H * dk).data
    right = Tensor(y).transpose(0, 2, 1, 3).reshape(B, T, H * dk).data

    assert wrong.shape == right.shape
    assert not np.array_equal(wrong, right)


def test_split_then_merge_is_the_identity():
    B, T, H, dk = 2, 5, 3, 8
    x = rand(B, T, H * dk)
    t = Tensor(x)
    back = t.reshape(B, T, H, dk).transpose(0, 2, 1, 3) \
            .transpose(0, 2, 1, 3).reshape(B, T, H * dk)
    assert np.array_equal(back.data, x)


# ---- structure ----

def test_csa_split_heads_gives_each_head_its_channel_slice():
    """The (1) proof applied to the LAYER, not just to the Tensor ops.

    Added after a mutation run: the two tests above assert the property of
    reshape/transpose directly and never touch CausalSelfAttention, so the
    layer's own split was guarded only indirectly, by the reference and
    gradcheck comparisons. This closes that.
    """
    B, T, H, dk = 2, 5, 3, 8
    att = CausalSelfAttention(H * dk, H)
    x = rand(B, T, H * dk)

    split = att._split_heads(Tensor(x), B, T).data

    assert split.shape == (B, H, T, dk)
    for h in range(H):
        assert np.array_equal(split[:, h], x[:, :, h * dk:(h + 1) * dk]), f"head {h}"


def test_csa_merge_heads_puts_each_head_back_in_its_channel_slice():
    """The (2) proof applied to the layer, same reasoning as above."""
    B, T, H, dk = 2, 5, 3, 8
    att = CausalSelfAttention(H * dk, H)
    y = rand(B, H, T, dk)

    merged = att._merge_heads(Tensor(y), B, T, H * dk).data

    assert merged.shape == (B, T, H * dk)
    for h in range(H):
        assert np.array_equal(merged[:, :, h * dk:(h + 1) * dk], y[:, h]), f"head {h}"


def test_csa_split_and_merge_round_trip_through_the_layer():
    B, T, H, dk = 2, 5, 3, 8
    att = CausalSelfAttention(H * dk, H)
    x = rand(B, T, H * dk)
    back = att._merge_heads(att._split_heads(Tensor(x), B, T), B, T, H * dk)
    assert np.array_equal(back.data, x)


def test_csa_has_four_separate_projections_not_a_fused_one():
    """Mirrors PersonaCore gpt.py:71-74 — q/k/v separate, plus the output proj."""
    att = CausalSelfAttention(24, 3)
    for name in ("q_proj", "k_proj", "v_proj", "c_proj"):
        proj = getattr(att, name)
        assert isinstance(proj, Linear)
        assert proj.weight.shape == (24, 24)
        assert proj.bias.shape == (24,)


def test_csa_projections_are_independent_tensors():
    att = CausalSelfAttention(24, 3)
    ids = {id(att.q_proj.weight), id(att.k_proj.weight),
           id(att.v_proj.weight), id(att.c_proj.weight)}
    assert len(ids) == 4
    assert not np.array_equal(att.q_proj.weight.data, att.k_proj.weight.data)


def test_csa_parameters_are_all_eight_tensors():
    att = CausalSelfAttention(24, 3)
    assert len(att.parameters()) == 8


def test_csa_output_shape():
    att = CausalSelfAttention(24, 3)
    assert att(Tensor(rand(2, 5, 24))).shape == (2, 5, 24)


def test_csa_rejects_n_embd_not_divisible_by_n_head():
    with pytest.raises(Exception):
        CausalSelfAttention(25, 3)


def test_csa_is_a_module():
    assert isinstance(CausalSelfAttention(24, 3), Module)


def test_csa_forward_matches_numpy_reference():
    B, T, C, H = 2, 5, 24, 3
    x = rand(B, T, C)
    Ws = [rand(C, C), rand(C), rand(C, C, seed=1), rand(C, seed=1),
          rand(C, C, seed=2), rand(C, seed=2), rand(C, C, seed=3), rand(C, seed=3)]

    att = CausalSelfAttention(C, H)
    att.q_proj.weight, att.q_proj.bias = Tensor(Ws[0]), Tensor(Ws[1])
    att.k_proj.weight, att.k_proj.bias = Tensor(Ws[2]), Tensor(Ws[3])
    att.v_proj.weight, att.v_proj.bias = Tensor(Ws[4]), Tensor(Ws[5])
    att.c_proj.weight, att.c_proj.bias = Tensor(Ws[6]), Tensor(Ws[7])

    assert np.allclose(att(Tensor(x)).data, csa_np(x, *Ws, H))


def test_csa_is_causal_end_to_end():
    """Perturbing a future input token must not move earlier outputs."""
    att = CausalSelfAttention(24, 3)
    x = rand(2, 5, 24)
    base = att(Tensor(x)).data
    bumped = x.copy()
    bumped[:, 4, :] += 100.0
    after = att(Tensor(bumped)).data

    assert np.allclose(base[:, :4], after[:, :4], atol=1e-9)
    assert not np.allclose(base[:, 4], after[:, 4])


# ---- (3) gradcheck of the whole layer ----

def test_csa_gradcheck_full_layer():
    """n_head=3, n_embd=24, input (2,5,24) — x and all eight parameters."""
    B, T, C, H = 2, 5, 24, 3
    arrays = [rand(B, T, C),
              rand(C, C) * WSCALE, rand(C) * WSCALE,
              rand(C, C, seed=1) * WSCALE, rand(C, seed=1) * WSCALE,
              rand(C, C, seed=2) * WSCALE, rand(C, seed=2) * WSCALE,
              rand(C, C, seed=3) * WSCALE, rand(C, seed=3) * WSCALE]
    up = rand(B, T, C) + 2.0

    gradcheck(lambda *a: (up * csa_np(*a, H)).sum(), csa_build(C, H, up), arrays)


def test_csa_gradcheck_single_head():
    """H=1 degenerates the split to a no-op reshape — a useful boundary."""
    B, T, C, H = 2, 4, 6, 1
    arrays = [rand(B, T, C),
              rand(C, C) * WSCALE, rand(C) * WSCALE,
              rand(C, C, seed=1) * WSCALE, rand(C, seed=1) * WSCALE,
              rand(C, C, seed=2) * WSCALE, rand(C, seed=2) * WSCALE,
              rand(C, C, seed=3) * WSCALE, rand(C, seed=3) * WSCALE]
    up = rand(B, T, C) + 2.0

    gradcheck(lambda *a: (up * csa_np(*a, H)).sum(), csa_build(C, H, up), arrays)


def test_csa_grad_shapes():
    att = CausalSelfAttention(24, 3)
    (att(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    for p in att.parameters():
        assert p.grad.shape == p.shape


# ==================== MLP ====================
#
# PersonaCore gpt.py:118-129, confirmed in the source rather than from the
# config: two Linears with GELU between them, hidden = 4 * n_embd hardcoded in
# the constructor (no n_inner/ffn_dim field anywhere), bias ON in both — the
# model's only bias=False is lm_head, for weight tying. Dropout is the
# identity at dropout=0.0, so three effective operations:
#
#     fc_in -> gelu -> fc_out
#
# Every test below goes through mlp(x). Testing `Tensor(h).gelu()` on its own
# would prove the op works and prove nothing about the layer wiring — the
# lesson from the head-split cycle, where exactly that mistake left the
# layer's split guarded only by accident.

def mlp_np(x, W1, b1, W2, b2):
    return gelu_np(x @ W1 + b1) @ W2 + b2


def mlp_build(n_embd, upstream):
    def build(x, W1, b1, W2, b2):
        mlp = MLP(n_embd)
        mlp.fc_in.weight, mlp.fc_in.bias = W1, b1
        mlp.fc_out.weight, mlp.fc_out.bias = W2, b2
        return (mlp(x) * Tensor(upstream)).sum()
    return build


def loaded_mlp(n_embd, W1, b1, W2, b2):
    mlp = MLP(n_embd)
    mlp.fc_in.weight, mlp.fc_in.bias = Tensor(W1), Tensor(b1)
    mlp.fc_out.weight, mlp.fc_out.bias = Tensor(W2), Tensor(b2)
    return mlp


# ---- (1) shapes: in == out, hidden stays internal ----

def test_mlp_output_shape_equals_input_shape():
    assert MLP(24)(Tensor(rand(2, 5, 24))).shape == (2, 5, 24)


def test_mlp_hidden_is_four_times_n_embd_and_internal():
    mlp = MLP(24)
    assert mlp.fc_in.weight.shape == (24, 96)
    assert mlp.fc_in.bias.shape == (96,)
    assert mlp.fc_out.weight.shape == (96, 24)
    assert mlp.fc_out.bias.shape == (24,)


def test_mlp_both_linears_have_bias():
    mlp = MLP(24)
    assert mlp.fc_in.bias is not None
    assert mlp.fc_out.bias is not None


def test_mlp_parameters_are_four_tensors():
    assert len(MLP(24).parameters()) == 4


def test_mlp_is_a_module():
    assert isinstance(MLP(24), Module)


def test_mlp_works_on_a_2d_input_too():
    assert MLP(24)(Tensor(rand(5, 24))).shape == (5, 24)


# ---- (3) the ordering proof, against a manual composition ----

def test_mlp_is_linear_then_gelu_then_linear():
    """mlp(x) must equal the manual fc_in -> gelu -> fc_out composition, and
    must NOT equal any of the three plausible mis-orderings.

    Built from Tensor ops that each already carry their own gradcheck, so the
    reference is independent of the layer rather than a restatement of it.
    Every wrong ordering below produces the identical output SHAPE, which is
    why the shape tests above cannot stand in for this one.
    """
    B, T, C = 2, 5, 24
    x, W1, b1 = rand(B, T, C), rand(C, 4 * C), rand(4 * C)
    W2, b2 = rand(4 * C, C), rand(C)

    got = loaded_mlp(C, W1, b1, W2, b2)(Tensor(x)).data

    tx, tW1, tb1 = Tensor(x), Tensor(W1), Tensor(b1)
    tW2, tb2 = Tensor(W2), Tensor(b2)

    right = ((tx @ tW1 + tb1).gelu() @ tW2 + tb2).data
    no_gelu = (tx @ tW1 + tb1) @ tW2 + tb2
    gelu_first = (tx.gelu() @ tW1 + tb1) @ tW2 + tb2
    gelu_last = ((tx @ tW1 + tb1) @ tW2 + tb2).gelu()

    assert np.allclose(got, right, rtol=1e-14, atol=0.0)
    for name, wrong in [("no gelu", no_gelu), ("gelu before fc_in", gelu_first),
                        ("gelu after fc_out", gelu_last)]:
        assert wrong.shape == got.shape, f"{name}: shape alone cannot tell these apart"
        assert not np.allclose(got, wrong.data), f"mlp matched the '{name}' ordering"


def test_mlp_forward_matches_numpy_reference():
    B, T, C = 2, 5, 24
    x, W1, b1 = rand(B, T, C), rand(C, 4 * C), rand(4 * C)
    W2, b2 = rand(4 * C, C), rand(C)
    got = loaded_mlp(C, W1, b1, W2, b2)(Tensor(x)).data
    assert np.allclose(got, mlp_np(x, W1, b1, W2, b2), rtol=1e-14, atol=0.0)


def test_mlp_is_nonlinear():
    """Without the GELU the layer would collapse to a single affine map, and
    mlp(2x) would be 2*mlp(x) minus a constant. The activation breaks that."""
    C = 24
    W1, b1 = rand(C, 4 * C), rand(4 * C)
    W2, b2 = rand(4 * C, C), rand(C)
    mlp = loaded_mlp(C, W1, b1, W2, b2)

    x = rand(3, C)
    fx = mlp(Tensor(x)).data
    f2x = mlp(Tensor(2.0 * x)).data
    f0 = mlp(Tensor(np.zeros((3, C)))).data

    assert not np.allclose(f2x - f0, 2.0 * (fx - f0))


# ---- (2) gradcheck ----

def test_mlp_gradcheck_full_layer():
    """(2,5,24) with all four parameters, non-uniform upstream."""
    B, T, C = 2, 5, 24
    arrays = [rand(B, T, C), rand(C, 4 * C), rand(4 * C), rand(4 * C, C), rand(C)]
    up = rand(B, T, C) + 2.0
    gradcheck(lambda *a: (up * mlp_np(*a)).sum(), mlp_build(C, up), arrays)


def test_mlp_gradcheck_small():
    B, T, C = 2, 3, 4
    arrays = [rand(B, T, C), rand(C, 4 * C), rand(4 * C), rand(4 * C, C), rand(C)]
    up = rand(B, T, C) + 2.0
    gradcheck(lambda *a: (up * mlp_np(*a)).sum(), mlp_build(C, up), arrays)


def test_mlp_grad_shapes():
    mlp = MLP(24)
    (mlp(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    for p in mlp.parameters():
        assert p.grad.shape == p.shape


def test_mlp_zero_grad_clears_all_four():
    mlp = MLP(24)
    (mlp(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    assert any(np.any(p.grad != 0) for p in mlp.parameters())
    mlp.zero_grad()
    assert all(np.all(p.grad == 0) for p in mlp.parameters())


# ==================== Block ====================
#
# PersonaCore gpt.py:142-145, pre-norm:
#
#     x = x + self.attn(self.ln_1(x))
#     x = x + self.mlp(self.ln_2(x))
#
# LayerNorm BEFORE each sublayer, residual AROUND it. ln_f is deliberately not
# here: it is the model's final norm, applied once after the last block, and
# belongs to the GPT assembly.
#
# Every test goes through block(x). The manual reference is built from the
# block's OWN sublayers wired by hand — the sublayers are already tested
# independently, and the wiring is the only thing Block contributes.

def block_np(x, g1, be1, Wq, bq, Wk, bk, Wv, bv, Wc, bc, g2, be2, W1, bb1, W2, bb2, H):
    h = x + csa_np(layernorm_np(x, g1, be1), Wq, bq, Wk, bk, Wv, bv, Wc, bc, H)
    return h + mlp_np(layernorm_np(h, g2, be2), W1, bb1, W2, bb2)


def block_build(n_embd, n_head, upstream):
    def build(x, g1, be1, Wq, bq, Wk, bk, Wv, bv, Wc, bc, g2, be2, W1, bb1, W2, bb2):
        blk = Block(n_embd, n_head)
        blk.ln_1.gamma, blk.ln_1.beta = g1, be1
        blk.attn.q_proj.weight, blk.attn.q_proj.bias = Wq, bq
        blk.attn.k_proj.weight, blk.attn.k_proj.bias = Wk, bk
        blk.attn.v_proj.weight, blk.attn.v_proj.bias = Wv, bv
        blk.attn.c_proj.weight, blk.attn.c_proj.bias = Wc, bc
        blk.ln_2.gamma, blk.ln_2.beta = g2, be2
        blk.mlp.fc_in.weight, blk.mlp.fc_in.bias = W1, bb1
        blk.mlp.fc_out.weight, blk.mlp.fc_out.bias = W2, bb2
        return (blk(x) * Tensor(upstream)).sum()
    return build


# Weight scale for the gradcheck fixtures. Standard-normal weights over 24
# channels drive attention scores to +-120 and the loss to ~1e5; the central
# difference then has a roundoff floor of eps*|loss|/EPS ~ 2e-5, well above the
# harness atol of 1e-6, and the gradcheck fails on numerical noise rather than
# on anything being wrong. PersonaCore inits at std=0.02 (gpt.py:188), so a
# small scale is also the realistic regime, not a concession.
WSCALE = 0.2


def block_arrays(B, T, C):
    def w(*shape, seed=0):
        return rand(*shape, seed=seed) * WSCALE

    return [rand(B, T, C),
            rand(C), rand(C, seed=1),                             # ln_1
            w(C, C), w(C), w(C, C, seed=1), w(C, seed=2),          # q, k
            w(C, C, seed=2), w(C, seed=3), w(C, C, seed=3), w(C, seed=4),  # v, c
            rand(C, seed=5), rand(C, seed=6),                      # ln_2
            w(C, 4 * C), w(4 * C), w(4 * C, C), w(C, seed=7)]      # mlp


# ---- (1) the two norms must be distinct objects ----

def test_block_norms_are_distinct_arrays_not_a_shared_one():
    """LayerNorm has no rng: both norms init to the SAME values (ones/zeros).

    So comparing values proves nothing here — two independent norms and one
    shared norm are value-identical at init. Identity is the only thing that
    separates them, same as the Adam per-parameter state test.
    """
    blk = Block(24, 3)

    assert blk.ln_1 is not blk.ln_2
    assert id(blk.ln_1.gamma) != id(blk.ln_2.gamma)
    assert id(blk.ln_2.beta) != id(blk.ln_1.beta)
    assert blk.ln_1.gamma.data is not blk.ln_2.gamma.data
    # and the value check that would have passed either way:
    assert np.array_equal(blk.ln_1.gamma.data, blk.ln_2.gamma.data)


def test_block_norms_get_independent_gradients():
    """Training the block must be able to move the two norms apart."""
    blk = Block(24, 3)
    (blk(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    assert not np.allclose(blk.ln_1.gamma.grad, blk.ln_2.gamma.grad)


def test_block_parameters_are_all_sixteen_and_unique():
    blk = Block(24, 3)
    params = blk.parameters()
    assert len(params) == 16          # 2 + 8 + 2 + 4
    assert len({id(p) for p in params}) == 16


# ---- (2) the structural proof ----

def test_block_is_prenorm_with_both_residuals():
    """block(x) must equal the hand-wired pre-norm composition, and must NOT
    equal the variants missing either residual, nor the post-norm ordering.

    All of the wrong variants below produce the identical output shape.
    """
    blk = Block(24, 3)
    x = Tensor(rand(2, 5, 24))

    got = blk(x).data

    h1 = x + blk.attn(blk.ln_1(x))
    right = (h1 + blk.mlp(blk.ln_2(h1))).data

    a = blk.attn(blk.ln_1(x))                       # no first residual
    no_first = (a + blk.mlp(blk.ln_2(a))).data
    no_second = blk.mlp(blk.ln_2(h1)).data          # no second residual
    p1 = blk.ln_1(x + blk.attn(x))                  # post-norm
    post_norm = blk.ln_2(p1 + blk.mlp(p1)).data

    assert np.allclose(got, right, rtol=1e-14, atol=0.0)
    for name, wrong in [("no first residual", no_first),
                        ("no second residual", no_second),
                        ("post-norm", post_norm)]:
        assert wrong.shape == got.shape, f"{name}: shape alone cannot separate these"
        assert not np.allclose(got, wrong), f"block matched the '{name}' wiring"


def test_block_forward_matches_numpy_reference():
    B, T, C, H = 2, 5, 24, 3
    arrays = block_arrays(B, T, C)
    blk = Block(C, H)
    ts = [Tensor(a) for a in arrays[1:]]
    (blk.ln_1.gamma, blk.ln_1.beta,
     blk.attn.q_proj.weight, blk.attn.q_proj.bias,
     blk.attn.k_proj.weight, blk.attn.k_proj.bias,
     blk.attn.v_proj.weight, blk.attn.v_proj.bias,
     blk.attn.c_proj.weight, blk.attn.c_proj.bias,
     blk.ln_2.gamma, blk.ln_2.beta,
     blk.mlp.fc_in.weight, blk.mlp.fc_in.bias,
     blk.mlp.fc_out.weight, blk.mlp.fc_out.bias) = ts

    got = blk(Tensor(arrays[0])).data
    assert np.allclose(got, block_np(*arrays, H), rtol=1e-13, atol=0.0)


def test_block_does_not_apply_a_final_norm():
    """ln_f belongs to the model, not the block: the output of a block is a
    raw residual stream, so its rows are not standardized."""
    blk = Block(24, 3)
    out = blk(Tensor(rand(2, 5, 24) * 5.0)).data
    assert not np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)


def test_block_has_no_ln_f_attribute():
    assert not hasattr(Block(24, 3), "ln_f")


def test_block_output_shape_equals_input_shape():
    assert Block(24, 3)(Tensor(rand(2, 5, 24))).shape == (2, 5, 24)


def test_block_is_a_module():
    assert isinstance(Block(24, 3), Module)


def test_block_is_causal_end_to_end():
    blk = Block(24, 3)
    x = rand(2, 5, 24)
    base = blk(Tensor(x)).data
    bumped = x.copy()
    bumped[:, 4, :] += 100.0
    assert np.allclose(base[:, :4], blk(Tensor(bumped)).data[:, :4], atol=1e-9)


# ---- (3) gradcheck ----

def test_block_gradcheck_full():
    """(2,5,24), n_head=3, all sixteen parameters plus the input."""
    B, T, C, H = 2, 5, 24, 3
    arrays = block_arrays(B, T, C)
    up = rand(B, T, C) + 2.0
    gradcheck(lambda *a: (up * block_np(*a, H)).sum(),
              block_build(C, H, up), arrays)


def test_block_grad_shapes():
    blk = Block(24, 3)
    (blk(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    for p in blk.parameters():
        assert p.grad.shape == p.shape


def test_block_zero_grad_clears_everything():
    blk = Block(24, 3)
    (blk(Tensor(rand(2, 5, 24))) * Tensor(rand(2, 5, 24) + 2.0)).sum().backward()
    assert any(np.any(p.grad != 0) for p in blk.parameters())
    blk.zero_grad()
    assert all(np.all(p.grad == 0) for p in blk.parameters())


# ==================== GPT ====================
#
# PersonaCore gpt.py:195-213, confirmed in the source:
#
#     x = wte(idx) + wpe(arange(T))     <- pos built at RUNTIME from the real T,
#     for block in blocks: x = block(x)     not fixed at block_size; wpe holds
#     x = ln_f(x)                           block_size rows and only the first T
#     logits = x @ wte.weight.T             are read. (T,C) broadcasts over batch.
#
# lm_head is tied to wte and has no bias (gpt.py:169,184 — a head bias would be
# untied). So wte.weight is read on TWO routes and its gradient is the sum of
# both, which is the property test (2) pins down.

GPT_DIMS = dict(vocab_size=16, n_embd=8, n_head=2, n_layer=3, block_size=6)


def gpt_idx(B=2, T=4, vocab=16, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, size=(B, T))


def gpt_arrays(vocab_size, n_embd, n_layer, block_size):
    """[wte, wpe, <16 params per block>..., ln_f.gamma, ln_f.beta]."""
    C = n_embd

    def w(*shape, seed):
        return rand(*shape, seed=seed) * WSCALE

    arrays = [w(vocab_size, C, seed=100), w(block_size, C, seed=101)]
    for i in range(n_layer):
        s = 200 + 20 * i
        arrays += [rand(C, seed=s), rand(C, seed=s + 1),                    # ln_1
                   w(C, C, seed=s + 2), w(C, seed=s + 3),                   # q
                   w(C, C, seed=s + 4), w(C, seed=s + 5),                   # k
                   w(C, C, seed=s + 6), w(C, seed=s + 7),                   # v
                   w(C, C, seed=s + 8), w(C, seed=s + 9),                   # c
                   rand(C, seed=s + 10), rand(C, seed=s + 11),              # ln_2
                   w(C, 4 * C, seed=s + 12), w(4 * C, seed=s + 13),         # fc_in
                   w(4 * C, C, seed=s + 14), w(C, seed=s + 15)]             # fc_out
    return arrays + [rand(C, seed=900), rand(C, seed=901)]


def gpt_np(arrays, idx, n_head, n_layer):
    wte, wpe = arrays[0], arrays[1]
    T = idx.shape[1]
    x = wte[idx] + wpe[:T]
    off = 2
    for _ in range(n_layer):
        x = block_np(x, *arrays[off:off + 16], n_head)
        off += 16
    x = layernorm_np(x, arrays[off], arrays[off + 1])
    return x @ wte.T


def load_gpt(model, tensors, n_layer):
    model.wte.weight, model.wpe.weight = tensors[0], tensors[1]
    off = 2
    for blk in model.blocks:
        (blk.ln_1.gamma, blk.ln_1.beta,
         blk.attn.q_proj.weight, blk.attn.q_proj.bias,
         blk.attn.k_proj.weight, blk.attn.k_proj.bias,
         blk.attn.v_proj.weight, blk.attn.v_proj.bias,
         blk.attn.c_proj.weight, blk.attn.c_proj.bias,
         blk.ln_2.gamma, blk.ln_2.beta,
         blk.mlp.fc_in.weight, blk.mlp.fc_in.bias,
         blk.mlp.fc_out.weight, blk.mlp.fc_out.bias) = tensors[off:off + 16]
        off += 16
    model.ln_f.gamma, model.ln_f.beta = tensors[off], tensors[off + 1]
    return model


def gpt_build(dims, idx, upstream):
    def build(*arrays):
        model = load_gpt(GPT(**dims), list(arrays), dims["n_layer"])
        return (model(idx) * Tensor(upstream)).sum()
    return build


def loaded_gpt(dims=None, arrays=None):
    dims = dims or GPT_DIMS
    arrays = arrays or gpt_arrays(dims["vocab_size"], dims["n_embd"],
                                  dims["n_layer"], dims["block_size"])
    return load_gpt(GPT(**dims), [Tensor(a) for a in arrays], dims["n_layer"]), arrays


# ---- basic shape / structure ----

def test_gpt_output_is_logits_over_the_vocabulary():
    model = GPT(**GPT_DIMS)
    assert model(gpt_idx(B=2, T=4)).shape == (2, 4, 16)


def test_gpt_accepts_a_sequence_shorter_than_block_size():
    """wpe holds block_size rows but only the first T are read."""
    model = GPT(**GPT_DIMS)
    for T in (1, 2, 4, 6):
        assert model(gpt_idx(B=2, T=T)).shape == (2, T, 16)


def test_gpt_rejects_a_sequence_longer_than_block_size():
    with pytest.raises(Exception):
        GPT(**GPT_DIMS)(gpt_idx(B=2, T=7))


def test_gpt_is_a_module():
    assert isinstance(GPT(**GPT_DIMS), Module)


def test_gpt_has_no_separate_head_parameter():
    """The tied table is the only head weight, and there is no head bias."""
    model = GPT(**GPT_DIMS)
    assert not hasattr(model, "lm_head")
    n_blocks = GPT_DIMS["n_layer"] * 16
    assert len(model.parameters()) == 2 + n_blocks + 2


def test_gpt_parameters_are_unique():
    model = GPT(**GPT_DIMS)
    params = model.parameters()
    assert len({id(p) for p in params}) == len(params)


# ---- (1) the blocks must be distinct instances ----

def test_gpt_blocks_are_distinct_instances():
    """Same category as the ln_1/ln_2 test: three blocks reusing one instance
    would still produce the right shape and a plausible forward."""
    model = GPT(**GPT_DIMS)
    assert len(model.blocks) == 3
    assert len({id(b) for b in model.blocks}) == 3
    ids = [id(p) for b in model.blocks for p in b.parameters()]
    assert len(set(ids)) == len(ids)


def test_gpt_blocks_get_different_gradients():
    model = GPT(**GPT_DIMS)
    idx = gpt_idx(B=2, T=4)
    up = rand(2, 4, 16) + 2.0
    (model(idx) * Tensor(up)).sum().backward()

    g0 = model.blocks[0].ln_1.gamma.grad
    g1 = model.blocks[1].ln_1.gamma.grad
    g2 = model.blocks[2].ln_1.gamma.grad
    assert not np.allclose(g0, g1)
    assert not np.allclose(g1, g2)


# ---- (2) weight tying: the gradient is the sum of both routes ----

def run_gpt_with_tables(model, idx, emb_table, head_table, up, n_layer):
    """Forward the stack with the embedding and the head reading SEPARATE
    tables, so each route's gradient can be collected on its own tensor."""
    model.wte.weight = emb_table
    T = idx.shape[1]
    x = model.wte(idx) + model.wpe(np.arange(T))
    for blk in model.blocks:
        x = blk(x)
    x = model.ln_f(x)
    logits = x @ head_table.transpose(1, 0)
    return (logits * Tensor(up)).sum()


def test_weight_tying_gradient_is_the_sum_of_both_routes():
    """wte.weight is read twice: as the lookup table and as the head.

    Split the two routes onto separate tensors holding identical values, take
    each gradient alone, and check the tied run equals their sum.

    Caveat, confirmed by mutation: separating the routes requires two tables,
    which GPT.forward by construction does not offer, so this test runs a
    hand-wired stack and CANNOT guard the tying in forward. Detaching the head
    there leaves this test green. The one that guards forward is
    test_weight_tying_uses_one_tensor_for_both_reads, which goes through
    model(idx) and checks that rows the lookup never touches still receive
    gradient — something only the head route can produce.
    """
    dims = GPT_DIMS
    n_layer = dims["n_layer"]
    idx = gpt_idx(B=2, T=4)
    up = rand(2, 4, 16) + 2.0
    _, arrays = loaded_gpt()
    table = arrays[0]

    tied_model = load_gpt(GPT(**dims), [Tensor(a) for a in arrays], n_layer)
    shared = tied_model.wte.weight
    run_gpt_with_tables(tied_model, idx, shared, shared, up, n_layer).backward()
    both = shared.grad.copy()

    split_model = load_gpt(GPT(**dims), [Tensor(a) for a in arrays], n_layer)
    emb_only, head_only = Tensor(table.copy()), Tensor(table.copy())
    run_gpt_with_tables(split_model, idx, emb_only, head_only, up, n_layer).backward()

    assert np.allclose(both, emb_only.grad + head_only.grad, rtol=1e-10, atol=1e-12)
    # neither route is vacuous on its own
    assert np.any(emb_only.grad != 0.0)
    assert np.any(head_only.grad != 0.0)
    assert not np.allclose(emb_only.grad, head_only.grad)


def test_weight_tying_uses_one_tensor_for_both_reads():
    model = GPT(**GPT_DIMS)
    idx = gpt_idx(B=2, T=4)
    (model(idx) * Tensor(rand(2, 4, 16) + 2.0)).sum().backward()
    # the table accumulates from both reads, so its gradient is strictly larger
    # than what the lookup alone deposits on the rows that idx touches
    assert np.any(model.wte.weight.grad != 0.0)
    untouched = [v for v in range(16) if v not in set(idx.flatten())]
    assert untouched, "pick a T/vocab where some rows are never looked up"
    # rows never looked up still get gradient — only the head route can do that
    assert np.any(model.wte.weight.grad[untouched] != 0.0)


# ---- (4) structural proof against manual composition ----

def test_gpt_order_is_embed_then_blocks_then_lnf_then_head():
    """Wrong orderings all produce the same logits shape."""
    model, _ = loaded_gpt()
    idx = gpt_idx(B=2, T=4)
    pos = np.arange(4)
    got = model(idx).data

    head = model.wte.weight.transpose(1, 0)

    x = model.wte(idx) + model.wpe(pos)
    for blk in model.blocks:
        x = blk(x)
    right = (model.ln_f(x) @ head).data

    y = model.ln_f(model.wte(idx) + model.wpe(pos))     # ln_f before the blocks
    for blk in model.blocks:
        y = blk(y)
    lnf_first = (y @ head).data

    z = model.blocks[0](model.wte(idx))                 # wpe added after block 0
    z = z + model.wpe(pos)
    for blk in model.blocks[1:]:
        z = blk(z)
    pos_late = (model.ln_f(z) @ head).data

    m = model.wte(idx)                                  # no positional embedding
    for blk in model.blocks:
        m = blk(m)
    no_pos = (model.ln_f(m) @ head).data

    assert np.allclose(got, right, rtol=1e-13, atol=0.0)
    for name, wrong in [("ln_f before blocks", lnf_first),
                        ("wpe after block 0", pos_late),
                        ("no positional embedding", no_pos)]:
        assert wrong.shape == got.shape, f"{name}: shape cannot separate these"
        assert not np.allclose(got, wrong), f"model matched the '{name}' ordering"


def test_gpt_forward_matches_numpy_reference():
    model, arrays = loaded_gpt()
    idx = gpt_idx(B=2, T=4)
    got = model(idx).data
    assert np.allclose(got, gpt_np(arrays, idx, GPT_DIMS["n_head"],
                                   GPT_DIMS["n_layer"]), rtol=1e-12, atol=0.0)


def test_gpt_is_causal_end_to_end():
    """A token at position 3 cannot change the logits at positions 0..2."""
    model, _ = loaded_gpt()
    idx = gpt_idx(B=2, T=4)
    base = model(idx).data
    bumped = idx.copy()
    bumped[:, 3] = (bumped[:, 3] + 5) % 16
    after = model(bumped).data
    assert np.allclose(base[:, :3], after[:, :3], atol=1e-9)
    assert not np.allclose(base[:, 3], after[:, 3])


def test_gpt_applies_a_final_norm():
    """ln_f lives here, not in Block: the pre-head stream is standardized.

    Uses a default-initialized model on purpose — the mean-zero claim holds
    for gamma=1/beta=0, and a loaded model with random affine parameters
    rescales and shifts each channel, so its output mean is not zero.
    """
    model = GPT(**GPT_DIMS)
    idx = gpt_idx(B=2, T=4)
    x = model.wte(idx) + model.wpe(np.arange(4))
    for blk in model.blocks:
        x = blk(x)
    normed = model.ln_f(x).data
    assert np.allclose(normed.mean(axis=-1), 0.0, atol=1e-9)


# ---- (3) gradcheck of the whole stack ----

def test_gpt_gradcheck_full_stack():
    """Every parameter of a 3-layer model, with T=4 < block_size=6."""
    dims = GPT_DIMS
    idx = gpt_idx(B=2, T=4)
    up = rand(2, 4, dims["vocab_size"]) + 2.0
    arrays = gpt_arrays(dims["vocab_size"], dims["n_embd"],
                        dims["n_layer"], dims["block_size"])

    gradcheck(lambda *a: (up * gpt_np(list(a), idx, dims["n_head"],
                                      dims["n_layer"])).sum(),
              gpt_build(dims, idx, up), arrays)


def test_gpt_grad_shapes():
    model = GPT(**GPT_DIMS)
    (model(gpt_idx(B=2, T=4)) * Tensor(rand(2, 4, 16) + 2.0)).sum().backward()
    for p in model.parameters():
        assert p.grad.shape == p.shape


def test_gpt_zero_grad_clears_everything():
    model = GPT(**GPT_DIMS)
    (model(gpt_idx(B=2, T=4)) * Tensor(rand(2, 4, 16) + 2.0)).sum().backward()
    assert any(np.any(p.grad != 0) for p in model.parameters())
    model.zero_grad()
    assert all(np.all(p.grad == 0) for p in model.parameters())


def test_layernorm_after_linear_gradcheck():
    x, W, b, g, be = (rand(5, 3), rand(3, 8), rand(8),
                      rand(8, seed=1), rand(8, seed=2))

    def f(x, W, b, g, be):
        return layernorm_np(x @ W + b, g, be).sum()

    def build(x, W, b, g, be):
        lin, ln = Linear(3, 8), LayerNorm(8)
        lin.weight, lin.bias = W, b
        ln.gamma, ln.beta = g, be
        return ln(lin(x)).sum()

    gradcheck(f, build, [x, W, b, g, be])


# =============================================================================
# cross_entropy — composição pura, zero backward próprio
# =============================================================================
#
#     cross_entropy(logits, targets) = -log_softmax(logits, axis=-1).pick(targets).mean()
#
# Cada peça já tem gradcheck próprio: log_softmax (ciclo anterior), pick (ciclo
# anterior), mean sem eixo (M2), e a negação é o __neg__ do M1. Nenhuma
# derivada nova entra aqui.
#
# API confirmada antes de escrever, não assumida: Tensor.mean() com axis=None
# reduz um (N,) a um escalar 0-d, com grad 1/N uniforme — verificado rodando.
# Não precisa de axis=0 explícito, e o assert de backward() (ndim == 0) aceita
# o resultado direto.
#
# Convenção fixada pelo PersonaCore (gpt.py:212), F.cross_entropy sem kwargs:
#   * redução MEAN sobre todos os B*T tokens (não sum, não mean-sobre-válidos)
#   * sem ignore_index — não existe padding, as janelas são sempre cheias
#   * sem label smoothing, sem peso por classe
# O teste (4) trava a média contra sum()/N explícito: as duas só diferem por um
# fator N, que é invisível pra qualquer gradcheck (o numérico e o analítico
# escalam juntos) e muda o learning rate efetivo por um fator 5 aqui, 8192 no
# modelo real.
#
# Por que o gradcheck aqui NÃO precisa de upstream não-uniforme
# --------------------------------------------------------------
# Nos ciclos anteriores o truque era obrigatório porque a saída era um vetor: a
# semente do backward é escolhida por quem escreve o teste, e uma semente
# uniforme pode fazer o erro cancelar (softmax(x).sum() é constante = 1, então
# gradcheck nisso dá zero mesmo com o backward quebrado).
#
# Aqui a loss JÁ é escalar — não há semente a escolher, .backward() semeia 1 e
# pronto. E a saída não é constante em direção nenhuma: dL/dlogits =
# (softmax - onehot)/N depende de cada entrada, então não existe o modo
# degenerado que o softmax tinha.
#
# O que ISSO não prova, e é honesto registrar: como a loss é uma média, o
# upstream que chega no pick é uniforme (-1/N em toda linha). Um bug que
# permutasse linhas dentro do pick seria invisível daqui. É exatamente por isso
# que o pick carrega o seu próprio gradcheck com w não-uniforme — a cobertura
# vem de lá, não deste teste.

# Referências de F.cross_entropy(logits.double(), targets, reduction=...) em
# float64 explícito, torch 2.7.1, geradas no PersonaCore. N=5, V=8 (N != V, o
# mesmo motivo do ciclo do pick), com as classes 3 e 0 repetidas entre linhas.
CE_LOGITS = np.array([
    [0.1257302210933933, -0.1321048632913019, 0.6404226504432821, 0.10490011715303971,
     -0.535669373161111, 0.36159505490948474, 1.3040000451301372, 0.9470809631292422],
    [-0.7037352358069926, -1.2654214710460525, -0.6232744625373522, 0.0413259793472436,
     -2.3250307746388343, -0.21879166393254573, -1.2459109472530652, -0.7322673547034516],
    [-0.5442589828573099, -0.31630015636915454, 0.4116305363741328, 1.0425133694426776,
     -0.12853466294403426, 1.3664634705496859, -0.6651946734866135, 0.3515100700930197],
    [0.9034701816518086, 0.09401229776087457, -0.7434992493538084, -0.9217253762584194,
     -0.45772582566733916, 0.2201951234700494, -1.009618183538736, -0.20917557487171307],
    [-0.15922500991447772, 0.5408455846858077, 0.2146591225063409, 0.3553727090399214,
     -0.6538286094183394, -0.12961363369276946, 0.7839754700613295, 1.4934311452207607],
])
CE_TARGETS = np.array([3, 0, 7, 3, 0])
CE_MEAN_REFERENCE = 2.4851746725158379
CE_SUM_REFERENCE = 12.425873362579189

# Tolerância derivada, não ajustada até passar. A loss é: 8 exponenciais
# somadas -> log -> subtração -> média sobre 5. São ~15 operações de ponto
# flutuante em série, cada uma com no máximo 1 ulp de erro relativo
# (eps = 2.22e-16), e nenhuma delas é uma subtração de valores próximos — não
# há cancelamento catastrófico neste regime (logits ~1.5, logsumexp ~2.5). O
# limite é então ~15·eps ≈ 3.3e-15 relativo. rtol=1e-14 é esse limite com uma
# margem de 3x. O erro medido é reportado no próprio teste.
CE_RTOL = 1e-14


def naive_ce_np(logits, targets):
    """-mean(log(softmax(logits))[i, targets[i]]) — a definição direta, com
    softmax INGÊNUO (sem shift). Só serve em regime moderado; o teste que a usa
    documenta o porquê."""
    e = np.exp(logits)
    probs = e / e.sum(axis=-1, keepdims=True)
    return -np.mean([np.log(probs[i, t]) for i, t in enumerate(targets)])


# ---- (1) contra a referência do PyTorch ----

def test_cross_entropy_matches_pytorch_float64():
    got = cross_entropy(Tensor(CE_LOGITS), CE_TARGETS)

    err = abs(got.data - CE_MEAN_REFERENCE) / abs(CE_MEAN_REFERENCE)
    assert err < CE_RTOL, f"erro relativo {err:.3e} excede {CE_RTOL:.0e}"
    assert got.data.ndim == 0, "a loss é escalar"


def test_cross_entropy_gradient_matches_pytorch_float64():
    """dL/dlogits = (softmax - onehot)/N, contra o autograd do torch em float64."""
    x = Tensor(CE_LOGITS)
    cross_entropy(x, CE_TARGETS).backward()

    probs = softmax_np(CE_LOGITS, -1)
    expected = probs.copy()
    expected[np.arange(len(CE_TARGETS)), CE_TARGETS] -= 1.0
    expected /= len(CE_TARGETS)
    # bate com o grad do torch colado abaixo (primeira linha, como amostra)
    assert np.allclose(expected[0][:2], [0.01705076258088749, 0.013175510542050498], atol=1e-15)
    assert np.allclose(x.grad, expected, rtol=CE_RTOL, atol=1e-16)


# ---- (2) contra a definição matemática direta ----

def test_cross_entropy_matches_the_naive_definition():
    """-mean(log(softmax(x))[i, t_i]) com softmax ingênuo, em regime seguro.

    Prova que a composição bate com a DEFINIÇÃO, não só com o PyTorch — se as
    duas referências fossem a mesma implementação, um erro compartilhado
    passaria pelas duas. Os logits aqui vão de -2.3 a 1.5, então exp() não
    chega perto de transbordar e a versão ingênua é exata; é o regime onde ela
    pode servir de oráculo, e só ele.
    """
    got = cross_entropy(Tensor(CE_LOGITS), CE_TARGETS)
    assert np.isclose(got.data, naive_ce_np(CE_LOGITS, CE_TARGETS), rtol=CE_RTOL)


def test_cross_entropy_survives_logits_that_break_the_naive_version():
    """Onde o ingênuo transborda, a composição continua exata.

    exp(1000) = inf, e o softmax ingênuo quebra de DOIS jeitos distintos,
    dependendo de qual classe é o alvo — a premissa está fixada nos dois
    porque errar qual deles acontece é fácil:

      * alvo no perdedor: probs = exp(0)/inf = 0 exato, log(0) = -inf, loss +inf
      * alvo no vencedor: probs = inf/inf = nan, loss nan

    A referência do torch para estes valores é exatamente 1000 e exatamente 0 —
    o gap de 1000 satura a probabilidade em 1 e 0 dentro do float64.
    """
    big = np.array([[0.0, 1000.0, 2.0], [1000.0, 0.0, 3.0]])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        assert naive_ce_np(big, [0, 1]) == np.inf, "premissa: perdedor -> +inf"
        assert np.isnan(naive_ce_np(big, [1, 0])), "premissa: vencedor -> nan"

    assert cross_entropy(Tensor(big), np.array([0, 1])).data == 1000.0
    assert cross_entropy(Tensor(big), np.array([1, 0])).data == 0.0


# ---- (3) gradcheck ----

def test_cross_entropy_gradcheck():
    """Sem upstream não-uniforme: a loss já é escalar (ver o comentário acima)."""
    x = rand(5, 8, seed=3)
    gradcheck(lambda a: naive_ce_np(a, CE_TARGETS),
              lambda a: cross_entropy(a, CE_TARGETS), [x])


def test_cross_entropy_gradcheck_with_repeated_targets():
    """Todas as linhas no mesmo alvo — o caso que exercita o pick sem colisão."""
    x = rand(5, 8, seed=4)
    t = np.array([2, 2, 2, 2, 2])
    gradcheck(lambda a: naive_ce_np(a, t), lambda a: cross_entropy(a, t), [x])


def test_cross_entropy_gradcheck_through_a_linear():
    """A loss no fim de um grafo real — o grad atravessa a projeção até W e b."""
    x = rand(5, 4, seed=5)
    W = rand(4, 8, seed=6)
    b = rand(8, seed=7)

    gradcheck(lambda p, q, r: naive_ce_np(p @ q + r, CE_TARGETS),
              lambda p, q, r: cross_entropy(p @ q + r, CE_TARGETS), [x, W, b])


# ---- (4) a redução é média sobre N, não soma ----

def test_cross_entropy_reduces_by_mean_not_sum():
    """Trava a convenção do PersonaCore (gpt.py:212, F.cross_entropy default).

    mean e sum diferem por um fator N exato. Nenhum gradcheck pega isso — o
    numérico e o analítico escalam juntos — e o efeito prático é o learning
    rate efetivo mudar por N: 5 aqui, 8192 (B*T) no modelo real.
    """
    got = cross_entropy(Tensor(CE_LOGITS), CE_TARGETS).data
    n = len(CE_TARGETS)

    per_row = -log_softmax_np(CE_LOGITS, -1)[np.arange(n), CE_TARGETS]
    assert np.isclose(got, per_row.sum() / n, rtol=CE_RTOL)
    assert np.isclose(got * n, CE_SUM_REFERENCE, rtol=CE_RTOL), "sum = mean * N"
    assert not np.isclose(got, CE_SUM_REFERENCE), "não é a soma"


def test_cross_entropy_gradient_sums_to_zero_per_row():
    """Σⱼ dL/dlogits[i,j] = 0: softmax soma 1 e o onehot soma 1.

    Uma consequência estrutural do sinal e da redução ao mesmo tempo — se o
    negativo sumisse, a soma por linha continuaria zero, mas com sum em vez de
    mean cada linha somaria zero também. Serve como invariante, não como
    discriminador; quem discrimina são (1)/(2)/(4).
    """
    x = Tensor(CE_LOGITS)
    cross_entropy(x, CE_TARGETS).backward()
    assert np.allclose(x.grad.sum(axis=-1), 0.0, atol=1e-15)


def test_cross_entropy_is_positive_and_beats_uniform_when_confident():
    """Loss >= 0 sempre, e um logit alto no alvo certo dá loss menor que uniforme."""
    uniform = np.zeros((5, 8))
    confident = np.zeros((5, 8))
    confident[np.arange(5), CE_TARGETS] = 10.0

    assert np.isclose(cross_entropy(Tensor(uniform), CE_TARGETS).data, np.log(8))
    assert 0.0 < cross_entropy(Tensor(confident), CE_TARGETS).data < 1e-3
