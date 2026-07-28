"""
Gradcheck tests for the tensor autograd engine (M1).

Same contract as M0: numerical gradient (central difference) vs analytical
(.backward()). Difference is that a gradient is now an array, so every check
also asserts grad.shape == param.shape — that is what catches the classic
broadcasting bug (grad must be *reduced back* to the input's shape).

Every build_* ends in .sum() so backward() starts from a scalar.


Derivação manual — primitivos de redução (mean, var)
====================================================

d(mean)/dx, reduzindo UM eixo de tamanho N
------------------------------------------
    m_j = (1/N) · Σ_{i ∈ grupo j} x_i

    ∂m_j/∂x_i = 1/N   se x_i entra em m_j
              = 0     caso contrário

Backward: cada x_i recebe 1/N da grad que chegou no m_j dele.

    dL/dx = broadcast(G, x.shape) / N

Pegadinha do keepdims=False: o eixo reduzido some de G, e broadcast do NumPy
alinha pela direita — reinserir o eixo antes (expand_dims), senão a grad se
espalha no eixo errado e ainda por cima o shape casa por acidente.


d(var)/dx — aqui a mean é um nó COMPARTILHADO
---------------------------------------------
    v = (1/N) · Σ_i (x_i − m)²,   com   m = (1/N) · Σ_i x_i

m depende de todo x, então (x_i − m) não pode ser derivado tratando m como
constante. Regra da cadeia passando pelos dois caminhos:

    ∂v/∂x_k = (1/N) · Σ_i 2(x_i − m) · ∂(x_i − m)/∂x_k
            = (2/N) · Σ_i (x_i − m) · (δ_ik − ∂m/∂x_k)
            = (2/N) · Σ_i (x_i − m) · (δ_ik − 1/N)
            = (2/N) · [ (x_k − m) − (1/N) · Σ_i (x_i − m) ]
                                            └─────┬─────┘
                                          = 0, por definição de m

    ∂v/∂x_k = (2/N)(x_k − m)

O termo de correção morre porque Σ(x_i − m) = 0. Consequência incômoda: quem
esquecer que m depende de x chega no MESMO número. A derivação importa mesmo
assim, e o cancelamento é sorte local da var — não generaliza. No LayerNorm o
passo seguinte é (x − m)/√(v+ε), e ali as duas dependências (via m e via v)
sobrevivem e viram a fórmula de três termos.

É por isso que var aqui é composta de ops que já existem (sub, mul, mean) em
vez de uma _backward escrita à mão: o nó compartilhado fica explícito no grafo
e o acúmulo de gradiente resolve a cadeia sozinho, sem ninguém precisar
confiar no cancelamento.

Convenção: var enviesada (ddof=0, divide por N) — a que o LayerNorm usa e o
default do np.var.


Derivação manual — sqrt e divisão
=================================

d(sqrt(x))/dx
-------------
    s = √x   →   ds/dx = ½ · x^(−1/2) = 1/(2s)

    dL/dx = G / (2s)

Escrever em função de s (a saída, que o forward já calculou) em vez de x evita
uma segunda raiz no backward. Mesma conta, um sqrt a menos por nó.

Domínio: a derivada explode quando x → 0⁺ e vira nan para x < 0. Não há guarda
no código — quem chama é que soma o eps. O LayerNorm faz √(v + eps), nunca √v,
e é justamente por isso: v pode ser 0 numa linha constante, e aí 1/(2s) é inf.


d(a/b)/da e d(a/b)/db
---------------------
    q = a / b

    ∂q/∂a = 1/b        →   dL/da = G / b
    ∂q/∂b = −a / b²    →   dL/db = −G · a / b²

O sinal do lado de b é o erro clássico: b maior faz q menor, então a derivada é
negativa. Um gradcheck com b não-constante pega isso na hora.

Os dois lados podem ter broadcastado — no LayerNorm b é o desvio padrão, shape
(…, 1), dividindo um a de shape (…, N) — então cada gradiente passa pelo
_unbroadcast antes de acumular, exatamente como add e mul já fazem. É o mesmo
mecanismo do M1, sem matemática nova: só não pode esquecer de chamar.


Derivação manual — softmax (M3)
===============================

    yᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)        (soma sobre o eixo normalizado)

Jacobiano: cada saída depende de TODAS as entradas do grupo, então não é uma
op elemento-a-elemento. Duas metades:

    ∂yᵢ/∂xₖ = yᵢ(δᵢₖ − yₖ)

Contraindo com a grad que vem de cima (g = dL/dy):

    dL/dxₖ = Σᵢ gᵢ · yᵢ(δᵢₖ − yₖ)
           = gₖyₖ − yₖ Σᵢ gᵢyᵢ
           = yₖ · (gₖ − Σⱼ gⱼyⱼ)

O Σⱼ gⱼyⱼ é um escalar por grupo (keepdims=True pra broadcastar de volta). Só
precisa de y — a entrada x não aparece no backward.


Estabilidade numérica: o −max
-----------------------------
exp(1000) = inf, e inf/inf = nan. Como softmax é invariante a deslocamento
(somar c no numerador e no denominador cancela), dá pra subtrair o máximo do
eixo antes do exp:

    z = x − max(x)   →   maior expoente é 0   →   exp(z) ∈ (0, 1]

Sem overflow. Os termos pequenos viram underflow pra 0, o que é inofensivo:
eles já eram desprezíveis no denominador. O backward não muda nada com isso,
porque está escrito só em função de y.


Armadilha no teste
------------------
softmax(x).sum() == 1 para QUALQUER x. Logo o gradiente dessa expressão é
exatamente zero, e um gradcheck em cima dela passa mesmo com o backward
completamente quebrado — desde que o backward devolva zeros. É a única
derivação aqui onde o teste óbvio é inútil.

Os gradchecks de softmax usam upstream não-uniforme: Σ(w · softmax(x)) com w
fixo e não-constante, ou MSE contra um one-hot. O
test_softmax_sum_is_constant_so_its_gradient_is_zero documenta a armadilha em
vez de escondê-la.


reshape e transpose (M3)
========================

reshape — sem derivada nova
---------------------------
Reordenar a view não muda valor nenhum: cada elemento da saída É um elemento da
entrada. O backward só devolve a grad ao shape de origem.

    dL/dx = out.grad.reshape(x.shape)

Cuidado no teste: com .sum() puro a grad é toda 1, então um backward quebrado
que devolva ones do shape certo passa. Todo gradcheck de reshape aqui usa
upstream não-uniforme (w · reshape(x)).


transpose — a inversa NÃO é a mesma permutação
----------------------------------------------
    y = transpose(x, axes)      →      dL/dx = transpose(out.grad, argsort(axes))

O forward permuta com `axes`; o backward tem que DESFAZER isso, e desfazer é a
permutação inversa, não `axes` de novo:

    inv = np.argsort(axes)      # inv[axes[i]] == i

Por que quase ninguém percebe o erro: as permutações usadas no dia a dia são
involutivas (inversa == ela mesma), e nelas `axes` e `argsort(axes)` são
idênticos.

    (1, 0)          → argsort = (1, 0)          IGUAL   — swap 2-D
    (0, 2, 1, 3)    → argsort = (0, 2, 1, 3)    IGUAL   — split de heads
    (1, 2, 0)       → argsort = (2, 0, 1)       DIFERENTE

Ou seja: um backward que reusa `axes` passa em todo teste de swap e em todo
teste do padrão de attention. Só um ciclo de 3+ eixos separa os dois. É por
isso que (1,2,0) é obrigatório aqui, e por que existe um teste que compara
argsort(axes) contra axes diretamente em vez de confiar no gradcheck.

Com dims distintos o erro estoura como ValueError de shape; num tensor cúbico
(3,3,3) ele passa silencioso e devolve números errados — o caso cúbico é o que
realmente prova a inversão.


Derivação manual — tanh (M3, pré-requisito da GELU)
===================================================

    t = tanh(x)     →     dt/dx = 1 − tanh(x)² = 1 − t²

Escrita em função de t, a saída que o forward já calculou — mesmo truque do
sqrt. Evita um segundo tanh por nó no backward.

    dL/dx = (1 − t²) · G

Sem problema de domínio: tanh é definida em toda a reta e limitada em (−1, 1),
então nem overflow nem nan em lugar nenhum.


Por que a região saturada precisa de teste próprio
--------------------------------------------------
Perto de 0 a tanh é quase a identidade e a derivada é ~1. Praticamente
qualquer fórmula errada acerta ali. O caso que separa é |x| grande, onde
t → ±1 e a derivada colapsa pra perto de zero.

O erro clássico é de digitação: (1 − t)² no lugar de 1 − t².

    x = 0   →   (1−0)² = 1          e   1−0² = 1          IGUAL
    x = 5   →   (1−t)² ≈ 8.2e-9     e   1−t² ≈ 1.8e-4     4 ordens de grandeza

Ou seja: um teste só em torno de zero não enxerga o bug.

Armadilha do gradcheck na saturação
-----------------------------------
O gradcheck aqui compara com np.allclose(atol=1e-6, rtol=1e-5). Em x = ±10 a
derivada verdadeira é ~8.2e-9 — MUITO abaixo do atol. Nessa região o allclose
passa comparando qualquer coisa pequena com qualquer outra coisa pequena, e o
teste vira decoração.

Por isso a divisão:
  * x = ±5  (derivada ~1.8e-4, acima do atol) → gradcheck normal serve;
  * x = ±10 (derivada ~8.2e-9, abaixo do atol) → comparação direta contra a
    forma fechada 1−t² com tolerância RELATIVA, mais finitude e sinal.

Precisão real de 1−t² na saturação (achado do mutation check)
--------------------------------------------------------------
1−t² com t≈1 é cancelamento catastrófico: em x=10, t = 0.9999999958…, e o
erro de arredondamento de t² (~2.2e-16 absoluto) vira ~2.7e-8 RELATIVO no
resultado ~8.2e-9. A fórmula entrega ~7-8 dígitos relativos ali, não 16.

Por isso o rtol=1e-12 do teste passa com folga enorme: os dois lados chamam
np.tanh, então usam o MESMO t e o cancelamento é idêntico nos dois. O teste
prende a forma da fórmula, não a exatidão numérica na saturação — e é
justamente por isso que ele pegou a variante sinh/cosh, algebricamente igual
mas com último bit diferente, amplificado pelo cancelamento.

Não vale trocar por sech²=1/cosh², que seria mais preciso ali: na saturação o
gradiente é desprezível pro treino de qualquer jeito, e 1−t² reusa a saída
sem um cosh extra por nó.


GELU (aproximação tanh — o gelu_new do GPT-2)
=============================================

    c1 = √(2/π)          calculado em runtime, não o decimal truncado
    c2 = 0.044715
    gelu(x) = 0.5·x·(1 + tanh(c1·(x + c2·x³)))

Composição pura: mul, add, tanh — todos já com gradcheck próprio. Zero
_backward escrito à mão, então não há derivada nova pra derivar aqui.

Não é a GELU exata (erf). O PersonaCore usa F.gelu(approximate="tanh")
(model/gpt.py:126), e as duas diferem o suficiente pra atrapalhar comparação
numérica contra o checkpoint do v1.0.


Valores de referência
---------------------
Gerados no repo do PersonaCore com torch 2.7.1:

    F.gelu(torch.tensor(xs, dtype=torch.float64), approximate="tanh")

float64 EXPLÍCITO. Com o float32 default a referência teria ~7 dígitos, e a
tolerância do teste passaria a medir a precisão do PyTorch em vez da correção
do tensorforge.


Tolerância — derivada antes de rodar, não ajustada até passar
--------------------------------------------------------------
A cadeia tem ~6 operações de ponto flutuante mais um tanh. Cada uma custa até
~1 ULP, e o tanh do numpy não é bit-a-bit o do PyTorch. Em região bem
condicionada isso dá ~1e-15 relativo; uso rtol=1e-14 (~45 ULPs de float64,
eps=2.22e-16) como folga honesta.

MAS a cauda negativa não é bem condicionada. Com x muito negativo, tanh→−1 e o
(1 + tanh) é subtração de números quase iguais — cancelamento catastrófico, o
mesmo fenômeno do 1−t² acima. O fator de amplificação é 1/(1+t):

    x = −5  →  inner ≈ −8.449,  t ≈ −0.99999990833,  1+t ≈ 9.17e−8
               amplificação = eps/(1+t) = 2.22e−16 / 9.17e−8 ≈ 2.4e−9

Ou seja: uma diferença de 1 ULP no tanh entre numpy e PyTorch vira ~2.4e−9
relativo na saída. Exigir 1e−14 em x=−5 seria exigir que dois tanh diferentes
concordassem bit-a-bit — o teste falharia por motivo errado. Tolerância em
x=−5: rtol=1e−8, que são ~4 ULPs depois da amplificação.

Nos extremos o resultado é EXATO dos dois lados, e a asserção é de igualdade:
x=±8 satura tanh em exatamente ∓1, então (1+t) é 0.0 ou 2.0 exato, dando −0.0
e 8.0. Em x=0 o fator x zera o produto inteiro.


Derivação manual — log_softmax (M4)
===================================

    yₖ = log softmax(x)ₖ = xₖ − log Σⱼ exp(xⱼ)

Jacobiano:

    ∂yᵢ/∂xₖ = δᵢₖ − exp(xₖ)/Σⱼexp(xⱼ) = δᵢₖ − sₖ      (s = softmax(x))

Contraindo com g = dL/dy:

    dL/dxₖ = Σᵢ gᵢ(δᵢₖ − sₖ) = gₖ − sₖ · Σᵢ gᵢ

Compare com o backward do softmax, yₖ(gₖ − Σⱼgⱼyⱼ): aqui não há multiplicação
por y. É por isso que a cross-entropy se escreve sobre log_softmax e não sobre
log(softmax) — o gradiente não passa por um fator que satura.

Op monolítica com backward fechado, mesmo padrão do softmax: mantém exp fora
do Tensor até algo mais precisar dele.


O shift de estabilidade fica FORA do grafo
------------------------------------------
    shift = x.data.max(axis, keepdims=True)     # NumPy puro, nunca Tensor

Prova de cancelamento — para QUALQUER constante c por linha:

    (xₖ − c) − log Σⱼ exp(xⱼ − c)
      = xₖ − c − log(e^(−c) · Σⱼ exp(xⱼ))
      = xₖ − c − (−c + log Σⱼ exp(xⱼ))
      = xₖ − log Σⱼ exp(xⱼ)

O c some por completo. A saída é matematicamente INDEPENDENTE do shift, logo
∂saída/∂c = 0 exatamente, e as duas rotas por onde c entraria (o −c no
numerador e o +c dentro do logsumexp) se cancelam termo a termo.

Consequência que os testes exploram: trocar o max por qualquer outro shift não
muda nem o valor nem o gradiente — só muda se o exp transborda. O mutation
check troca max por mean e por 0 pra provar isso, em vez de só afirmar.

Por que não usar um max diferenciável mesmo assim
--------------------------------------------------
Sabendo que cancela, seria tentador deixar o shift no grafo por uniformidade.
Quatro motivos pra não:

1. max não é diferenciável em empate. Com dois máximos iguais o subgradiente é
   uma escolha arbitrária (o NumPy pega o primeiro argmax). O total cancela,
   mas o grafo passaria a carregar um nó cuja derivada local é uma convenção
   de desempate.
2. O cancelamento é exato em aritmética exata, não em float. As duas
   contribuições opostas seriam calculadas separadamente e subtraídas; a
   diferença não dá zero binário, então cada gradiente ganharia ruído de
   arredondamento — de graça, já que o valor certo é zero.
3. Custo: nós a mais no caminho mais quente do treino, guardando buffers que
   existem só pra somar zero.
4. Fragilidade: se alguém depois editar a expressão de um jeito que quebre o
   cancelamento, o erro é silencioso — o resultado continua plausível.

Detached, nada disso existe: o shift é uma constante numérica escolhida por
linha, e o grafo nem sabe que ela esteve lá.


pick(idx) — uma entrada por linha
==================================
    x de forma (N, V), idx de forma (N,)  ->  out[i] = x[i, idx[i]],  forma (N,)

Forward é gather em duas coordenadas: a linha é i (posicional), a coluna é
idx[i] (dado). Backward, cada out[i] veio de exatamente uma célula, então

    dL/dx[i, j] = g[i]  se j == idx[i],  0 caso contrário

ou seja: uma única célula não-zero por linha, todo o resto exatamente zero.


Por que isso NÃO é o risco do Embedding
----------------------------------------
No Embedding o backward escreve nas células (idx[b,t], :) — a LINHA é escolhida
pelo dado. Dois tokens iguais no batch escrevem na mesma linha, e o += com
índice repetido do NumPy é bufferizado (lê tudo, soma, escreve tudo), então das
duas escritas só a última sobrevive. Daí o np.add.at.

Aqui a colisão é impossível, e a prova não depende de nada sobre idx:

    Sejam i ≠ j duas posições. As células escritas são (i, idx[i]) e (j, idx[j]).
    Um par ordenado só é igual a outro se AMBAS as coordenadas forem iguais.
    A primeira coordenada é i e j, e i ≠ j por hipótese. Logo os pares diferem.

Dito de outro jeito: a aplicação i ↦ (i, idx[i]) tem inversa à esquerda (a
projeção na primeira coordenada devolve i), e toda aplicação com inversa à
esquerda é injetiva. A injetividade vem só da primeira coordenada — idx pode
ser o que for, inclusive constante. Como cada célula é escrita no máximo uma
vez por chamada, a bufferização não tem o que perder, e o += simples basta.

O contraste com o Embedding é estrutural, não estatístico: lá as duas
coordenadas do endereço vêm do dado (na verdade a linha inteira), aqui uma
delas é arange(N). É por isso que classes-alvo repetidas — que num batch real
acontecem o tempo todo, o token mais comum aparecendo em dezenas de posições —
não são um problema aqui, e o teste (4) prova isso positivamente: linhas
diferentes com o MESMO alvo recebem cada uma o seu próprio gradiente, em vez de
uma delas receber a soma das duas.

Mas += e não =
--------------
"Sem add.at" não quer dizer atribuição. self.grad NÃO é zero em geral: se x for
lido por outro consumidor além do pick, já chega com gradiente acumulado, e um
`=` apagaria essa contribuição. É a mesma categoria do x + x do M0, só que a
soma acontece ENTRE ops e não dentro de uma. Então:

    self.grad[rows, idx] += out.grad   # += pelo multi-consumidor,
                                       # bufferizado basta pela injetividade

As duas metades justificam coisas diferentes e um teste separado cobre cada
uma: o de alvos repetidos (4) prova que não há colisão dentro da chamada, e o
de consumidor duplo (5) prova que o += não é decorativo — sem ele a rota do
outro consumidor some.


Por que N ≠ V em todos os testes deste ciclo
---------------------------------------------
O bug clássico da indexação é trocar a ordem dos eixos: x[idx, arange(N)] em
vez de x[arange(N), idx]. Com N == V a troca continua dentro dos limites do
array e devolve um (N,) perfeitamente válido — forma certa, valores errados —
então qualquer teste que só olhe forma passa em silêncio. Com N ≠ V (aqui
N=5, V=8) a versão trocada indexa a primeira dimensão com valores até V-1=7
num eixo de tamanho 5 e estoura IndexError na hora.

Escolher N ≠ V é o que faz a diferença entre um teste que precisa comparar
valores pra pegar o bug e um em que o bug não consegue nem rodar.
"""
import math

import numpy as np
import pytest
from core.tensor import Tensor

EPS = 1e-6
TOL = 1e-6


def numerical_grads(f, arrays):
    """Central-difference grad of scalar f(*arrays) wrt every element of each array."""
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            orig = a[idx]
            a[idx] = orig + EPS
            plus = f(*arrays)
            a[idx] = orig - EPS
            minus = f(*arrays)
            a[idx] = orig
            g[idx] = (plus - minus) / (2 * EPS)
        grads.append(g)
    return grads


def analytical_grads(build, arrays):
    """Build a Tensor graph, backward from the scalar output, return each .grad."""
    tensors = [Tensor(a.copy()) for a in arrays]
    build(*tensors).backward()
    return [t.grad for t in tensors]


def gradcheck(f, build, arrays):
    """f: numpy scalar fn. build: same fn over Tensors. arrays: float64 inputs."""
    expected = numerical_grads(f, arrays)
    got = analytical_grads(build, arrays)
    for a, e, g in zip(arrays, expected, got):
        assert g.shape == a.shape, f"grad shape {g.shape} != input shape {a.shape}"
        assert np.allclose(g, e, atol=TOL, rtol=1e-5), f"\nexpected\n{e}\ngot\n{g}"


def rand(*shape, seed=0):
    """Deterministic normal array.

    The generator is re-seeded on every call, so the same (shape, seed) always
    returns the SAME values — which keeps tests order-independent, but means
    two calls with the same shape produce IDENTICAL arrays.

    Pass distinct seeds whenever same-shaped arrays play different roles (q/k/v,
    a LayerNorm's gamma and beta, the four attention projections). A mutation
    run found this the hard way: with q == k == v, swapping two of them is
    invisible to any forward comparison.
    """
    return np.random.default_rng(seed).standard_normal(shape)


def rand_pos(*shape, seed=0):
    """Strictly positive inputs, for ops whose domain excludes 0 (sqrt, 1/b)."""
    return np.abs(rand(*shape, seed=seed)) + 0.5


# ---- matmul ----

def test_matmul_gradient():
    a, b = rand(2, 3), rand(3, 4)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_chained_with_mul():
    """Upstream grad is not all-ones — catches a backward that ignores out.grad."""
    a, b = rand(3, 2), rand(2, 3)
    gradcheck(lambda x, y: ((x @ y) * (x @ y)).sum(),
              lambda x, y: ((x @ y) * (x @ y)).sum(), [a, b])


def test_matmul_batched_same_batch():
    a, b = rand(4, 2, 3), rand(4, 3, 5)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_batched_broadcasts_weight():
    """(B,N,K) @ (K,M) — the shape a Linear layer actually sees. dW must sum over B."""
    a, b = rand(4, 2, 3), rand(3, 5)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_two_batch_dims_q_at_k_transposed():
    """(B,H,T,dk) @ (B,H,dk,T) — Q@K^T in multi-head. B=2, H=3 so the two batch
    axes have different lengths and cannot be silently swapped for each other."""
    a, b = rand(2, 3, 5, 8), rand(2, 3, 8, 5)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_two_batch_dims_attn_at_v():
    """(B,H,T,T) @ (B,H,T,dk) — attn@V."""
    a, b = rand(2, 3, 5, 5), rand(2, 3, 5, 8)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_two_batch_dims_nonuniform_upstream():
    a, b = rand(2, 3, 5, 8), rand(2, 3, 8, 5)
    w = rand(2, 3, 5, 5) + 2.0
    gradcheck(lambda x, y: (w * (x @ y)).sum(),
              lambda x, y: ((x @ y) * Tensor(w)).sum(), [a, b])


def test_matmul_two_batch_dims_grad_shapes():
    a, b = Tensor(rand(2, 3, 5, 8)), Tensor(rand(2, 3, 8, 5))
    (a @ b).sum().backward()
    assert a.grad.shape == (2, 3, 5, 8)
    assert b.grad.shape == (2, 3, 8, 5)


def test_matmul_weight_shared_across_two_batch_dims():
    """(B,H,T,dk) @ (dk,M): dW must reduce over BOTH leading axes, not just one."""
    a, b = rand(2, 3, 5, 8), rand(8, 4)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_broadcasts_a_single_batch_dim():
    """(B,1,T,dk) @ (B,H,dk,M): the head axis stretches, so da reduces over it
    with keepdims while the batch axis stays put."""
    a, b = rand(2, 1, 5, 8), rand(2, 3, 8, 4)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_three_batch_dims():
    a, b = rand(2, 3, 4, 5, 6), rand(2, 3, 4, 6, 7)
    gradcheck(lambda x, y: (x @ y).sum(),
              lambda x, y: (x @ y).sum(), [a, b])


# ---- broadcasting ----

def test_add_broadcast_row_vector():
    """(2,3) + (3,) — grad of the bias must be summed over the batch axis."""
    a, b = rand(2, 3), rand(3)
    gradcheck(lambda x, y: (x + y).sum(),
              lambda x, y: (x + y).sum(), [a, b])


def test_mul_broadcast_outer():
    """(2,1) * (1,3) — both sides broadcast, both grads need a different reduction."""
    a, b = rand(2, 1), rand(1, 3)
    gradcheck(lambda x, y: (x * y).sum(),
              lambda x, y: (x * y).sum(), [a, b])


def test_add_broadcast_keeps_singleton_dim():
    """(4,3) + (1,3): grad must be (1,3), not (3,) — sum over axis, keepdims."""
    a, b = rand(4, 3), rand(1, 3)
    gradcheck(lambda x, y: (x + y).sum(),
              lambda x, y: (x + y).sum(), [a, b])


def test_mul_broadcast_scalar_tensor():
    a, b = rand(2, 3), rand(1)
    gradcheck(lambda x, y: (x * y).sum(),
              lambda x, y: (x * y).sum(), [a, b])


def test_broadcast_extra_leading_dims():
    """(2,3,4) + (4,) — grad reduces over two axes at once."""
    a, b = rand(2, 3, 4), rand(4)
    gradcheck(lambda x, y: (x + y).sum(),
              lambda x, y: (x + y).sum(), [a, b])


# ---- the two together: a Linear layer forward ----

def test_linear_layer_gradient():
    """(x @ W + b).sum() — matmul and broadcast in one graph, like M2 will need."""
    x, W, b = rand(5, 3), rand(3, 4), rand(4)
    gradcheck(lambda x, W, b: (x @ W + b).sum(),
              lambda x, W, b: (x @ W + b).sum(), [x, W, b])


def test_grad_accumulates_when_tensor_used_twice():
    """Same bug as M0, now shaped: x used twice must sum gradients."""
    x = Tensor(np.ones((2, 3)))
    (x + x).sum().backward()
    assert np.allclose(x.grad, 2.0)


def test_backward_requires_scalar_output():
    with pytest.raises(Exception):
        Tensor(np.ones((2, 3))).backward()


# ---- mean ----

def test_mean_forward_matches_numpy():
    a = rand(4, 3)
    assert np.allclose(Tensor(a).mean(axis=0).data, a.mean(axis=0))
    assert np.allclose(Tensor(a).mean(axis=1, keepdims=True).data,
                       a.mean(axis=1, keepdims=True))
    assert np.allclose(Tensor(a).mean().data, a.mean())


def test_mean_output_shape():
    a = Tensor(rand(4, 3))
    assert a.mean(axis=0).shape == (3,)
    assert a.mean(axis=0, keepdims=True).shape == (1, 3)
    assert a.mean(axis=1).shape == (4,)
    assert a.mean().shape == ()


def test_mean_gradcheck_axis0():
    a = rand(4, 3)
    gradcheck(lambda x: x.mean(axis=0).sum(),
              lambda x: x.mean(axis=0).sum(), [a])


def test_mean_gradcheck_axis1():
    a = rand(4, 3)
    gradcheck(lambda x: x.mean(axis=1).sum(),
              lambda x: x.mean(axis=1).sum(), [a])


def test_mean_gradcheck_all_axes():
    a = rand(4, 3)
    gradcheck(lambda x: x.mean().sum(),
              lambda x: x.mean().sum(), [a])


def test_mean_gradcheck_keepdims():
    a = rand(4, 3)
    gradcheck(lambda x: x.mean(axis=1, keepdims=True).sum(),
              lambda x: x.mean(axis=1, keepdims=True).sum(), [a])


def test_mean_gradcheck_middle_axis_3d():
    """Non-trivial axis on a 3-D input — the case keepdims=False gets wrong."""
    a = rand(2, 5, 3)
    gradcheck(lambda x: x.mean(axis=1).sum(),
              lambda x: x.mean(axis=1).sum(), [a])


def test_mean_gradcheck_last_axis_3d():
    a = rand(2, 5, 3)
    gradcheck(lambda x: x.mean(axis=-1).sum(),
              lambda x: x.mean(axis=-1).sum(), [a])


def test_mean_gradcheck_nonuniform_upstream():
    """Upstream grad is not all-ones, so a wrong 1/N factor cannot hide."""
    a = rand(4, 3)
    gradcheck(lambda x: (x.mean(axis=0) * x.mean(axis=0)).sum(),
              lambda x: (x.mean(axis=0) * x.mean(axis=0)).sum(), [a])


def test_mean_gradcheck_asymmetric_shape():
    """Rows and cols differ in length: a 1/N with the wrong N shows up here."""
    a = rand(2, 7)
    gradcheck(lambda x: x.mean(axis=1).sum(),
              lambda x: x.mean(axis=1).sum(), [a])


def test_mean_grad_is_exactly_one_over_n():
    """Closed form: every element gets 1/N of its group's upstream grad."""
    x = Tensor(rand(4, 3))
    x.mean(axis=0).sum().backward()
    assert np.allclose(x.grad, 1.0 / 4)


# ---- var (ddof=0, the LayerNorm convention) ----

def test_var_forward_matches_numpy():
    a = rand(4, 3)
    assert np.allclose(Tensor(a).var(axis=0).data, a.var(axis=0))
    assert np.allclose(Tensor(a).var(axis=1, keepdims=True).data,
                       a.var(axis=1, keepdims=True))
    assert np.allclose(Tensor(a).var().data, a.var())


def test_var_output_shape():
    a = Tensor(rand(4, 3))
    assert a.var(axis=0).shape == (3,)
    assert a.var(axis=0, keepdims=True).shape == (1, 3)
    assert a.var(axis=1).shape == (4,)
    assert a.var().shape == ()


def test_var_of_constant_is_zero():
    assert np.allclose(Tensor(np.full((4, 3), 2.5)).var(axis=0).data, 0.0)


def test_var_gradcheck_axis0():
    a = rand(4, 3)
    gradcheck(lambda x: x.var(axis=0).sum(),
              lambda x: x.var(axis=0).sum(), [a])


def test_var_gradcheck_axis1():
    a = rand(4, 3)
    gradcheck(lambda x: x.var(axis=1).sum(),
              lambda x: x.var(axis=1).sum(), [a])


def test_var_gradcheck_all_axes():
    a = rand(4, 3)
    gradcheck(lambda x: x.var().sum(),
              lambda x: x.var().sum(), [a])


def test_var_gradcheck_keepdims():
    a = rand(4, 3)
    gradcheck(lambda x: x.var(axis=1, keepdims=True).sum(),
              lambda x: x.var(axis=1, keepdims=True).sum(), [a])


def test_var_gradcheck_middle_axis_3d():
    a = rand(2, 5, 3)
    gradcheck(lambda x: x.var(axis=1).sum(),
              lambda x: x.var(axis=1).sum(), [a])


def test_var_gradcheck_last_axis_3d():
    """The axis LayerNorm actually normalizes over."""
    a = rand(2, 5, 3)
    gradcheck(lambda x: x.var(axis=-1).sum(),
              lambda x: x.var(axis=-1).sum(), [a])


def test_var_gradcheck_nonuniform_upstream():
    a = rand(4, 3)
    gradcheck(lambda x: (x.var(axis=0) * x.var(axis=0)).sum(),
              lambda x: (x.var(axis=0) * x.var(axis=0)).sum(), [a])


def test_var_gradcheck_asymmetric_shape():
    a = rand(2, 7)
    gradcheck(lambda x: x.var(axis=1).sum(),
              lambda x: x.var(axis=1).sum(), [a])


def test_var_gradcheck_mixed_with_mean():
    """mean and var over the same axis in one graph — x feeds both branches."""
    a = rand(4, 3)
    gradcheck(lambda x: (x.mean(axis=0) * x.var(axis=0)).sum(),
              lambda x: (x.mean(axis=0) * x.var(axis=0)).sum(), [a])


def test_var_grad_matches_closed_form():
    """dv/dx = (2/N)(x - m), the result the derivation at the top lands on.

    Note this is also what you get by (wrongly) holding m constant — the
    correction term is Sum(x_i - m) = 0. So this test does NOT prove the shared
    node was handled; it only pins the number. What guarantees the chain rule
    is that var is built from sub/mul/mean, each already gradchecked.
    """
    a = rand(4, 3)
    x = Tensor(a)
    x.var(axis=0).sum().backward()
    assert np.allclose(x.grad, (2.0 / 4) * (a - a.mean(axis=0, keepdims=True)))


def test_var_grad_sums_to_zero_over_reduced_axis():
    """Sum(x_i - m) = 0 means the gradient of var sums to zero along that axis."""
    x = Tensor(rand(4, 3))
    x.var(axis=0).sum().backward()
    assert np.allclose(x.grad.sum(axis=0), 0.0, atol=1e-12)


# ---- sqrt ----

def test_sqrt_forward_matches_numpy():
    a = rand_pos(4, 3)
    assert np.allclose(Tensor(a).sqrt().data, np.sqrt(a))


def test_sqrt_preserves_shape():
    assert Tensor(rand_pos(2, 5, 3)).sqrt().shape == (2, 5, 3)


def test_sqrt_gradcheck_2d():
    a = rand_pos(4, 3)
    gradcheck(lambda x: np.sqrt(x).sum(),
              lambda x: x.sqrt().sum(), [a])


def test_sqrt_gradcheck_3d():
    a = rand_pos(2, 5, 3)
    gradcheck(lambda x: np.sqrt(x).sum(),
              lambda x: x.sqrt().sum(), [a])


def test_sqrt_gradcheck_nonuniform_upstream():
    """Upstream grad is not ones — a missing out.grad factor shows up here."""
    a = rand_pos(4, 3)
    gradcheck(lambda x: (np.sqrt(x) * np.sqrt(x) * np.sqrt(x)).sum(),
              lambda x: (x.sqrt() * x.sqrt() * x.sqrt()).sum(), [a])


def test_sqrt_gradcheck_small_values():
    """Near 0 the derivative is large — where a wrong factor of 2 is loudest."""
    a = rand_pos(3, 4) * 0.05
    gradcheck(lambda x: np.sqrt(x).sum(),
              lambda x: x.sqrt().sum(), [a])


def test_sqrt_gradcheck_over_var_plus_eps():
    """sqrt(var(axis) + eps) — the exact denominator LayerNorm will build."""
    a = rand(2, 5, 3)
    gradcheck(lambda x: np.sqrt(x.var(axis=-1) + 1e-5).sum(),
              lambda x: (x.var(axis=-1) + 1e-5).sqrt().sum(), [a])


def test_sqrt_grad_matches_closed_form():
    """dL/dx = 1/(2*sqrt(x))."""
    a = rand_pos(4, 3)
    x = Tensor(a)
    x.sqrt().sum().backward()
    assert np.allclose(x.grad, 1.0 / (2 * np.sqrt(a)))


def test_sqrt_of_negative_is_nan_not_an_exception():
    """Documented behaviour: no domain guard here, callers add eps themselves."""
    with np.errstate(invalid="ignore"):
        assert np.isnan(Tensor(np.array([-1.0])).sqrt().data).all()


# ---- truediv ----

def test_div_forward_matches_numpy():
    a, b = rand(4, 3), rand_pos(4, 3)
    assert np.allclose((Tensor(a) / Tensor(b)).data, a / b)


def test_div_forward_broadcasts():
    a, b = rand(4, 3), rand_pos(4, 1)
    assert np.allclose((Tensor(a) / Tensor(b)).data, a / b)


def test_div_by_python_float():
    a = rand(4, 3)
    assert np.allclose((Tensor(a) / 2.0).data, a / 2.0)


def test_div_gradcheck_same_shape():
    a, b = rand(4, 3), rand_pos(4, 3)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_broadcast_column():
    """(4,3) / (4,1) — the LayerNorm shape: db must reduce back to (4,1)."""
    a, b = rand(4, 3), rand_pos(4, 1)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_broadcast_row():
    a, b = rand(4, 3), rand_pos(3)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_broadcast_numerator():
    """(4,1) / (4,3) — the NUMERATOR broadcasts here, so da is what must reduce.

    Added after a mutation check: every other div test puts the full-shape
    operand on top, which makes the _unbroadcast on da a no-op and lets a
    missing call pass silently.
    """
    a, b = rand(4, 1), rand_pos(4, 3)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_broadcast_both_sides():
    """(4,1) / (1,3) — both operands stretch, each grad reduces differently."""
    a, b = rand(4, 1), rand_pos(1, 3)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_numerator_row_vector():
    a, b = rand(3), rand_pos(4, 3)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_broadcast_numerator_grad_keeps_input_shape():
    a, b = Tensor(rand(2, 5, 1)), Tensor(rand_pos(2, 5, 3))
    (a / b).sum().backward()
    assert a.grad.shape == (2, 5, 1)
    assert b.grad.shape == (2, 5, 3)


def test_div_gradcheck_broadcast_scalar_denominator():
    a, b = rand(4, 3), rand_pos(1)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_3d_last_axis():
    """(2,5,3) / (2,5,1) — exactly how LayerNorm divides by the std."""
    a, b = rand(2, 5, 3), rand_pos(2, 5, 1)
    gradcheck(lambda a, b: (a / b).sum(),
              lambda a, b: (a / b).sum(), [a, b])


def test_div_gradcheck_nonuniform_upstream():
    a, b = rand(4, 3), rand_pos(4, 3)
    gradcheck(lambda a, b: ((a / b) * (a / b)).sum(),
              lambda a, b: ((a / b) * (a / b)).sum(), [a, b])


def test_div_gradcheck_same_tensor_both_sides():
    """x / x — the shared node must accumulate from both branches."""
    a = rand_pos(4, 3)
    gradcheck(lambda x: (x / (x + 1.0)).sum(),
              lambda x: (x / (x + 1.0)).sum(), [a])


def test_div_grad_matches_closed_form():
    """dL/da = 1/b, dL/db = -a/b^2."""
    a_np, b_np = rand(4, 3), rand_pos(4, 3)
    a, b = Tensor(a_np), Tensor(b_np)
    (a / b).sum().backward()
    assert np.allclose(a.grad, 1.0 / b_np)
    assert np.allclose(b.grad, -a_np / b_np ** 2)


def test_div_denominator_grad_is_negative_for_positive_inputs():
    """Sign check on its own: bigger b means smaller a/b."""
    a, b = Tensor(rand_pos(4, 3)), Tensor(rand_pos(4, 3))
    (a / b).sum().backward()
    assert np.all(b.grad < 0)


def test_div_broadcast_grads_keep_input_shapes():
    a, b = Tensor(rand(2, 5, 3)), Tensor(rand_pos(2, 5, 1))
    (a / b).sum().backward()
    assert a.grad.shape == (2, 5, 3)
    assert b.grad.shape == (2, 5, 1)


# ---- tanh ----

def test_tanh_forward_matches_numpy():
    a = rand(4, 5)
    assert np.allclose(Tensor(a).tanh().data, np.tanh(a))


def test_tanh_preserves_shape():
    assert Tensor(rand(2, 3, 4)).tanh().shape == (2, 3, 4)


def test_tanh_output_is_bounded():
    a = np.array([[-50.0, -1.0, 0.0, 1.0, 50.0]])
    out = Tensor(a).tanh().data
    assert np.all(out > -1.0) and np.all(out < 1.0) or np.all(np.abs(out) <= 1.0)


# (1) near zero, non-uniform upstream

def test_tanh_gradcheck_near_zero():
    a, w = rand(4, 5) * 0.1, rand(4, 5) + 2.0
    gradcheck(lambda x: (w * np.tanh(x)).sum(),
              lambda x: (x.tanh() * Tensor(w)).sum(), [a])


def test_tanh_gradcheck_moderate_values():
    a, w = rand(4, 5), rand(4, 5) + 2.0
    gradcheck(lambda x: (w * np.tanh(x)).sum(),
              lambda x: (x.tanh() * Tensor(w)).sum(), [a])


def test_tanh_gradcheck_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * np.tanh(x)).sum(),
              lambda x: (x.tanh() * Tensor(w)).sum(), [a])


# (2) the saturated region

def test_tanh_gradcheck_at_plus_minus_five():
    """|x|=5: derivative ~1.8e-4, still above the harness atol, so a wrong
    formula here really does fail the comparison."""
    a = np.array([[-5.0, -5.0, 5.0, 5.0]])
    w = np.array([[1.0, 2.0, 3.0, 4.0]])
    gradcheck(lambda x: (w * np.tanh(x)).sum(),
              lambda x: (x.tanh() * Tensor(w)).sum(), [a])


def test_tanh_saturated_gradient_matches_closed_form_relatively():
    """|x|=10: the true derivative (~8.2e-9) is far below the gradcheck atol,
    so allclose there would compare 'small' with 'small' and pass on anything.
    Compared against 1-t^2 with a RELATIVE tolerance instead."""
    a = np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
    x = Tensor(a)
    x.tanh().sum().backward()

    expected = 1.0 - np.tanh(a) ** 2
    assert np.all(np.isfinite(x.grad))
    assert np.allclose(x.grad, expected, rtol=1e-12, atol=0.0)


def test_tanh_saturated_gradient_is_small_but_strictly_positive():
    x = Tensor(np.array([-10.0, 10.0]))
    x.tanh().sum().backward()
    assert np.all(x.grad > 0.0), "saturated gradient collapsed to zero"
    assert np.all(x.grad < 1e-7)


def test_tanh_gradient_shrinks_as_x_grows():
    x = Tensor(np.array([0.0, 1.0, 2.0, 5.0, 10.0]))
    x.tanh().sum().backward()
    assert np.all(np.diff(x.grad) < 0.0)


def test_tanh_extreme_values_stay_finite():
    """x = +-700 saturates completely; the gradient must be 0, never nan."""
    x = Tensor(np.array([-700.0, 700.0]))
    x.tanh().sum().backward()
    assert np.all(np.isfinite(x.tanh().data))
    assert np.all(np.isfinite(x.grad))


# (3) the exact edge case at zero

def test_tanh_at_exactly_zero():
    """t=0 there, so the gradient is exactly 1 — no tolerance needed."""
    x = Tensor(np.array([0.0]))
    out = x.tanh()
    out.sum().backward()
    assert out.data[0] == 0.0
    assert x.grad[0] == 1.0


def test_tanh_gradient_at_zero_with_nonuniform_upstream():
    """Same point, upstream 3.0: the gradient must be exactly 3.0, not 1.0."""
    x = Tensor(np.array([0.0, 0.0]))
    (x.tanh() * Tensor(np.array([3.0, -2.0]))).sum().backward()
    assert x.grad[0] == 3.0
    assert x.grad[1] == -2.0


def test_tanh_is_odd():
    a = rand(4, 5)
    assert np.allclose(Tensor(a).tanh().data, -Tensor(-a).tanh().data)


def test_tanh_chained_with_other_ops_gradcheck():
    a, w = rand(4, 5), rand(4, 5) + 2.0
    gradcheck(lambda x: (w * np.tanh(x * x + 1.0)).sum(),
              lambda x: ((x * x + 1.0).tanh() * Tensor(w)).sum(), [a])


# ---- gelu (tanh approximation) ----

def gelu_np(x):
    c1 = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c1 * (x + 0.044715 * x ** 3)))


# torch 2.7.1, F.gelu(torch.tensor(x, dtype=torch.float64), approximate="tanh").
# float64 explicit — see the tolerance analysis in the module docstring.
GELU_REFERENCE = [
    (-8.0, -0.0),
    (-5.0, -2.2917961972623857e-07),
    (-2.0, -0.04540230591222494),
    (-1.0, -0.1588080093917233),
    (-0.5, -0.15428599017485606),
    (-0.1, -0.04601724895456484),
    (0.0, 0.0),
    (0.1, 0.053982751045435165),
    (0.5, 0.34571400982514394),
    (1.0, 0.8411919906082768),
    (2.0, 1.954597694087775),
    (5.0, 4.999999770820381),
    (8.0, 8.0),
]

# Per-point tolerance, derived from the cancellation analysis, not fitted:
#   exact  -> the saturated ends and x=0, where both sides are exactly equal
#   1e-8   -> x=-5, where 1/(1+tanh) amplifies one ULP to ~2.4e-9
#   1e-14  -> everywhere else, ~45 ULPs for a 6-op chain plus a foreign tanh
GELU_TOLERANCE = {-8.0: 0.0, 0.0: 0.0, 8.0: 0.0, -5.0: 1e-8}


# (2) against the PyTorch reference

@pytest.mark.parametrize("x0,expected", GELU_REFERENCE)
def test_gelu_matches_pytorch_reference(x0, expected):
    got = Tensor(np.array([x0])).gelu().data[0]
    rtol = GELU_TOLERANCE.get(x0, 1e-14)
    if rtol == 0.0:
        assert got == expected, f"expected exact agreement at x={x0}"
    else:
        assert got == pytest.approx(expected, rel=rtol)


def test_gelu_reference_is_not_the_erf_variant():
    """Guard: the tanh approximation and the exact erf GELU differ measurably,
    so the reference above cannot be silently satisfied by the wrong one."""
    x = 1.0
    erf_gelu = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    tanh_ref = dict(GELU_REFERENCE)[1.0]
    assert abs(erf_gelu - tanh_ref) > 1e-6


def test_gelu_forward_matches_local_reference_on_an_array():
    a = rand(4, 5)
    assert np.allclose(Tensor(a).gelu().data, gelu_np(a), rtol=1e-14, atol=0.0)


# (3) exact behaviour at zero

def test_gelu_at_exactly_zero():
    """The leading x factor zeroes the whole product — exactly, no tolerance."""
    x = Tensor(np.array([0.0]))
    out = x.gelu()
    out.sum().backward()
    assert out.data[0] == 0.0
    # d/dx = 0.5*(1+tanh(0)) + 0.5*0*(...) = 0.5 exactly
    assert x.grad[0] == 0.5


def test_gelu_saturates_to_identity_for_large_positive():
    assert Tensor(np.array([8.0, 20.0])).gelu().data.tolist() == [8.0, 20.0]


def test_gelu_saturates_to_zero_for_large_negative():
    out = Tensor(np.array([-8.0, -20.0])).gelu().data
    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


def test_gelu_is_not_relu_for_small_negatives():
    """GELU passes a little negative signal through; ReLU would give exactly 0."""
    out = Tensor(np.array([-0.5])).gelu().data[0]
    assert out < 0.0 and out > -0.2


# (1) gradcheck

def test_gelu_gradcheck_moderate_values():
    a, w = rand(4, 5), rand(4, 5) + 2.0
    gradcheck(lambda x: (w * gelu_np(x)).sum(),
              lambda x: (x.gelu() * Tensor(w)).sum(), [a])


def test_gelu_gradcheck_near_zero():
    a, w = rand(4, 5) * 0.1, rand(4, 5) + 2.0
    gradcheck(lambda x: (w * gelu_np(x)).sum(),
              lambda x: (x.gelu() * Tensor(w)).sum(), [a])


def test_gelu_gradcheck_negative_region():
    """Where GELU is non-monotonic — the minimum sits near x = -0.75."""
    a = np.array([[-2.0, -1.5, -1.0, -0.75, -0.5]])
    w = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    gradcheck(lambda x: (w * gelu_np(x)).sum(),
              lambda x: (x.gelu() * Tensor(w)).sum(), [a])


def test_gelu_gradcheck_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * gelu_np(x)).sum(),
              lambda x: (x.gelu() * Tensor(w)).sum(), [a])


def test_gelu_gradient_is_negative_somewhere():
    """GELU dips below zero and comes back, so its derivative changes sign.
    A monotone stand-in (relu, sigmoid-times-x done wrong) would not."""
    x = Tensor(np.array([-1.5, -0.5, 1.0]))
    x.gelu().sum().backward()
    assert x.grad[0] < 0.0
    assert x.grad[2] > 0.0


def test_gelu_preserves_shape():
    assert Tensor(rand(2, 3, 4)).gelu().shape == (2, 3, 4)


def test_gelu_after_a_matmul_gradcheck():
    """The shape it will actually be used in: MLP hidden activation."""
    a, b = rand(4, 3), rand(3, 6)
    w = rand(4, 6) + 2.0
    gradcheck(lambda a, b: (w * gelu_np(a @ b)).sum(),
              lambda a, b: ((a @ b).gelu() * Tensor(w)).sum(), [a, b])


# ---- log_softmax ----

def log_softmax_np(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def naive_log_softmax_np(x, axis=-1):
    """log(softmax(x)) done the obvious way — the version that breaks."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.exp(x)
        return np.log(e / e.sum(axis=axis, keepdims=True))


# (1) forward where the naive version still works

def test_log_softmax_matches_naive_on_moderate_values():
    a = rand(4, 5)
    got = Tensor(a).log_softmax(axis=-1).data
    assert np.allclose(got, naive_log_softmax_np(a), rtol=1e-13, atol=0.0)
    assert np.allclose(got, np.log(softmax_np(a, -1)), rtol=1e-13, atol=0.0)


def test_log_softmax_forward_matches_reference_on_other_axes():
    a = rand(2, 3, 4)
    for axis in (0, 1, -1):
        assert np.allclose(Tensor(a).log_softmax(axis=axis).data,
                           log_softmax_np(a, axis), rtol=1e-13, atol=0.0)


def test_log_softmax_preserves_shape():
    assert Tensor(rand(2, 3, 4)).log_softmax(axis=-1).shape == (2, 3, 4)


def test_log_softmax_is_softmax_then_log():
    a = rand(4, 5)
    assert np.allclose(np.exp(Tensor(a).log_softmax(axis=-1).data),
                       Tensor(a).softmax(axis=-1).data, rtol=1e-13, atol=0.0)


# (2) stability, where the naive version does not survive

def test_naive_log_softmax_really_does_break():
    """Pins the premise: without this, the stability tests prove nothing."""
    a = np.array([[0.0, 1000.0]])
    naive = naive_log_softmax_np(a)
    assert not np.all(np.isfinite(naive)), "naive was expected to blow up here"


def test_log_softmax_survives_a_huge_spread():
    """log(softmax) underflows the small entry to 0 and then logs it: -inf.
    log_softmax keeps the true value, -1000."""
    a = np.array([[0.0, 1000.0]])
    got = Tensor(a).log_softmax(axis=-1).data
    assert np.all(np.isfinite(got))
    assert got[0, 0] == pytest.approx(-1000.0, abs=1e-9)
    assert got[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_log_softmax_survives_huge_magnitudes():
    a = np.array([[1000.0, 1001.0, 999.0], [1e4, 1e4, 1e4]])
    got = Tensor(a).log_softmax(axis=-1).data
    assert np.all(np.isfinite(got))
    assert np.allclose(got, log_softmax_np(a), rtol=1e-13, atol=0.0)


def test_log_softmax_survives_large_negative_values():
    a = np.array([[-1000.0, -1001.0, -2000.0]])
    got = Tensor(a).log_softmax(axis=-1).data
    assert np.all(np.isfinite(got))
    assert np.allclose(got, log_softmax_np(a), rtol=1e-13, atol=0.0)


def test_log_softmax_gradient_is_finite_on_huge_values():
    x = Tensor(np.array([[0.0, 1000.0]]))
    (x.log_softmax(axis=-1) * Tensor(np.array([[1.0, 3.0]]))).sum().backward()
    assert np.all(np.isfinite(x.grad))


def test_log_softmax_is_shift_invariant():
    """The cancellation proof, checked numerically on the op itself."""
    a = rand(4, 5)
    assert np.allclose(Tensor(a).log_softmax(axis=-1).data,
                       Tensor(a + 500.0).log_softmax(axis=-1).data,
                       rtol=1e-13, atol=0.0)


# (4) each row is a proper log-distribution

def test_log_softmax_rows_exponentiate_to_one():
    out = Tensor(rand(4, 5)).log_softmax(axis=-1).data
    assert np.allclose(np.exp(out).sum(axis=-1), 1.0, rtol=0.0, atol=1e-15)


def test_log_softmax_rows_sum_to_one_on_other_axes():
    a = rand(2, 3, 4)
    for axis in (0, 1, -1):
        out = Tensor(a).log_softmax(axis=axis).data
        assert np.allclose(np.exp(out).sum(axis=axis), 1.0, atol=1e-15)


def test_log_softmax_rows_exponentiate_to_one_even_at_huge_values():
    out = Tensor(np.array([[0.0, 1000.0], [1e4, 1e4]])).log_softmax(axis=-1).data
    assert np.allclose(np.exp(out).sum(axis=-1), 1.0, atol=1e-15)


def test_log_softmax_output_is_never_positive():
    """A log-probability cannot exceed log(1) = 0."""
    assert np.all(Tensor(rand(4, 5)).log_softmax(axis=-1).data <= 0.0)


# (3) gradcheck with non-uniform upstream

def test_log_softmax_gradcheck_last_axis():
    a, w = rand(4, 5), rand(4, 5, seed=1) + 2.0
    gradcheck(lambda x: (w * log_softmax_np(x, -1)).sum(),
              lambda x: (x.log_softmax(axis=-1) * Tensor(w)).sum(), [a])


def test_log_softmax_gradcheck_axis0():
    a, w = rand(4, 5), rand(4, 5, seed=1) + 2.0
    gradcheck(lambda x: (w * log_softmax_np(x, 0)).sum(),
              lambda x: (x.log_softmax(axis=0) * Tensor(w)).sum(), [a])


def test_log_softmax_gradcheck_middle_axis_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4, seed=1) + 2.0
    gradcheck(lambda x: (w * log_softmax_np(x, 1)).sum(),
              lambda x: (x.log_softmax(axis=1) * Tensor(w)).sum(), [a])


def test_log_softmax_gradcheck_on_a_wide_spread():
    """Values where the naive forward already produced -inf, so only the
    stable implementation can be gradchecked here at all.

    The entries have to go below about -745: that is where exp underflows to
    exactly 0, softmax returns a hard zero, and log(0) is -inf. A merely wide
    spread is not enough — exp(-80) is still a perfectly ordinary float64,
    which is why an earlier version of this test asserted a premise that was
    simply false.
    """
    a = np.array([[0.0, -800.0, 3.0], [5.0, -900.0, 20.0]])
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert not np.all(np.isfinite(naive_log_softmax_np(a))), "premise: naive breaks"
    gradcheck(lambda x: (w * log_softmax_np(x, -1)).sum(),
              lambda x: (x.log_softmax(axis=-1) * Tensor(w)).sum(), [a])


def test_log_softmax_gradcheck_after_a_matmul():
    a, b = rand(4, 3), rand(3, 6)
    w = rand(4, 6, seed=1) + 2.0
    gradcheck(lambda a, b: (w * log_softmax_np(a @ b, -1)).sum(),
              lambda a, b: ((a @ b).log_softmax(axis=-1) * Tensor(w)).sum(), [a, b])


def test_log_softmax_gradient_matches_closed_form():
    """dL/dx = g - softmax(x) * sum(g)."""
    a = rand(4, 5)
    w = rand(4, 5, seed=1) + 2.0
    x = Tensor(a)
    (x.log_softmax(axis=-1) * Tensor(w)).sum().backward()
    expected = w - softmax_np(a, -1) * w.sum(axis=-1, keepdims=True)
    assert np.allclose(x.grad, expected, rtol=1e-12, atol=0.0)


def test_log_softmax_uniform_upstream_gives_the_softmax_complement():
    """With g = 1 everywhere the gradient is 1 - N*softmax(x)."""
    a = rand(3, 4)
    x = Tensor(a)
    x.log_softmax(axis=-1).sum().backward()
    assert np.allclose(x.grad, 1.0 - 4 * softmax_np(a, -1), rtol=1e-12, atol=0.0)


# ---- reshape ----

def test_reshape_forward_matches_numpy():
    a = rand(2, 3, 4)
    assert np.allclose(Tensor(a).reshape(6, 4).data, a.reshape(6, 4))


def test_reshape_accepts_a_tuple():
    assert Tensor(rand(2, 3, 4)).reshape((6, 4)).shape == (6, 4)


def test_reshape_infers_minus_one():
    assert Tensor(rand(2, 3, 4)).reshape(6, -1).shape == (6, 4)


def test_reshape_flatten_gradcheck():
    a, w = rand(2, 3, 4), rand(24) + 2.0
    gradcheck(lambda x: (w * x.reshape(24)).sum(),
              lambda x: (x.reshape(24) * Tensor(w)).sum(), [a])


def test_reshape_unflatten_gradcheck():
    a, w = rand(24), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * x.reshape(2, 3, 4)).sum(),
              lambda x: (x.reshape(2, 3, 4) * Tensor(w)).sum(), [a])


def test_reshape_regroup_gradcheck():
    a, w = rand(4, 6), rand(6, 4) + 2.0
    gradcheck(lambda x: (w * x.reshape(6, 4)).sum(),
              lambda x: (x.reshape(6, 4) * Tensor(w)).sum(), [a])


def test_reshape_split_heads_gradcheck():
    """(B,T,H*D) -> (B,T,H,D), the first half of a head split."""
    a, w = rand(2, 5, 12), rand(2, 5, 3, 4) + 2.0
    gradcheck(lambda x: (w * x.reshape(2, 5, 3, 4)).sum(),
              lambda x: (x.reshape(2, 5, 3, 4) * Tensor(w)).sum(), [a])


def test_reshape_grad_has_the_original_shape():
    x = Tensor(rand(2, 3, 4))
    (x.reshape(24) * Tensor(rand(24) + 2.0)).sum().backward()
    assert x.grad.shape == (2, 3, 4)


def test_reshape_roundtrip_is_identity_on_the_gradient():
    x = Tensor(rand(2, 3, 4))
    w = rand(2, 3, 4) + 2.0
    (x.reshape(24).reshape(2, 3, 4) * Tensor(w)).sum().backward()
    assert np.allclose(x.grad, w)


def test_reshape_rejects_incompatible_size():
    with pytest.raises(Exception):
        Tensor(rand(2, 3, 4)).reshape(5, 5)


# ---- transpose ----

def test_transpose_forward_matches_numpy():
    a = rand(2, 3, 4)
    assert np.allclose(Tensor(a).transpose(1, 2, 0).data, a.transpose(1, 2, 0))


def test_transpose_default_reverses_all_axes():
    a = rand(2, 3, 4)
    assert np.allclose(Tensor(a).transpose().data, a.T)


def test_transpose_swap_2d_gradcheck():
    """Involutive: argsort((1,0)) == (1,0). Passes even with a wrong backward."""
    a, w = rand(4, 6), rand(6, 4) + 2.0
    gradcheck(lambda x: (w * x.transpose(1, 0)).sum(),
              lambda x: (x.transpose(1, 0) * Tensor(w)).sum(), [a])


def test_transpose_head_split_gradcheck():
    """(B,T,H,D) -> (B,H,T,D). Also involutive — same blind spot as the 2-D swap."""
    a, w = rand(2, 5, 3, 4), rand(2, 3, 5, 4) + 2.0
    gradcheck(lambda x: (w * x.transpose(0, 2, 1, 3)).sum(),
              lambda x: (x.transpose(0, 2, 1, 3) * Tensor(w)).sum(), [a])


def test_transpose_3cycle_gradcheck():
    """(1,2,0) — inverse is (2,0,1), NOT itself. The case that proves inversion.

    Distinct dims, so reusing the forward axes in the backward raises a shape
    error rather than returning quiet nonsense.
    """
    a, w = rand(2, 3, 4), rand(3, 4, 2) + 2.0
    gradcheck(lambda x: (w * x.transpose(1, 2, 0)).sum(),
              lambda x: (x.transpose(1, 2, 0) * Tensor(w)).sum(), [a])


def test_transpose_3cycle_other_direction_gradcheck():
    a, w = rand(2, 3, 4), rand(4, 2, 3) + 2.0
    gradcheck(lambda x: (w * x.transpose(2, 0, 1)).sum(),
              lambda x: (x.transpose(2, 0, 1) * Tensor(w)).sum(), [a])


def test_transpose_3cycle_on_a_cube_gradcheck():
    """Cube shape: a wrong inverse stays shape-valid, so only the VALUES betray it."""
    a, w = rand(3, 3, 3), rand(3, 3, 3) + 2.0
    gradcheck(lambda x: (w * x.transpose(1, 2, 0)).sum(),
              lambda x: (x.transpose(1, 2, 0) * Tensor(w)).sum(), [a])


def test_transpose_4cycle_gradcheck():
    """(1,2,3,0), inverse (3,0,1,2)."""
    a, w = rand(2, 3, 4, 5), rand(3, 4, 5, 2) + 2.0
    gradcheck(lambda x: (w * x.transpose(1, 2, 3, 0)).sum(),
              lambda x: (x.transpose(1, 2, 3, 0) * Tensor(w)).sum(), [a])


def test_transpose_backward_uses_argsort_not_the_forward_axes():
    """(3) The direct comparison, on a cube where both permutations are legal.

    argsort((1,2,0)) == (2,0,1) != (1,2,0), so the two candidate backwards give
    genuinely different arrays and the assertion can tell them apart.
    """
    axes = (1, 2, 0)
    inv = tuple(np.argsort(axes))
    assert inv != axes, "this test is only meaningful for a non-involutive perm"

    w = rand(3, 3, 3) + 2.0
    x = Tensor(rand(3, 3, 3))
    (x.transpose(*axes) * Tensor(w)).sum().backward()

    assert np.allclose(x.grad, w.transpose(inv))
    assert not np.allclose(x.grad, w.transpose(axes))


def test_transpose_involutive_perms_cannot_tell_the_two_apart():
    """Why the 3-cycle is mandatory: for these, argsort(axes) IS axes."""
    for axes in [(1, 0), (0, 2, 1, 3), (0, 1), (2, 1, 0)]:
        assert tuple(np.argsort(axes)) == axes


def test_transpose_grad_has_the_original_shape():
    x = Tensor(rand(2, 3, 4))
    (x.transpose(1, 2, 0) * Tensor(rand(3, 4, 2) + 2.0)).sum().backward()
    assert x.grad.shape == (2, 3, 4)


def test_reshape_and_transpose_compose_as_a_head_split():
    """(B,T,H*D) -> reshape -> (B,T,H,D) -> transpose -> (B,H,T,D)."""
    a, w = rand(2, 5, 12), rand(2, 3, 5, 4) + 2.0

    def f(x):
        return (w * x.reshape(2, 5, 3, 4).transpose(0, 2, 1, 3)).sum()

    def build(x):
        return (x.reshape(2, 5, 3, 4).transpose(0, 2, 1, 3) * Tensor(w)).sum()

    gradcheck(f, build, [a])


# ---- softmax ----

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# (3) sanity / smoke — not a gradcheck

def test_softmax_forward_matches_reference():
    a = rand(4, 5)
    assert np.allclose(Tensor(a).softmax(axis=-1).data, softmax_np(a, -1))


def test_softmax_sums_to_one_per_group_last_axis():
    out = Tensor(rand(4, 5)).softmax(axis=-1).data
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_softmax_sums_to_one_per_group_other_axes():
    a = rand(2, 3, 4)
    assert np.allclose(Tensor(a).softmax(axis=0).data.sum(axis=0), 1.0)
    assert np.allclose(Tensor(a).softmax(axis=1).data.sum(axis=1), 1.0)


def test_softmax_output_is_strictly_positive():
    assert np.all(Tensor(rand(4, 5)).softmax(axis=-1).data > 0)


def test_softmax_preserves_shape():
    assert Tensor(rand(2, 3, 4)).softmax(axis=-1).shape == (2, 3, 4)


def test_softmax_of_uniform_input_is_uniform():
    out = Tensor(np.zeros((3, 5))).softmax(axis=-1).data
    assert np.allclose(out, 1.0 / 5)


# (2) numerical stability

def test_softmax_does_not_overflow_on_large_values():
    """exp(1000) is inf and inf/inf is nan — the -max shift is what prevents it."""
    a = np.array([[1000.0, 1001.0, 999.0], [1e4, 1e4, 1e4]])
    out = Tensor(a).softmax(axis=-1).data
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_softmax_gradient_is_finite_on_large_values():
    a = np.array([[1000.0, 1001.0, 999.0]])
    x = Tensor(a)
    (x.softmax(axis=-1) * Tensor(np.array([[1.0, 2.0, 3.0]]))).sum().backward()
    assert np.all(np.isfinite(x.grad))


def test_softmax_is_shift_invariant():
    a = rand(4, 5)
    assert np.allclose(Tensor(a).softmax(axis=-1).data,
                       Tensor(a + 1000.0).softmax(axis=-1).data)


def test_softmax_does_not_underflow_to_all_zeros():
    """Very negative values still produce a real distribution, not zeros or nan.

    Note the gaps, not the magnitudes, decide the output: [-1000,-1001,-2000]
    shifts to [0,-1,-1000], so the top two split ~0.73/0.27 and only the far
    one underflows to 0. That underflow is harmless — it was negligible anyway.
    """
    a = np.array([[-1000.0, -1001.0, -2000.0]])
    out = Tensor(a).softmax(axis=-1).data
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(), 1.0)
    assert np.allclose(out, softmax_np(np.array([[0.0, -1.0, -1000.0]]), -1))
    assert out[0, 0] > out[0, 1] > out[0, 2]


# the trap, documented rather than hidden

def test_softmax_sum_is_constant_so_its_gradient_is_zero():
    """softmax(x).sum() == 1 for any x, so d/dx of it is 0 everywhere.

    A gradcheck built on .sum() alone would therefore pass against a backward
    that returns zeros. Every gradcheck below uses a non-uniform upstream.
    """
    x = Tensor(rand(4, 5))
    out = x.softmax(axis=-1).sum()
    out.backward()
    assert np.allclose(out.data, 4.0)
    assert np.allclose(x.grad, 0.0, atol=1e-12)


# (1) gradcheck with non-uniform upstream, varied shapes and axes

def test_softmax_gradcheck_weighted_last_axis():
    a, w = rand(4, 5), rand(4, 5) + 2.0
    gradcheck(lambda x: (w * softmax_np(x, -1)).sum(),
              lambda x: (x.softmax(axis=-1) * Tensor(w)).sum(), [a])


def test_softmax_gradcheck_weighted_axis0():
    a, w = rand(4, 5), rand(4, 5) + 2.0
    gradcheck(lambda x: (w * softmax_np(x, 0)).sum(),
              lambda x: (x.softmax(axis=0) * Tensor(w)).sum(), [a])


def test_softmax_gradcheck_weighted_middle_axis_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * softmax_np(x, 1)).sum(),
              lambda x: (x.softmax(axis=1) * Tensor(w)).sum(), [a])


def test_softmax_gradcheck_weighted_last_axis_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * softmax_np(x, -1)).sum(),
              lambda x: (x.softmax(axis=-1) * Tensor(w)).sum(), [a])


def test_softmax_gradcheck_weighted_axis0_3d():
    a, w = rand(2, 3, 4), rand(2, 3, 4) + 2.0
    gradcheck(lambda x: (w * softmax_np(x, 0)).sum(),
              lambda x: (x.softmax(axis=0) * Tensor(w)).sum(), [a])


def test_softmax_gradcheck_against_one_hot_target():
    """MSE against a one-hot row — the shape of loss a classifier produces.

    Real cross-entropy needs log(), which does not exist yet; this keeps the
    same 'compare the distribution to a target index' structure without it.
    """
    a = rand(3, 5)
    onehot = np.zeros((3, 5))
    onehot[np.arange(3), [1, 4, 0]] = 1.0

    def f(x):
        d = softmax_np(x, -1) - onehot
        return (d * d).sum()

    def build(x):
        d = x.softmax(axis=-1) - Tensor(onehot)
        return (d * d).sum()

    gradcheck(f, build, [a])


def test_softmax_gradcheck_after_a_matmul():
    """softmax over the scores of a matmul — the attention shape."""
    a, w = rand(4, 3), rand(4, 6) + 2.0
    b = rand(3, 6)

    gradcheck(lambda a, b: (w * softmax_np(a @ b, -1)).sum(),
              lambda a, b: ((a @ b).softmax(axis=-1) * Tensor(w)).sum(), [a, b])


def test_div_by_sqrt_of_var_gradcheck():
    """(x - mean) / sqrt(var + eps) — every M2 primitive in one graph."""
    a = rand(2, 5, 3)

    def f(x):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        return ((x - m) / np.sqrt(v + 1e-5)).sum()

    def build(x):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        return ((x - m) / (v + 1e-5).sqrt()).sum()

    gradcheck(f, build, [a])


# ---- pick ----
#
# N != V em TODOS os testes abaixo (N=5, V=8) — ver a justificativa no topo:
# com N == V a troca de eixos na indexação passaria em silêncio.

N, V = 5, 8


def pick_np(x, idx):
    """out[i] = x[i, idx[i]], por loop puro — a referência independente do NumPy fancy."""
    return np.array([x[i, idx[i]] for i in range(len(idx))])


def test_pick_forward_matches_a_python_loop():
    """Forward contra o loop explícito, não contra outra indexação do NumPy."""
    x = rand(N, V)
    idx = np.array([3, 0, 7, 1, 5])

    got = Tensor(x).pick(idx)

    assert got.shape == (N,)
    for i in range(N):
        assert got.data[i] == x[i, idx[i]]
    assert np.array_equal(got.data, pick_np(x, idx))


def test_pick_gradcheck_non_uniform_upstream():
    """Upstream não-uniforme: com w constante o grad seria o mesmo em toda linha."""
    x = rand(N, V)
    idx = np.array([3, 0, 7, 1, 5])
    w = np.array([0.3, -1.7, 2.1, 0.5, -0.9])

    gradcheck(lambda a: (w * pick_np(a, idx)).sum(),
              lambda a: (a.pick(idx) * Tensor(w)).sum(), [x])


def test_pick_gradcheck_after_a_matmul():
    """pick no fim de um grafo, não sozinho — o grad tem que atravessar o matmul."""
    a = rand(N, 3, seed=1)
    b = rand(3, V, seed=2)
    idx = np.array([7, 7, 0, 4, 2])
    w = np.array([1.3, -0.4, 0.8, -2.0, 0.6])

    gradcheck(lambda p, q: (w * pick_np(p @ q, idx)).sum(),
              lambda p, q: ((p @ q).pick(idx) * Tensor(w)).sum(), [a, b])


def test_pick_leaves_every_unselected_cell_at_exactly_zero():
    """Uma única célula não-zero por linha, na coluna certa, e zero BINÁRIO no resto."""
    x = Tensor(rand(N, V))
    idx = np.array([3, 0, 7, 1, 5])
    w = np.array([0.3, -1.7, 2.1, 0.5, -0.9])

    (x.pick(idx) * Tensor(w)).sum().backward()

    for i in range(N):
        assert x.grad[i, idx[i]] == w[i], "a célula selecionada recebe o upstream da sua linha"
        others = np.delete(x.grad[i], idx[i])
        assert np.count_nonzero(others) == 0, f"linha {i} vazou: {x.grad[i]}"


def test_pick_rows_stay_independent_when_the_target_class_repeats():
    """Todas as linhas com o MESMO alvo — o caso que no Embedding perderia escritas.

    Cada linha tem que receber o SEU upstream, não a soma dos cinco. A prova é
    posicional: as células (i, 2) são distintas para i distintos, então não há
    colisão pra bufferização perder. Compara por igualdade exata: se as escritas
    colidissem, toda linha sobrevivente carregaria w.sum() ou o w da última.
    """
    x = Tensor(rand(N, V))
    idx = np.array([2, 2, 2, 2, 2])
    w = np.array([0.3, -1.7, 2.1, 0.5, -0.9])

    (x.pick(idx) * Tensor(w)).sum().backward()

    assert np.array_equal(x.grad[:, 2], w)
    assert np.count_nonzero(np.delete(x.grad, 2, axis=1)) == 0
    assert not np.allclose(x.grad[:, 2], w.sum()), "colisão: linhas somadas entre si"


def test_pick_accumulates_onto_gradient_from_another_consumer():
    """x lido por pick E por outra rota — o grad é a soma, não só o do pick.

    Isto é o que exige `+=` em vez de `=`: self.grad já chega com a contribuição
    da outra rota. Nada a ver com colisão dentro do pick (essa é impossível),
    é a categoria x + x do M0, entre ops em vez de dentro de uma.
    """
    x = Tensor(rand(N, V))
    idx = np.array([3, 0, 7, 1, 5])

    (x.pick(idx).sum() + (x * Tensor(2.0)).sum()).backward()

    expected = np.full((N, V), 2.0)
    for i in range(N):
        expected[i, idx[i]] += 1.0
    assert np.array_equal(x.grad, expected)


def test_pick_with_a_list_of_indices():
    """Aceita lista Python, não só ndarray — o call site da cross-entropy varia."""
    x = rand(N, V)
    assert np.array_equal(Tensor(x).pick([3, 0, 7, 1, 5]).data,
                          pick_np(x, [3, 0, 7, 1, 5]))
