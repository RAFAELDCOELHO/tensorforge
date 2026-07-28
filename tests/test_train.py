"""
Tests for train_step — a única iteração de treino, com as peças já validadas.

Ordem, investigada em PersonaCore training/loop.py:140-159 (_optimizer_step) e
confirmada rodando o LambdaLR real, não deduzida:

    optimizer.zero_grad(set_to_none=True)
    forward -> loss -> backward                (por micro-batch; accum=1 no real)
    scaler.unscale_(optimizer)                 no-op sem AMP, e AMP nunca foi ligado
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer) -> optimizer.step()
    scaler.update()
    scheduler.step()                           <- DEPOIS do optimizer.step()

O índice do learning rate
--------------------------
O LR não é uma etapa do loop: o scheduler muta optimizer.param_groups[0]["lr"]
in-place e o optimizer lê esse campo quando roda. Como scheduler.step() vem
DEPOIS de optimizer.step(), o LR aplicado numa iteração foi escrito na anterior.

E o LambdaLR já chama step() uma vez na própria construção, deixando
last_epoch=0 e lr = base·λ(0) antes da primeira iteração. Medido:

    iteração 1 -> λ(0)·base = 3.0e-06
    iteração 2 -> λ(1)·base = 6.0e-06
    iteração k -> λ(k-1)·base

Com step 0-based, que é a convenção da assinatura aqui, isso é exatamente
lr_at_step(step). A iteração de índice S usa λ(S).

O corolário fácil de errar: o LR gravado no checkpoint em step=49000 é
λ(49000) = 3.0267460081032337e-05, e ele NUNCA foi usado por nenhum
optimizer.step() daquela corrida — é o valor já engatilhado para a iteração
49001, que não rodou. A 49000ª iteração usou λ(48999). O número que o ciclo do
schedule validou é o LR engatilhado, não o aplicado.

Grad accumulation não está aqui. A corrida real usa grad_accum_steps=1, então o
laço de micro-batches é degenerado: a divisão /accum é /1 e o "somar antes do
clip" tem uma parcela só. Um train_step de um batch é a réplica fiel do que o
PersonaCore fez, e acumulação entra quando alguma corrida precisar dela.
"""
import os

import numpy as np
import pytest

from core.nn import GPT, cross_entropy
from core.optim import AdamW, clip_grad_norm_, lr_at_step
from core.train import train_step

PARITY = os.path.join(os.path.dirname(__file__), os.pardir,
                      "fixtures", "personacore_parity.npz")

# Hiperparâmetros reais, do train_config gravado em checkpoints/best.pt.
REAL = dict(max_norm=1.0, base_lr=3e-4, warmup_steps=100,
            max_steps=50000, min_ratio=0.1)

SMALL = dict(vocab_size=16, n_embd=8, n_head=2, n_layer=2, block_size=6)


def small_model(seed=0):
    """Modelo pequeno e determinístico — duas chamadas com o mesmo seed dão
    pesos idênticos, o que é o que permite comparar rotas byte a byte."""
    return GPT(SMALL["vocab_size"], SMALL["n_embd"], SMALL["n_head"],
               SMALL["n_layer"], SMALL["block_size"],
               rng=np.random.default_rng(seed))


def batch(seed=1, B=2, T=4):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, SMALL["vocab_size"], size=(B, T + 1))
    return ids[:, :T], ids[:, 1:].reshape(-1)


def test_two_models_with_the_same_seed_start_identical():
    """A premissa de toda comparação byte a byte abaixo."""
    a, b = small_model(), small_model()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.array_equal(pa.data, pb.data)


# ---- (1) prova de estrutura contra composição manual ----

def manual_step(model, opt, x, y, step, **hp):
    """As mesmas peças, chamadas na ordem investigada, à mão."""
    opt.zero_grad()
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y)
    loss.backward()
    gn = clip_grad_norm_(model.parameters(), hp["max_norm"])
    opt.lr = lr_at_step(step, hp["base_lr"], hp["warmup_steps"],
                        hp["max_steps"], hp["min_ratio"])
    opt.step()
    return float(loss.data), gn


def test_train_step_equals_the_manual_composition_byte_for_byte():
    """Forma sozinha não distingue ordem certa de errada — daí a comparação
    de valores, em todos os parâmetros, por array_equal."""
    x, y = batch()
    ma, mb = small_model(), small_model()
    oa = AdamW(ma.parameters(), weight_decay=0.1)
    ob = AdamW(mb.parameters(), weight_decay=0.1)

    got = train_step(ma, oa, x, y, 0, **REAL)
    expected = manual_step(mb, ob, x, y, 0, **REAL)

    assert got == expected, "loss/norma retornadas divergiram"
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert np.array_equal(pa.data, pb.data)
    for a, b in zip(oa.m, ob.m):
        assert np.array_equal(a, b)


def test_train_step_matches_the_manual_route_over_several_steps():
    """Uma iteração só poderia coincidir por acaso; cinco em sequência não."""
    ma, mb = small_model(), small_model()
    oa = AdamW(ma.parameters(), weight_decay=0.1)
    ob = AdamW(mb.parameters(), weight_decay=0.1)

    for s in range(5):
        x, y = batch(seed=s)
        assert train_step(ma, oa, x, y, s, **REAL) == manual_step(mb, ob, x, y, s, **REAL)
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert np.array_equal(pa.data, pb.data)


def test_a_wrong_order_really_does_produce_different_numbers():
    """Fixa a premissa do teste acima: com o clip DEPOIS do optimizer.step()
    — clipando tarde demais — os números mudam.

    Sem isto, o array_equal acima poderia estar comparando duas rotas que
    coincidem por serem insensíveis à ordem.
    """
    x, y = batch()
    ma, mb = small_model(), small_model()
    oa, ob = AdamW(ma.parameters(), weight_decay=0.1), AdamW(mb.parameters(), weight_decay=0.1)

    train_step(ma, oa, x, y, 0, **REAL)

    ob.zero_grad()
    logits = mb(x)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y)
    loss.backward()
    ob.lr = lr_at_step(0, REAL["base_lr"], REAL["warmup_steps"],
                       REAL["max_steps"], REAL["min_ratio"])
    ob.step()                                   # step ANTES do clip
    clip_grad_norm_(mb.parameters(), REAL["max_norm"])

    assert any(not np.array_equal(pa.data, pb.data)
               for pa, pb in zip(ma.parameters(), mb.parameters()))


# ---- (2) zero_grad ----

def test_the_second_step_does_not_inherit_the_first_steps_gradient():
    """Mesmo batch duas vezes. Sem zero_grad o segundo backward acumula em cima
    do primeiro e o gradiente vira o dobro — os pesos vão para outro lugar.

    A referência é construída zerando explicitamente entre os dois, com as ops
    soltas, então não depende do próprio train_step estar certo.
    """
    x, y = batch()
    ma, mb = small_model(), small_model()
    oa, ob = AdamW(ma.parameters(), weight_decay=0.1), AdamW(mb.parameters(), weight_decay=0.1)

    train_step(ma, oa, x, y, 0, **REAL)
    train_step(ma, oa, x, y, 1, **REAL)

    for s in (0, 1):
        for p in mb.parameters():
            p.grad = np.zeros_like(p.data)
        logits = mb(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y)
        loss.backward()
        clip_grad_norm_(mb.parameters(), REAL["max_norm"])
        ob.lr = lr_at_step(s, REAL["base_lr"], REAL["warmup_steps"],
                           REAL["max_steps"], REAL["min_ratio"])
        ob.step()

    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert np.array_equal(pa.data, pb.data)


def test_gradients_left_over_before_a_step_are_discarded():
    """Sujar o .grad antes de chamar train_step não pode mudar o resultado."""
    x, y = batch()
    ma, mb = small_model(), small_model()
    oa, ob = AdamW(ma.parameters(), weight_decay=0.1), AdamW(mb.parameters(), weight_decay=0.1)

    for p in ma.parameters():
        p.grad = np.full_like(p.data, 1e3)      # lixo grande e visível

    assert train_step(ma, oa, x, y, 0, **REAL) == train_step(mb, ob, x, y, 0, **REAL)
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert np.array_equal(pa.data, pb.data)


# ---- (3) o learning rate aplicado ----

class SpyAdamW(AdamW):
    """Registra o lr vigente NO MOMENTO do step() — prova que foi escrito antes."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lrs_at_step = []

    def step(self):
        self.lrs_at_step.append(self.lr)
        super().step()


def test_the_lr_applied_is_lr_at_step_of_the_current_index():
    """Dois passos consecutivos com os hiperparâmetros reais do checkpoint.

    O índice é 0-based e o LR é λ(step), não λ(step-1) nem λ(step+1) — a
    conclusão da investigação do LambdaLR, testada em vez de assumida.
    """
    x, y = batch()
    model = small_model()
    opt = SpyAdamW(model.parameters(), weight_decay=0.1)

    for s in (0, 1):
        train_step(model, opt, x, y, s, **REAL)

    for s, used in zip((0, 1), opt.lrs_at_step):
        assert used == lr_at_step(s, REAL["base_lr"], REAL["warmup_steps"],
                                  REAL["max_steps"], REAL["min_ratio"])


def test_the_lr_of_the_neighbouring_steps_would_be_different():
    """Fixa a premissa: λ(s-1), λ(s) e λ(s+1) são três números distintos aqui,
    então o teste acima discrimina de verdade em vez de passar por empate."""
    vals = [lr_at_step(s, REAL["base_lr"], REAL["warmup_steps"],
                       REAL["max_steps"], REAL["min_ratio"]) for s in (0, 1, 2)]
    assert len(set(vals)) == 3


def test_the_lr_is_not_the_one_staged_for_the_next_iteration():
    """O erro clássico do off-by-one: usar λ(step+1), que é o que o campo lr do
    scheduler do PersonaCore guarda depois de scheduler.step()."""
    x, y = batch()
    model = small_model()
    opt = SpyAdamW(model.parameters(), weight_decay=0.1)

    train_step(model, opt, x, y, 7, **REAL)

    used = opt.lrs_at_step[0]
    for wrong in (6, 8):
        assert used != lr_at_step(wrong, REAL["base_lr"], REAL["warmup_steps"],
                                  REAL["max_steps"], REAL["min_ratio"])


def test_the_lr_reaches_the_weights_not_just_the_optimizer_field():
    """Escrever o campo e não usar passaria nos testes acima; dois lrs muito
    diferentes têm que produzir passos de tamanhos diferentes."""
    x, y = batch()
    slow, fast = small_model(), small_model()
    o_slow = AdamW(slow.parameters(), weight_decay=0.1)
    o_fast = AdamW(fast.parameters(), weight_decay=0.1)
    before = [p.data.copy() for p in slow.parameters()]

    train_step(slow, o_slow, x, y, 0, **{**REAL, "base_lr": 3e-6})
    train_step(fast, o_fast, x, y, 0, **{**REAL, "base_lr": 3e-2})

    d_slow = max(np.abs(p.data - b).max() for p, b in zip(slow.parameters(), before))
    d_fast = max(np.abs(p.data - b).max() for p, b in zip(fast.parameters(), before))
    assert d_fast > 100 * d_slow


# ---- (4) um passo completo com pesos e batch reais ----

@pytest.mark.skipif(not os.path.exists(PARITY), reason="fixture de paridade ausente")
def test_one_real_step_reproduces_the_validated_loss():
    """A loss que train_step devolve é a mesma já validada contra o PyTorch.

    Reusa o carregador de pesos do teste de paridade; a loss é a de ANTES do
    passo, então tem que bater com ref_loss exatamente como no forward puro.
    """
    from tests.test_parity import load_personacore

    ref = np.load(PARITY)
    model = load_personacore(ref)
    opt = AdamW(model.parameters(), weight_decay=0.1)
    before = model.wte.weight.data.copy()

    loss, grad_norm = train_step(model, opt, ref["input_x"][None, :], ref["input_y"],
                                 step=49000, **REAL)

    expected = float(ref["ref_loss"])
    assert abs(loss - expected) / expected < 1e-11
    assert grad_norm > 0.0
    assert not np.array_equal(model.wte.weight.data, before), "o passo não moveu nada"


@pytest.mark.skipif(not os.path.exists(PARITY), reason="fixture de paridade ausente")
def test_the_real_step_reports_the_pre_clip_norm(capsys):
    """A norma devolvida é a PRÉ-clip, e ela é quem decide se houve corte."""
    from tests.test_parity import load_personacore

    ref = np.load(PARITY)
    model = load_personacore(ref)
    opt = AdamW(model.parameters(), weight_decay=0.1)

    _, grad_norm = train_step(model, opt, ref["input_x"][None, :], ref["input_y"],
                              step=49000, **REAL)

    clipped = grad_norm > REAL["max_norm"]
    print(f"\n[train] norma pré-clip do passo real: {grad_norm:.6f} "
          f"(max_norm={REAL['max_norm']}) -> {'clipou' if clipped else 'não clipou'}")
    assert np.isfinite(grad_norm)


def test_train_step_returns_a_scalar_loss_and_the_pre_clip_norm():
    x, y = batch()
    model = small_model()
    loss, gn = train_step(model, AdamW(model.parameters()), x, y, 0, **REAL)
    assert isinstance(loss, float) and np.isfinite(loss) and loss > 0
    assert isinstance(gn, float) and gn >= 0


def test_repeated_steps_on_one_batch_drive_the_loss_down():
    """Fim a fim: as peças ligadas realmente aprendem, não só rodam."""
    x, y = batch()
    model = small_model()
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    hp = {**REAL, "base_lr": 1e-2, "warmup_steps": 1}

    first, _ = train_step(model, opt, x, y, 0, **hp)
    for s in range(1, 30):
        last, _ = train_step(model, opt, x, y, s, **hp)

    assert last < first * 0.5, f"loss saiu de {first:.4f} para {last:.4f}"


# ---- determinismo do backward ----
#
# Este teste existe por causa de um defeito que o byte-a-byte acima expôs. O
# _prev do Tensor era um set; iterar um set de objetos segue hashes derivados de
# id(), que mudam entre processos, então a ordem topológica reversa — e portanto
# a ordem em que os += caem no .grad — mudava a cada execução. Soma de ponto
# flutuante não é associativa, então duas execuções idênticas divergiam por ~1
# ulp por parâmetro.
#
# Não é ruído irrelevante para este projeto: a contratação de resume do
# PersonaCore é trajetória bit a bit idêntica, e o alvo do repositório é
# retreinar aquele modelo aqui. Um gradiente não-determinístico torna qualquer
# reprodutibilidade a nível de bit impossível. _prev virou tupla.
#
# (core/engine.py, o Value escalar do M0, tem o mesmo padrão. Ficou como está:
# não participa do caminho de treino.)

def test_the_same_step_twice_gives_bit_identical_weights():
    """Determinismo entre execuções dentro do mesmo processo."""
    x, y = batch()
    runs = []
    for _ in range(3):
        m = small_model()
        train_step(m, AdamW(m.parameters(), weight_decay=0.1), x, y, 0, **REAL)
        runs.append([p.data.copy() for p in m.parameters()])

    for other in runs[1:]:
        for a, b in zip(runs[0], other):
            assert np.array_equal(a, b)


def test_the_graphs_child_list_is_ordered_not_a_set():
    """A causa raiz, fixada diretamente: set não tem ordem estável entre
    processos, e é disso que a ordem de acumulação do gradiente depende."""
    from core.tensor import Tensor

    a, b = Tensor(np.ones(2)), Tensor(np.ones(2))
    assert not isinstance((a + b)._prev, set)
    assert list((a + b)._prev) == [a, b], "ordem preservada"


def test_a_shared_node_still_accumulates_once_per_read():
    """A tupla admite duplicatas onde o set não admitia — x + x tem que
    continuar somando as duas leituras, não uma."""
    from core.tensor import Tensor

    x = Tensor(np.array([3.0]))
    (x + x).sum().backward()
    assert x.grad[0] == 2.0
