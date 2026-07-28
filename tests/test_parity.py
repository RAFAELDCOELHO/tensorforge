"""
Paridade contra o PersonaCore treinado — o primeiro teste que não é auto-referente.

Todo teste até aqui comparou o tensorforge contra uma referência que eu mesmo
escrevi em NumPy, ou contra literais gerados op a op. Este carrega os 10.6M
pesos de um GPT REAL, treinado por 49k passos, roda uma janela REAL do corpus,
e compara logits, loss e gradientes contra o que o PyTorch produz com os mesmos
números. Um erro de arquitetura que passasse por todos os gradchecks — heads
trocados, residual faltando, ordem de norm errada — aparece aqui como
divergência numérica grande, porque não existe referência complacente.

Origem dos dados (tudo confirmado lendo, nada assumido)
-------------------------------------------------------
Checkpoint: PersonaCore/checkpoints/best.pt, torch.save de um dict aberto (NÃO
um state_dict pelado): chaves model/optimizer/scheduler/rng/step/val_loss/
model_config/train_config/git_sha/scaler/schema_version. step=49000,
val_loss=0.7378. model_config = vocab 8192, block 256, n_layer 6, n_head 6,
n_embd 384, dropout 0.0. Pesos em float32.

Entrada: PersonaCore/data/val.bin, memmap uint16 de 12.636.923 tokens. A janela
é tokens[0:257] — x = [0:256], y = [1:257], o mesmo shift do get_batch. Texto
real ("u don't have to be scared of the loud dog, I'll protect you"), não
sintético.

O env do tensorforge não tem torch (é o ponto do projeto), então os pesos, a
entrada e as referências foram congelados uma vez num .npz em float64 pelo venv
do PersonaCore. O cast float32 -> float64 é sem perda, então as duas
implementações fazem a MESMA matemática em float64: o que sobrar de diferença é
ordem de operação, não precisão de entrada.


O mapeamento de nomes, 1:1
---------------------------
101 tensores no state_dict, 100 storages únicos:

    wte.weight              (8192,384)  -> model.wte.weight          direto
    wpe.weight              (256,384)   -> model.wpe.weight          direto
    blocks.{i}.ln_1.weight  (384,)      -> blocks[i].ln_1.gamma      renomeado
    blocks.{i}.ln_1.bias    (384,)      -> blocks[i].ln_1.beta       renomeado
    blocks.{i}.attn.{q,k,v,c}_proj.weight (384,384) -> ...weight     TRANSPOSTO
    blocks.{i}.attn.{q,k,v,c}_proj.bias   (384,)    -> ...bias       direto
    blocks.{i}.ln_2.{weight,bias}                   -> gamma/beta    renomeado
    blocks.{i}.mlp.fc_in.weight  (1536,384) -> fc_in.weight          TRANSPOSTO
    blocks.{i}.mlp.fc_out.weight (384,1536) -> fc_out.weight         TRANSPOSTO
    ln_f.{weight,bias}                      -> ln_f.gamma/beta       renomeado
    lm_head.weight          (8192,384)  -> NÃO CARREGADO (ver abaixo)

Duas armadilhas no mapeamento, ambas verificadas e não deduzidas:

1. TRANSPOSIÇÃO. torch.nn.Linear guarda W como (out, in) e faz x @ W.T; o
   Linear daqui guarda (in, out) e faz x @ W. Como todas as projeções da
   atenção são quadradas (384,384), uma transposição esquecida ali não muda
   forma nenhuma e passaria em silêncio — é o mesmo padrão do N==V do pick. Só
   fc_in (1536,384) e fc_out (384,1536) estourariam erro de forma. Por isso o
   teste compara VALORES contra o torch, não formas.

2. lm_head.weight APARECE no state_dict. Eu esperava que não aparecesse por
   causa do weight tying, e estava errado: o PyTorch serializa parâmetros
   amarrados sob os DOIS nomes. Verificado — data_ptr() idêntico ao de
   wte.weight e torch.equal True, ou seja é a mesma tensor vista duas vezes,
   não um parâmetro independente (101 chaves, 100 storages). Carregá-lo como
   peso separado seria inofensivo aqui (os valores são iguais) mas mascararia
   uma quebra do tying; o teste assere a identidade em vez de carregar.

Máscara causal: o PersonaCore usa masked_fill(..., float("-inf")); este projeto
usa -1e9. Os dois dão exatamente 0 depois do softmax — exp(-inf) = 0, e
exp(-1e9 - max) faz underflow para 0 binário — então a diferença de convenção
não entra na comparação. attn_impl="manual" no export (o caminho que este
projeto espelha), não o F.scaled_dot_product_attention.


Tolerância, derivada antes de medir
------------------------------------
Não existe "bit a bit" a esperar aqui, e o motivo é estrutural: a adição de
ponto flutuante não é associativa, e as duas implementações somam os produtos
internos em ordens diferentes (BLAS com blocagem e acumulação por painéis vs o
matmul do NumPy). Mesmos números, mesma matemática, ordens diferentes.

O erro de um produto interno de comprimento K com soma em pares é ~sqrt(K)·eps
relativo (eps = 2.22e-16); com soma sequencial seria K·eps. Por bloco há ~7
contrações em série: q/k/v (K=384), os scores (K=64), attn@v (K=256), c_proj
(K=384), fc_in (K=384), fc_out (K=1536) — média de sqrt(K) na casa de 25. São
~7 contrações × 25 ≈ 175 eps por bloco. Seis blocos mais o lm_head (K=384,
sqrt ≈ 20) dão ~1070 eps ≈ 2.4e-13 relativo.

Duas coisas empurram para baixo: cada LayerNorm renormaliza a stream residual,
o que impede o erro relativo de compor multiplicativamente entre blocos. E uma
para cima: o gradiente atravessa a mesma profundidade de novo, então o limite
para grads é ~2x o dos logits.

Fixo então rtol=1e-11 para logits e loss (o limite derivado com ~40x de
margem, absorvendo o que a estimativa de sqrt(K) subestima) e rtol=1e-10 para
gradientes. O erro MEDIDO é impresso e asserido separadamente contra um limite
apertado — se um dia ele subir para perto da tolerância, o teste ainda passa
mas o número no relatório denuncia. Tolerância é um limite derivado; o valor
medido é o resultado.
"""
import os

import numpy as np
import pytest

from core.nn import GPT, cross_entropy

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "fixtures", "personacore_parity.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="fixtures/personacore_parity.npz ausente (gerado do checkpoint do PersonaCore)")

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 8192, 384, 6, 6, 256

LOGITS_RTOL = 1e-11
GRAD_RTOL = 1e-10


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


def load_personacore(ref):
    """Constrói o GPT daqui e sobrescreve todo peso com o do checkpoint.

    Cola só de teste — o core/ não ganha um from_state_dict por causa disto.
    Toda atribuição é asserida em forma: uma transposição esquecida numa das
    projeções não-quadradas (fc_in/fc_out) estoura aqui em vez de virar
    divergência numérica sem causa aparente lá na frente.
    """
    def w(key):
        return ref[f"w::{key}"].astype(np.float64)

    model = GPT(VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK)

    def put(tensor, value):
        assert tensor.shape == value.shape, f"{tensor.shape} != {value.shape}"
        tensor.data = value

    put(model.wte.weight, w("wte.weight"))
    put(model.wpe.weight, w("wpe.weight"))
    put(model.ln_f.gamma, w("ln_f.weight"))
    put(model.ln_f.beta, w("ln_f.bias"))

    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        put(block.ln_1.gamma, w(p + "ln_1.weight"))
        put(block.ln_1.beta, w(p + "ln_1.bias"))
        put(block.ln_2.gamma, w(p + "ln_2.weight"))
        put(block.ln_2.beta, w(p + "ln_2.bias"))
        for name in ("q_proj", "k_proj", "v_proj", "c_proj"):
            proj = getattr(block.attn, name)
            put(proj.weight, w(f"{p}attn.{name}.weight").T)   # (out,in) -> (in,out)
            put(proj.bias, w(f"{p}attn.{name}.bias"))
        put(block.mlp.fc_in.weight, w(p + "mlp.fc_in.weight").T)
        put(block.mlp.fc_in.bias, w(p + "mlp.fc_in.bias"))
        put(block.mlp.fc_out.weight, w(p + "mlp.fc_out.weight").T)
        put(block.mlp.fc_out.bias, w(p + "mlp.fc_out.bias"))

    return model


@pytest.fixture(scope="module")
def forward_pass(ref):
    """Um forward + backward, reusado pelos testes (T=256, 6 camadas: caro)."""
    model = load_personacore(ref)
    x = ref["input_x"][None, :]
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, VOCAB), ref["input_y"])
    loss.backward()
    return model, logits, loss


def relative_error(got, expected):
    return np.abs(got - expected).max() / np.abs(expected).max()


# ---- (0) o mapeamento em si ----

def test_lm_head_is_the_same_tensor_as_wte_not_a_separate_weight(ref):
    """O tying no checkpoint: lm_head.weight existe, mas é wte.weight de novo."""
    assert np.array_equal(ref["w::lm_head.weight"], ref["w::wte.weight"])


def test_the_fixture_carries_a_real_corpus_window(ref):
    """Entrada de verdade: tokens dentro do vocab, x e y deslocados de 1."""
    x, y = ref["input_x"], ref["input_y"]
    assert x.shape == (BLOCK,) and y.shape == (BLOCK,)
    assert x.max() < VOCAB and x.min() >= 0
    assert np.array_equal(x[1:], y[:-1]), "y é x deslocado de uma posição"
    assert len(np.unique(x)) > 50, "texto real, não um token repetido"


# ---- (1) logits ----

def test_logits_match_personacore(forward_pass, ref):
    _, logits, _ = forward_pass
    got = logits.data[0]
    expected = ref["ref_logits"]

    assert got.shape == expected.shape == (BLOCK, VOCAB)
    err = relative_error(got, expected)
    print(f"\n[paridade] erro relativo dos logits: {err:.3e}  (rtol {LOGITS_RTOL:.0e})")
    assert err < LOGITS_RTOL, f"erro {err:.3e} — investigar arquitetura, não tolerância"
    assert err < 1e-12, f"erro {err:.3e} acima do limite derivado de 2.4e-13"


def test_argmax_prediction_is_identical_token_for_token(forward_pass, ref):
    """Discreto, não numérico: as duas implementações prevêem o MESMO token.

    Complementa a comparação de valores — um erro pequeno demais para estourar a
    tolerância mas concentrado no topo da distribuição mudaria a previsão, e
    isto pega. Igualdade exata em todas as 256 posições.
    """
    _, logits, _ = forward_pass
    assert np.array_equal(logits.data[0].argmax(-1), ref["ref_logits"].argmax(-1))


# ---- (2) loss ----

def test_loss_matches_personacore(forward_pass, ref):
    _, _, loss = forward_pass
    expected = float(ref["ref_loss"])

    err = abs(loss.data - expected) / abs(expected)
    print(f"[paridade] erro relativo da loss: {err:.3e}  (loss = {loss.data:.10f})")
    assert err < LOGITS_RTOL


# ---- (3) gradientes ----

def test_sampled_parameter_gradients_match_personacore(forward_pass, ref):
    """Amostra de 4 parâmetros em profundidades diferentes, não todos (custo).

    q_proj do bloco 0 (o mais fundo no backward), ln_1.bias do bloco 0,
    fc_out do bloco 5 (o mais raso) e ln_f.gamma. Se o backward divergisse por
    uma camada só, uma dessas quatro cairia.
    """
    model, _, _ = forward_pass
    cases = [
        ("blocks.0.attn.q_proj.weight", model.blocks[0].attn.q_proj.weight, True),
        ("blocks.0.ln_1.bias", model.blocks[0].ln_1.beta, False),
        ("blocks.5.mlp.fc_out.weight", model.blocks[5].mlp.fc_out.weight, True),
        ("ln_f.weight", model.ln_f.gamma, False),
    ]
    for key, param, transposed in cases:
        expected = ref[f"g::{key}"]
        if transposed:
            expected = expected.T
        err = relative_error(param.grad, expected)
        print(f"[paridade] grad {key}: {err:.3e}")
        assert err < GRAD_RTOL, f"{key}: erro {err:.3e}"


def test_tied_embedding_gradient_matches_on_both_routes(forward_pass, ref):
    """wte é lido pela lookup E pela projeção de saída — o grad é a soma.

    O teste do GPT provou que a soma acontece; este prova que ela dá o MESMO
    número que o PyTorch, incluindo linhas de tokens que NÃO aparecem na janela
    (grad só pela rota do lm_head) e linhas que aparecem (as duas rotas).
    """
    model, _, _ = forward_pass
    rows = ref["g::wte.rows"]
    err = relative_error(model.wte.weight.grad[rows], ref["g::wte.weight"])
    print(f"[paridade] grad wte (12 linhas amostradas): {err:.3e}")
    assert err < GRAD_RTOL

    total = np.abs(model.wte.weight.grad).sum()
    ref_total = float(ref["g::wte.full_abs_sum"])
    assert abs(total - ref_total) / ref_total < GRAD_RTOL, "soma sobre a tabela inteira"
