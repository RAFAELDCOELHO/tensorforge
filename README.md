# tensorforge

Framework de deep learning construído do zero. Prova final: retreinar o PersonaCore
inteiro rodando só neste engine, sem PyTorch por baixo.

## Status

- [x] **M0 — Scalar autograd** — `Value`, grafo computacional, `.backward()`,
      ops: `+ * ** neg sub truediv relu tanh exp`.
- [x] **M1 — Tensor autograd** — `Tensor` sobre NumPy, broadcasting com
      `_unbroadcast`, `matmul` com dimensões de batch arbitrárias.
- [x] **M2 — Módulos e otimizadores** — `Linear`, `LayerNorm`, `SGD`, `Adam`.
- [x] **M3 — Transformer** — `softmax`, `reshape`/`transpose`, SDPA com máscara
      causal, `Embedding`, `tanh`, `gelu`, `CausalSelfAttention`, `MLP`,
      `Block`, `GPT` com weight tying.
- [x] **M4 — Treino do PersonaCore neste engine** — `log_softmax`, `pick`,
      `cross_entropy`, `AdamW`, `lr_at_step`, `clip_grad_norm_`, `train_step`.
      Paridade numérica contra o checkpoint real validada (abaixo).
- [ ] M5 — Stretch: backend MLX (GPU nativa no M3 Pro, sem custo de cloud)

**449 testes.** Cada ciclo seguiu o mesmo protocolo: derivação escrita antes do
teste, TDD red→green, e mutation check no fim — mutações são aplicadas linha a
linha e a suíte tem que pegar cada uma. Mutantes sobreviventes são investigados
e só aceitos quando provadamente equivalentes.

## Paridade com o PersonaCore

O engine é validado contra um oráculo externo, não só contra referências
escritas aqui. `tests/test_parity.py` carrega os 13.9M pesos de
`checkpoints/best.pt` (treinado por 49k passos) e roda uma janela real de
`data/val.bin`:

| | erro relativo vs PyTorch (float64) |
|---|---|
| logits (256×8192) | 2.7e-15 |
| loss | 0.0 (exato) |
| gradientes amostrados | 3.7e-15 a 9.6e-15 |
| um passo de AdamW a partir do estado real | 0.0 a 2.5e-16 |
| `clip_grad_norm_` em gradientes reais | ~1e-15 |

Argmax idêntico token a token nas 256 posições. As tolerâncias são limites
derivados por escrito (`sqrt(K)·eps` acumulado pela profundidade), não valores
achados por tentativa; os erros medidos ficaram 90x a 400x abaixo delas.

O que **não** é reproduzido, por princípio e não por bug: a trajetória de
treino. O PersonaCore treinou em float32 no MPS; aqui é float64 em NumPy. A
diferença no primeiro passo é ~1e-7 e o treino amplifica isso exponencialmente.
O alvo validado é **o passo**, não a curva.

## Decisões e achados

### Bug de determinismo no backward (achado em M4)

`Tensor._prev` (e `Value._prev`) guardavam os filhos do nó num `set`. Iterar um
`set` de objetos Python segue hashes derivados de `id()`, que mudam entre
processos — então a ordem topológica reversa mudava a cada execução, e com ela
a ordem em que as acumulações `+=` caíam no `.grad`. Soma de ponto flutuante
não é associativa: **duas execuções idênticas divergiam por ~1 ulp por
parâmetro.**

Como foi pego: o teste que compara `train_step` contra a composição manual das
peças exige igualdade **byte a byte**. Ele falhou. Antes de mexer em qualquer
coisa, a hipótese foi testada rodando a *mesma* rota duas vezes — que também
divergiu, descartando "train_step está errado" e apontando para o motor.
Imprimir a ordem topológica em duas execuções confirmou.

Por que importa aqui e não seria só ruído: a contratação de resume do
PersonaCore é trajetória bit a bit idêntica, e o objetivo deste repositório é
retreinar aquele modelo. Gradiente não-determinístico torna qualquer
reprodutibilidade a nível de bit impossível.

Conserto: `_prev` virou tupla nos dois arquivos. Duplicatas são inofensivas —
`build_topo` já deduplica via `visited`, então `x + x` continua acumulando duas
vezes. Fixado por teste em `tests/test_train.py` e `tests/test_engine.py`.

A lição de método: um teste de igualdade **exata** encontrou um defeito que
nenhum `allclose` encontraria. Tolerância frouxa esconde bug de ordem.

### Outras decisões registradas

- **Máscara causal com `-1e9`, não `float("-inf")`.** Uma linha inteiramente
  mascarada (padding, ainda não usado) daria `-inf - (-inf) = nan` no shift de
  estabilidade do softmax, e o `nan` se espalha pelo batch no backward. Com
  `-1e9` a linha degrada para uniforme. Numericamente idêntico ao `-inf` do
  PersonaCore no caso causal, onde a diagonal nunca é mascarada.
- **`Linear` guarda `W` como `(in, out)`**, ao contrário do PyTorch, que guarda
  `(out, in)` e transpõe no forward. O carregador de pesos transpõe. Como as
  projeções da atenção são quadradas, esquecer a transposição não muda forma
  nenhuma — só a comparação de valores contra o torch pega.
- **`weight_decay` aplicado a todos os parâmetros**, inclusive vieses e ganhos
  de LayerNorm. É a convenção do nanoGPT excluir 1-D; o PersonaCore não exclui
  (um param group só, 100 tensores), e o alvo é o PersonaCore real.
- **`lr_at_step(step)` com `step` 0-based.** No PersonaCore o
  `scheduler.step()` vem *depois* do `optimizer.step()` e o `LambdaLR` já anda
  uma vez na construção, então a iteração *k* usa `λ(k−1)`. O LR gravado no
  checkpoint em `step=49000` nunca foi aplicado por passo nenhum: é o valor
  engatilhado para a iteração seguinte.
- **`eps` do Adam fora da raiz**, `lr·m̂/(√v̂ + eps)`, seguindo o PyTorch.
- **Sem AMP.** O treino de referência rodou em MPS, onde o `RuntimeConfig` do
  PersonaCore desliga AMP; o `GradScaler` do checkpoint está vazio. fp32 puro
  é a réplica fiel, não uma simplificação.
- **Sem gradient accumulation.** A corrida real usa `grad_accum_steps=1`.

## Rodar os testes

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Os testes de paridade pulam sozinhos se as fixtures não estiverem presentes
(elas são geradas a partir do checkpoint do PersonaCore, que é gitignorado lá).

## Demonstração fim a fim

```bash
python scripts/demo_train.py
```

Carrega os pesos reais e treina neste engine, em dois regimes. Overfit de um
batch (o gate clássico, que prova que o motor *aprende*): loss de 0.756 para
0.006 em 15 passos, ~1.5s/passo para 13.9M parâmetros em NumPy. E continuação
fiel com o LR real do passo 49000 (que prova que o motor *roda* no regime
real): a loss oscila com o batch e não desce, exatamente como se espera de um
modelo já convergido.

## Como usar

```python
import numpy as np
from core.nn import GPT
from core.optim import AdamW
from core.train import train_step

model = GPT(vocab_size=8192, n_embd=384, n_head=6, n_layer=6, block_size=256)
opt = AdamW(model.parameters(), weight_decay=0.1)

loss, grad_norm = train_step(model, opt, x, y, step=0, max_norm=1.0,
                             base_lr=3e-4, warmup_steps=100, max_steps=50000)
```
