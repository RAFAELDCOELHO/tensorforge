"""
core/nn.py — network modules (M2).

A Module owns Tensors and knows how to list them (parameters()) so an
optimizer can step them and zero_grad() can clear them. Nothing here touches
the autograd graph directly: Linear is x @ W + b, and both of those ops
already carry their own backward, including the broadcast reduction that sums
dW back over the batch axis.
"""
import numpy as np

from core.tensor import Tensor


# Additive mask value for disallowed positions. Not float('-inf') on purpose:
# softmax subtracts the row max for stability, and a fully masked row would
# make that -inf - (-inf) = nan, which then spreads through the whole batch in
# the backward. Causal masking never produces a fully masked row (the diagonal
# is always allowed), but a padding mask does. -1e9 underflows to exactly 0
# after exp, so it is numerically identical here, and degrades to a uniform
# row instead of a nan in the case that will show up later.
MASKED = -1e9


def scaled_dot_product_attention(q, k, v, causal=False):
    """softmax(q @ k^T / sqrt(dk)) @ v, optionally causally masked.

    The mask is ADDED to the scores BEFORE the softmax. Added, because a
    multiplicative 0/1 mask would turn a forbidden score into exp(0) = 1, a
    large weight rather than a vanishing one. Before, because the softmax is
    what normalizes each row: masking afterwards strips mass from an already
    normalized distribution and leaves row i summing to less than 1, by a
    different amount for every row.

    Masking afterwards also leaks, in a way that is easy to miss: the weights
    that survive were divided by a denominator summed over the whole row, so
    row i still depends on the future KEYS even though its future value
    weights are zero. Causal in v, not causal in k.

    Pure composition of ops that each already carry a gradcheck. The only
    thing worth spelling out is the transpose: Tensor.transpose takes a full
    permutation, so swapping just the last two axes means building the
    identity permutation and swapping its tail — which keeps however many
    batch axes precede them (B and H here) exactly where they are.

    The 1/sqrt(dk) keeps the scores from growing with dk, which would drive
    softmax into a one-hot corner where its gradient vanishes.
    """
    axes = list(range(len(k.shape)))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    scores = (q @ k.transpose(*axes)) / np.sqrt(q.shape[-1])
    if causal:
        # strictly upper triangle = the future, relative to each query row
        scores = scores + Tensor(np.triu(np.full(scores.shape[-2:], MASKED), 1))
    return scores.softmax(axis=-1) @ v


def cross_entropy(logits, targets):
    """Mean negative log-likelihood — PersonaCore gpt.py:212.

        -log_softmax(logits, axis=-1).pick(targets).mean()

    logits are (N, V) and targets an integer (N,) of class indices; the caller
    flattens (B, T, V) to (B*T, V) itself, exactly as gpt.py does before
    calling F.cross_entropy.

    Pure composition — log_softmax, pick, mean and negation each carry their
    own gradcheck, so there is no backward here. Written on log_softmax rather
    than on log(softmax(x)) because the latter underflows to log(0) = -inf the
    moment one logit leads by ~745.

    The reduction is the MEAN over N, matching F.cross_entropy's default. Sum
    would differ by exactly a factor of N, which no gradcheck can see and which
    scales the effective learning rate by B*T.

    No ignore_index, no label smoothing, no class weights: PersonaCore passes
    none of them, and its data path has no padding to ignore — get_batch draws
    full block_size windows from a flat token stream, so every position is a
    real target, the inter-document eos included.
    """
    return -logits.log_softmax(axis=-1).pick(targets).mean()


class Module:
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """y = x @ W + b, with W shaped (in_features, out_features).

    PyTorch stores W transposed, (out, in), and matmuls against W.T. We have no
    transpose op yet and no reason to want one, so the weight is stored in the
    orientation the forward pass actually uses.
    """

    def __init__(self, in_features, out_features, bias=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # same init as torch.nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(rng.uniform(-bound, bound, (in_features, out_features)))
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        return out if self.bias is None else out + self.bias

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class Embedding(Module):
    """A lookup table: out[b, t] = weight[idx[b, t]].

    Serves both token and positional embeddings — same class, different table
    and different indices (token ids in one, arange(T) in the other).

    Indices are addresses, not numbers: they carry no gradient and are passed
    as a plain integer array rather than a Tensor. The only parameter is the
    table itself.

    The backward is a scatter-add, and it uses np.add.at rather than
    `grad[idx] += g` for one specific reason: NumPy's fancy-index += is
    buffered. It reads grad[idx] into a temporary, adds, and writes back, so
    with a repeated index every read sees the same original value and the last
    write wins — the row ends up with one contribution instead of their sum. A
    row read k times IS a shared node, the same situation as x + x, and every
    read has to accumulate. np.add.at is the unbuffered version that does so.
    """

    def __init__(self, num_embeddings, embedding_dim, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.weight = Tensor(rng.standard_normal((num_embeddings, embedding_dim)))

    def forward(self, idx):
        out = Tensor(self.weight.data[idx], (self.weight,), "embedding")
        weight = self.weight

        def _backward():
            np.add.at(weight.grad, idx, out.grad)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


class CausalSelfAttention(Module):
    """Multi-head causal self-attention.

    Four separate projections — q/k/v are NOT fused into one c_attn, matching
    PersonaCore gpt.py:71-74, where keeping them separate leaves the LoRA seam
    open by name. c_proj writes back to the residual stream.

    The head split goes (B,T,C) -> reshape (B,T,H,dk) -> transpose (B,H,T,dk),
    and never straight to (B,H,T,dk). Both give the same shape, so nothing
    about the shapes catches the difference, but C is the fastest-varying axis
    in row-major: the direct reshape hands head 0 the first T/H timesteps
    across every channel instead of the first dk channels across all time. The
    merge is the same route in reverse.

    No dropout: PersonaCore runs dropout=0.0 everywhere, so it would be the
    identity here.
    """

    def __init__(self, n_embd, n_head, rng=None):
        assert n_embd % n_head == 0, \
            f"n_embd {n_embd} must divide evenly into n_head {n_head}"
        if rng is None:
            rng = np.random.default_rng()
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.q_proj = Linear(n_embd, n_embd, rng=rng)
        self.k_proj = Linear(n_embd, n_embd, rng=rng)
        self.v_proj = Linear(n_embd, n_embd, rng=rng)
        self.c_proj = Linear(n_embd, n_embd, rng=rng)

    def _split_heads(self, z, B, T):
        return z.reshape(B, T, self.n_head, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, y, B, T, C):
        return y.transpose(0, 2, 1, 3).reshape(B, T, C)

    def forward(self, x):
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x), B, T)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        y = scaled_dot_product_attention(q, k, v, causal=True)

        return self.c_proj(self._merge_heads(y, B, T, C))

    def parameters(self):
        return (self.q_proj.parameters() + self.k_proj.parameters()
                + self.v_proj.parameters() + self.c_proj.parameters())


class GPT(Module):
    """Decoder-only transformer — PersonaCore gpt.py:195-213.

        x = wte(idx) + wpe(arange(T))
        for block in blocks: x = block(x)
        x = ln_f(x)
        logits = x @ wte.weight^T

    Positions are built at call time from the actual T, not fixed at
    block_size: wpe holds block_size rows and only the first T are ever read,
    which is what lets one model serve any sequence up to the limit. The
    (T, C) result broadcasts over the batch.

    The head is TIED to the token embedding and has no bias — a head bias
    would be an untied parameter. So wte.weight is read on two routes, as the
    lookup table and as the output projection, and its gradient is the sum of
    both. That sum is not special-cased anywhere: the table is one node with
    two consumers, and gradient accumulation does the rest, exactly like x + x.

    ln_f belongs here rather than in Block: it normalizes once, after the last
    block, not between every pair of them.
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.block_size = block_size
        self.wte = Embedding(vocab_size, n_embd, rng=rng)
        self.wpe = Embedding(block_size, n_embd, rng=rng)
        self.blocks = [Block(n_embd, n_head, rng=rng) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)

    def forward(self, idx):
        T = idx.shape[-1]
        assert T <= self.block_size, f"seq len {T} > block_size {self.block_size}"

        x = self.wte(idx) + self.wpe(np.arange(T))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.wte.weight.transpose(1, 0)

    def parameters(self):
        params = self.wte.parameters() + self.wpe.parameters()
        for block in self.blocks:
            params = params + block.parameters()
        return params + self.ln_f.parameters()


class Block(Module):
    """Pre-norm transformer block — PersonaCore gpt.py:142-145.

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    Norm BEFORE each sublayer, residual AROUND it. Post-norm — normalizing
    after adding the residual — is the other convention and gives different
    numbers; this one matches the checkpoint.

    ln_f is NOT here. The model applies one final norm after the last block,
    which is a property of the stack rather than of a block, and putting it
    here would normalize between every pair of blocks.

    ln_1 and ln_2 are separate instances. They initialize to identical values
    (ones and zeros), so nothing about their contents distinguishes two norms
    from one shared norm — only their identity does.
    """

    def __init__(self, n_embd, n_head, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, rng=rng)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, rng=rng)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def parameters(self):
        return (self.ln_1.parameters() + self.attn.parameters()
                + self.ln_2.parameters() + self.mlp.parameters())


class MLP(Module):
    """Position-wise feed-forward: fc_in -> GELU -> fc_out.

    Matches PersonaCore gpt.py:118-129. The 4x hidden width is hardcoded in
    the constructor there, not carried in the config, so it is computed here
    rather than taken as an argument. Both projections keep their bias — the
    model's only bias-free Linear is lm_head, and that is a consequence of
    weight tying.

    No dropout: PersonaCore runs dropout=0.0, which is the identity.
    """

    def __init__(self, n_embd, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.fc_in = Linear(n_embd, 4 * n_embd, rng=rng)
        self.fc_out = Linear(4 * n_embd, n_embd, rng=rng)

    def forward(self, x):
        return self.fc_out(self.fc_in(x).gelu())

    def parameters(self):
        return self.fc_in.parameters() + self.fc_out.parameters()


class LayerNorm(Module):
    """Normalize over the last axis, then scale and shift.

    Pure composition — every op below already carries its own gradcheck, so
    there is no new backward here. keepdims=True on both reductions is what
    makes (…,1) line up against the original (…,N) when broadcasting.

    gamma/beta start at ones/zeros, not at Linear's U(-1/sqrt(fan_in), …):
    the layer starts out as the identity on the normalized input and learns to
    move away from it.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))

    def forward(self, x):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        xhat = (x - m) / (v + self.eps).sqrt()
        return xhat * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]
