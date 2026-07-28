"""
core/optim.py — optimizers (M2).

Separate from nn.py on purpose: a Module builds graph nodes, an optimizer does
not. These read .grad and write .data in place, after backward() has already
run and the graph is done being useful. Nothing here is differentiable and
nothing here belongs in a gradcheck.

Both take the parameter list once (typically model.parameters()) and hold onto
it, so step() and zero_grad() always hit the same tensors.
"""
import numpy as np


class SGD:
    """p <- p - lr * grad. No momentum; add it when a run actually needs it."""

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """Adam (Kingma & Ba 2014), PyTorch's formulation.

        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        m_hat = m / (1 - b1^t)
        v_hat = v / (1 - b2^t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    eps sits OUTSIDE the square root, which is what torch.optim.Adam does. The
    other placement, sqrt(v_hat + eps), is also in circulation and gives
    different numbers for the same eps. Matching PyTorch means a run ported
    from there behaves the same here; test_adam_eps_is_outside_the_sqrt pins
    the choice so it cannot drift silently.

    m and v are per-parameter, one array per entry in params — a single shared
    buffer would mix the statistics of unrelated weights.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        b1, b2 = self.betas
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class AdamW(Adam):
    """Adam with DECOUPLED weight decay (Loshchilov & Hutter 2017), PyTorch's order.

        p <- p * (1 - lr*wd)          first, straight on the parameter
        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Decoupled means the decay never enters the gradient, so it never reaches m
    or v. Adam's other option — folding wd*p into g, which is what
    torch.optim.Adam(weight_decay=) does — normalizes the decay by sqrt(v_hat)
    along with everything else, making the shrink depend on how large that
    parameter's gradients have historically been, and leaves the decay term
    sitting in the moment estimates for every later step.

    The lr multiplying wd is the step's lr, i.e. the one the scheduler already
    scaled — not the base lr. Under a cosine decay the shrink winds down with
    the learning rate.

    Applied to EVERY parameter, with no ndim filter. The nanoGPT convention is
    to build two param groups and exclude 1-D tensors (biases, LayerNorm
    gains), on the reasoning that shrinking a normalization gain toward zero
    fights the normalization rather than regularizing it. PersonaCore does not
    do that: training/loop.py:258 hands model.parameters() straight to AdamW,
    and the checkpoint's param_groups confirms a single group of 100 tensors at
    weight_decay=0.1. Filtering here would be more defensible in the abstract
    and wrong for this project, whose target is that checkpoint.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            p.data *= 1 - self.lr * self.weight_decay
        super().step()


def lr_at_step(step, base_lr, warmup_steps, max_steps, min_ratio=0.1):
    """Linear warmup -> cosine decay -> floor, PersonaCore schedule.py:30-37.

        step < warmup:  (step + 1) / warmup
        step >= max:    min_ratio
        otherwise:      min_ratio + (1 - min_ratio) * (1 + cos(pi*progress))/2
                        with progress = (step - warmup) / (max - warmup)

    A pure function of the step, deliberately not a class. PersonaCore wraps
    the same math in a LambdaLR, but only because checkpoint.py needs
    scheduler.state_dict() to resume — and all LambdaLR serializes is the step
    counter, since the lambda itself is not pickled. With the step passed in
    there is nothing to serialize and nothing to rebuild identically on resume.

    (step + 1), not step: the first step runs at min(1, 1/warmup) of base_lr
    rather than at zero, and the ramp reaches full lr at warmup-1.

    The floor is not decoration. Past max_steps the cosine argument exceeds pi
    and the curve turns back UP; clamping to min_ratio is what keeps a run that
    overshoots max_steps from silently re-warming.

    The result feeds both halves of an AdamW step — the update and the
    decoupled decay, which multiplies by (1 - lr*wd) — so the schedule sets how
    fast weights move AND how hard they shrink.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


def clip_grad_norm_(params, max_norm):
    """Scale every .grad in place so their JOINT L2 norm is at most max_norm.

    Mirrors torch.nn.utils.clip_grad_norm_ (torch 2.7.1, clip_grad.py):

        total_norm = ||[ ||g_1||, ..., ||g_N|| ]||
        coef       = min(max_norm / (total_norm + 1e-6), 1.0)
        g_i       *= coef                  same coef for every i
        return total_norm                  the PRE-clip norm

    Global, not per tensor. Clipping each tensor against its own norm would
    preserve each tensor's direction but change the direction of the joint
    gradient, which is the vector the optimizer actually steps along.

    The norm is taken in two stages — the norm of each tensor, then the norm of
    those norms — rather than over one flattened vector. Algebraically the
    same; the summation order is not, and matching torch's order is what keeps
    the parity fixture tight.

    eps = 1e-6 lives in the denominator and is not configurable, matching
    torch. It also means a gradient sitting exactly at max_norm is still scaled
    (by ~1e-6): the boundary between untouched and scaled is max_norm - eps.

    The clamp at 1.0 is what keeps an already-small gradient from being
    amplified. Like torch, the multiply happens even when the coefficient was
    clamped — multiplying by 1.0 is exact, so those gradients come out
    bit-identical.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = float(np.linalg.norm([np.linalg.norm(g) for g in grads]))
    coef = min(max_norm / (total_norm + 1e-6), 1.0)
    for g in grads:
        g *= coef
    return total_norm
