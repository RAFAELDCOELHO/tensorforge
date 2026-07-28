"""
core/train.py — one training iteration, wiring the pieces M4 validated.

Nothing new is computed here. The model, cross_entropy, AdamW, lr_at_step and
clip_grad_norm_ each carry their own tests against PyTorch; what this file adds
is the ORDER, and the order is the part a shape check cannot see.

Taken from PersonaCore training/loop.py:140-159 (_optimizer_step):

    optimizer.zero_grad(set_to_none=True)
    forward -> loss -> backward
    scaler.unscale_(optimizer)            no-op: AMP was never enabled
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer) -> optimizer.step()
    scaler.update()
    scheduler.step()                      AFTER optimizer.step()

Because the scheduler advances after the optimizer, and because LambdaLR runs
one step() during its own construction (leaving lr = base*lambda(0) before the
first iteration), iteration k uses lambda(k-1). With the 0-based `step` taken
here that is lr_at_step(step) — index S uses lambda(S).

Watch out for the corollary: the lr recorded in the checkpoint at step=49000 is
lambda(49000), which no optimizer step in that run ever used. It is the value
staged for iteration 49001. The 49000th iteration ran on lambda(48999).

No gradient accumulation. The real run has grad_accum_steps=1, so the
micro-batch loop there is degenerate — /accum is /1 and the sum before the clip
has one term. One batch per step is the faithful replica; accumulation can come
back when a run needs it.
"""
from core.nn import cross_entropy
from core.optim import clip_grad_norm_, lr_at_step


def train_step(model, optimizer, x, y, step, max_norm, base_lr,
               warmup_steps, max_steps, min_ratio=0.1):
    """Run one iteration and return (loss, pre-clip gradient norm).

    x is (B, T) integer token ids, y the (B*T,) next-token targets — the same
    flattening PersonaCore does before F.cross_entropy.

    The loss returned is measured BEFORE the update, which is what a training
    curve plots: the model's loss on the batch it just learned from, not after.

    The gradient norm is the pre-clip one, matching what torch's
    clip_grad_norm_ returns. It is the number that says whether clipping bites.
    """
    optimizer.zero_grad()

    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y)
    loss.backward()

    grad_norm = clip_grad_norm_(model.parameters(), max_norm)

    # The scheduler's job, done inline: there is no LambdaLR here to mutate the
    # optimizer's lr field, so the step's lr is written straight onto it.
    optimizer.lr = lr_at_step(step, base_lr, warmup_steps, max_steps, min_ratio)
    optimizer.step()

    return float(loss.data), grad_norm
