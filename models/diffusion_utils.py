# I wrote this file. It has no upstream counterpart.
# It holds the masking schedule and token helpers for the diffusion variant.

import torch

PAD_TOKEN = 0
BLANK_TOKEN = 1
FIRST_DIGIT = 2


def mask_schedule(n_steps: int, kind: str = "cosine", device=None) -> torch.Tensor:
    k = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device)
    if kind == "cosine":
        m = torch.cos(0.5 * torch.pi * k / n_steps)
    elif kind == "linear":
        m = 1.0 - k / n_steps
    else:
        raise ValueError(f"unknown mask schedule {kind}")
    return m.clamp(0.0, 1.0)


def build_input(inputs: torch.Tensor, labels: torch.Tensor, reveal_frac: float,
                order: torch.Tensor, corrupt_rate: float, vocab_size: int,
                generator: torch.Generator = None):
    B, L = inputs.shape
    dev = inputs.device
    answer = (inputs == BLANK_TOKEN)
    n_ans = answer.sum(-1, keepdim=True)

    ord2 = torch.where(answer, order, torch.full_like(order, float("inf")))
    rank = ord2.argsort(dim=-1).argsort(dim=-1)
    reveal_count = (reveal_frac * n_ans.to(torch.float32)).round().long()
    revealed = answer & (rank < reveal_count)

    r = torch.rand(B, L, generator=generator, device=dev)
    corrupt = revealed & (r < corrupt_rate)  # wrong on purpose, teaches revision

    y = torch.where(answer, torch.full_like(inputs, BLANK_TOKEN), inputs)
    y = torch.where(revealed, labels, y)
    n_dig = vocab_size - FIRST_DIGIT
    if n_dig > 1:
        offset = torch.randint(1, n_dig, (B, L), generator=generator, device=dev)
        wrong = (labels - FIRST_DIGIT + offset) % n_dig + FIRST_DIGIT
        y = torch.where(corrupt, wrong, y)

    masked = answer & ~revealed
    weight = (masked | corrupt).to(torch.float32)
    return y, weight


MAZE_GIVENS = (1, 3, 4)
MAZE_OPEN = 2
MAZE_PATH = 5
MAZE_UNKNOWN = 6


def build_input_maze(inputs: torch.Tensor, labels: torch.Tensor, reveal_frac: float,
                     order: torch.Tensor, corrupt_rate: float, generator: torch.Generator = None):
    B, L = inputs.shape
    dev = inputs.device
    answer = (inputs == MAZE_OPEN)
    n_ans = answer.sum(-1, keepdim=True)

    ord2 = torch.where(answer, order, torch.full_like(order, float("inf")))
    rank = ord2.argsort(dim=-1).argsort(dim=-1)
    reveal_count = (reveal_frac * n_ans.to(torch.float32)).round().long()
    revealed = answer & (rank < reveal_count)

    r = torch.rand(B, L, generator=generator, device=dev)
    corrupt = revealed & (r < corrupt_rate)

    y = torch.where(answer, torch.full_like(inputs, MAZE_UNKNOWN), inputs)
    y = torch.where(revealed, labels, y)
    flipped = torch.where(labels == MAZE_PATH, torch.full_like(labels, MAZE_OPEN),
                          torch.full_like(labels, MAZE_PATH))
    y = torch.where(corrupt, flipped, y)

    masked = answer & ~revealed
    weight = (masked | corrupt).to(torch.float32)
    return y, weight


if __name__ == "__main__":
    torch.manual_seed(0)
    V = 11
    inputs = torch.tensor([
        [9, 1, 1, 5, 1, 4, 1, 2, 1],
        [1, 1, 3, 1, 7, 1, 6, 1, 1],
    ])
    labels = torch.tensor([
        [9, 8, 7, 5, 6, 4, 3, 2, 1 + 1],
        [2, 5, 3, 9, 7, 8, 6, 4, 10],
    ])
    order = torch.rand(2, 9)
    sched = mask_schedule(16, "cosine")
    print("cosine schedule (m_k):", [round(x, 3) for x in sched.tolist()])
    assert sched[0] > sched[-1] and abs(sched[-1].item()) < 1e-6, "schedule must go 1->0"

    giv = inputs >= 2
    for rf in [0.0, 0.5, 1.0]:
        y, w = build_input(inputs, labels, rf, order, corrupt_rate=0.0, vocab_size=V)
        assert torch.equal(y[giv], inputs[giv]), "givens must be clamped"
        if rf == 0.0:
            assert torch.equal(y, inputs), "rf=0 must equal the given puzzle"
        if rf == 1.0:
            ans = inputs == BLANK_TOKEN
            assert torch.equal(y[ans], labels[ans]), "rf=1 must reveal full solution"
            assert w.sum() == 0, "rf=1, no corrupt -> nothing left to produce"
        print(f"rf={rf}: revealed answer cells = {((inputs==1)&(y!=1)).sum().item()}, weight cells = {int(w.sum())}")

    y, w = build_input(inputs, labels, 1.0, order, corrupt_rate=1.0, vocab_size=V)
    ans = inputs == BLANK_TOKEN
    assert (y[ans] != labels[ans]).all(), "corrupt_rate=1 must make every revealed answer cell wrong"
    assert (y[ans] >= FIRST_DIGIT).all(), "corruptions must be valid digit tokens"
    assert w[ans].sum() == ans.sum(), "all corrupted answer cells must be in the loss weight"
    print("corrupt_rate=1: all revealed answer cells are wrong digits ✓")

    y_lo, _ = build_input(inputs, labels, 0.3, order, 0.0, V)
    y_hi, _ = build_input(inputs, labels, 0.7, order, 0.0, V)
    rev_lo = (inputs == 1) & (y_lo != 1)
    rev_hi = (inputs == 1) & (y_hi != 1)
    assert (rev_hi | ~rev_lo).all(), "reveals must be cumulative (monotonic)"
    print("monotonic reveal ✓")

    print("\n--- maze ---")
    minp = torch.tensor([
        [1, 2, 2, 3, 2, 1, 2, 2, 4],
        [2, 2, 1, 2, 4, 2, 3, 1, 2],
    ])
    mlab = torch.tensor([
        [1, 5, 5, 3, 2, 1, 2, 5, 4],
        [5, 2, 1, 5, 4, 2, 3, 1, 5],
    ])
    mgiv = torch.isin(minp, torch.tensor(list(MAZE_GIVENS)))
    morder = torch.rand(2, 9)
    for rf in [0.0, 0.5, 1.0]:
        y, w = build_input_maze(minp, mlab, rf, morder, corrupt_rate=0.0)
        assert torch.equal(y[mgiv], minp[mgiv]), "maze givens must be clamped"
        ans = minp == MAZE_OPEN
        if rf == 0.0:
            assert (y[ans] == MAZE_UNKNOWN).all(), "rf=0 -> all answer cells UNKNOWN(6)"
        if rf == 1.0:
            assert torch.equal(y[ans], mlab[ans]), "rf=1 -> full solution (path/non-path)"
            assert w.sum() == 0, "rf=1 no corrupt -> nothing to produce"
        print(f"maze rf={rf}: revealed path cells(5)={int((y==MAZE_PATH).sum())}, weight cells={int(w.sum())}")
    y, w = build_input_maze(minp, mlab, 1.0, morder, corrupt_rate=1.0)
    ans = minp == MAZE_OPEN
    assert (y[ans] != mlab[ans]).all(), "corrupt=1 must flip every revealed answer cell"
    assert torch.isin(y[ans], torch.tensor([MAZE_OPEN, MAZE_PATH])).all(), "maze answer tokens must be 2 or 5"
    print("maze corrupt=1: all revealed answer cells flipped ✓")
    print("\nALL DIFFUSION-UTILS SELF-TESTS PASSED")
