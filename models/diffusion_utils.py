"""
Approach C — masked/inpainting discrete-diffusion utilities (future_process.md §5).

Token convention (Sudoku-Extreme, verified): 0=PAD, 1=BLANK (answer cell to fill),
2..(1+n_digits)=digits. Givens = (inputs >= 2) and are NEVER masked (clamped every step).
Only answer cells (inputs == 1) are diffused.

Two public functions:
  mask_schedule(n_steps, kind)  -> fraction-still-masked m_k for k=0..n-1 (1.0 -> 0.0)
  build_input(inputs, labels, reveal_frac, order, corrupt_rate, vocab_size)
      -> (y_k token grid, score_weight)   # the teacher-forced partially-revealed board

Design (T20-locked): revision must be LEARNED, so a small fraction of REVEALED cells is
corrupted to a WRONG digit (not the true value) — this teaches the model to overwrite a
confident-wrong commit, the exact un-freeze capability the MaskGIT sampler needs at test.
"""
import torch

PAD_TOKEN = 0
BLANK_TOKEN = 1
FIRST_DIGIT = 2   # digit d (1..9) -> token d+1


def mask_schedule(n_steps: int, kind: str = "cosine", device=None) -> torch.Tensor:
    """Fraction of answer cells STILL masked at each of the n_steps denoising steps.
    m[0] ~ 1.0 (start = the given puzzle, all answer cells blank) -> m[n-1] = 0.0 (fully revealed).
    """
    k = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device)
    if kind == "cosine":
        m = torch.cos(0.5 * torch.pi * k / n_steps)      # reveal slow early, fast late
    elif kind == "linear":
        m = 1.0 - k / n_steps
    else:
        raise ValueError(f"unknown mask schedule {kind}")
    return m.clamp(0.0, 1.0)


def build_input(inputs: torch.Tensor, labels: torch.Tensor, reveal_frac: float,
                order: torch.Tensor, corrupt_rate: float, vocab_size: int,
                generator: torch.Generator = None):
    """Construct the step-k teacher-forced board.

    inputs  [B,L] clue grid  (BLANK at answer cells, >=2 at givens)
    labels  [B,L] full solution tokens (digits 2..)
    reveal_frac scalar in [0,1]: fraction of each puzzle's answer cells revealed this step
    order   [B,L] per-cell random priority (lower -> revealed earlier); fixed across the
            trajectory so reveals are CUMULATIVE as reveal_frac grows
    corrupt_rate scalar: fraction of revealed cells set to a WRONG digit
    Returns:
      y       [B,L] tokens  (givens clamped + revealed solution + corruption + blanks)
      weight  [B,L] float   (1.0 on cells the model must produce this step = masked OR corrupted)
    """
    B, L = inputs.shape
    dev = inputs.device
    answer = (inputs == BLANK_TOKEN)                              # [B,L] diffused cells
    n_ans = answer.sum(-1, keepdim=True)                          # [B,1]

    # rank answer cells by `order`; givens pushed to the back so they never count as revealed
    ord2 = torch.where(answer, order, torch.full_like(order, float("inf")))
    rank = ord2.argsort(dim=-1).argsort(dim=-1)                   # [B,L] 0..L-1, answer cells first
    reveal_count = (reveal_frac * n_ans.to(torch.float32)).round().long()  # [B,1]
    revealed = answer & (rank < reveal_count)                    # [B,L]

    # corruption among revealed cells
    r = torch.rand(B, L, generator=generator, device=dev)
    corrupt = revealed & (r < corrupt_rate)

    # base board: givens kept, all answer cells blank
    y = torch.where(answer, torch.full_like(inputs, BLANK_TOKEN), inputs)
    # reveal -> true solution
    y = torch.where(revealed, labels, y)
    # corrupt revealed -> a different digit token in [FIRST_DIGIT, vocab_size-1]
    n_dig = vocab_size - FIRST_DIGIT
    if n_dig > 1:
        offset = torch.randint(1, n_dig, (B, L), generator=generator, device=dev)
        wrong = (labels - FIRST_DIGIT + offset) % n_dig + FIRST_DIGIT
        y = torch.where(corrupt, wrong, y)

    masked = answer & ~revealed
    weight = (masked | corrupt).to(torch.float32)
    return y, weight


# ===================================================================== MAZE
# Maze-Hard token convention (verified): 1=wall, 2=open, 3=start, 4=goal, 5=path.
# Givens (never masked) = {1,3,4} (structure). Answer cells = open (==2); the solution
# marks ~111 of them as path (2->5). We add an UNKNOWN token (6 -> vocab 7) for masked
# answer cells, so "undecided" is distinct from "decided non-path" (token 2) — the clean
# analog of Sudoku's blank. Model predicts 2 (non-path) or 5 (path) per answer cell.
MAZE_GIVENS = (1, 3, 4)
MAZE_OPEN = 2
MAZE_PATH = 5
MAZE_UNKNOWN = 6   # intermediate-only token (never a label); requires vocab_size >= 7


def build_input_maze(inputs: torch.Tensor, labels: torch.Tensor, reveal_frac: float,
                     order: torch.Tensor, corrupt_rate: float, generator: torch.Generator = None):
    """Maze step-k teacher-forced board (inpaint the path).
    Masked answer cells -> UNKNOWN(6); revealed -> solution (path 5 / non-path 2);
    corrupted revealed -> FLIPPED (5<->2). Givens (walls/start/goal) clamped.
    Returns y [B,L] tokens, weight [B,L] (1.0 on cells the model must produce = masked OR corrupted).
    """
    B, L = inputs.shape
    dev = inputs.device
    answer = (inputs == MAZE_OPEN)                               # open cells = diffused
    n_ans = answer.sum(-1, keepdim=True)

    ord2 = torch.where(answer, order, torch.full_like(order, float("inf")))
    rank = ord2.argsort(dim=-1).argsort(dim=-1)
    reveal_count = (reveal_frac * n_ans.to(torch.float32)).round().long()
    revealed = answer & (rank < reveal_count)

    r = torch.rand(B, L, generator=generator, device=dev)
    corrupt = revealed & (r < corrupt_rate)

    # base board: givens kept, answer cells -> UNKNOWN
    y = torch.where(answer, torch.full_like(inputs, MAZE_UNKNOWN), inputs)
    # reveal -> true label (path 5 or non-path 2)
    y = torch.where(revealed, labels, y)
    # corrupt -> flip path<->non-path
    flipped = torch.where(labels == MAZE_PATH, torch.full_like(labels, MAZE_OPEN),
                          torch.full_like(labels, MAZE_PATH))
    y = torch.where(corrupt, flipped, y)

    masked = answer & ~revealed
    weight = (masked | corrupt).to(torch.float32)
    return y, weight


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    V = 11  # sudoku vocab (0=PAD,1=BLANK,2..10=digits)
    # toy: 2 puzzles, L=9, some givens some blanks
    inputs = torch.tensor([
        [9, 1, 1, 5, 1, 4, 1, 2, 1],   # blanks at idx 1,2,4,6,8
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
        # givens always preserved
        assert torch.equal(y[giv], inputs[giv]), "givens must be clamped"
        # reveal_frac=0 -> all answer cells blank (== the given puzzle)
        if rf == 0.0:
            assert torch.equal(y, inputs), "rf=0 must equal the given puzzle"
        # reveal_frac=1 -> all answer cells = solution (no corruption)
        if rf == 1.0:
            ans = inputs == BLANK_TOKEN
            assert torch.equal(y[ans], labels[ans]), "rf=1 must reveal full solution"
            assert w.sum() == 0, "rf=1, no corrupt -> nothing left to produce"
        print(f"rf={rf}: revealed answer cells = {((inputs==1)&(y!=1)).sum().item()}, weight cells = {int(w.sum())}")

    # corruption produces WRONG digits among revealed
    y, w = build_input(inputs, labels, 1.0, order, corrupt_rate=1.0, vocab_size=V)
    ans = inputs == BLANK_TOKEN
    assert (y[ans] != labels[ans]).all(), "corrupt_rate=1 must make every revealed answer cell wrong"
    assert (y[ans] >= FIRST_DIGIT).all(), "corruptions must be valid digit tokens"
    assert w[ans].sum() == ans.sum(), "all corrupted answer cells must be in the loss weight"
    print("corrupt_rate=1: all revealed answer cells are wrong digits ✓")

    # monotonic reveal: cells revealed at low rf stay revealed at higher rf
    y_lo, _ = build_input(inputs, labels, 0.3, order, 0.0, V)
    y_hi, _ = build_input(inputs, labels, 0.7, order, 0.0, V)
    rev_lo = (inputs == 1) & (y_lo != 1)
    rev_hi = (inputs == 1) & (y_hi != 1)
    assert (rev_hi | ~rev_lo).all(), "reveals must be cumulative (monotonic)"
    print("monotonic reveal ✓")

    # ---- MAZE build_input ----
    print("\n--- maze ---")
    minp = torch.tensor([
        [1, 2, 2, 3, 2, 1, 2, 2, 4],   # walls=1, open=2, start=3, goal=4
        [2, 2, 1, 2, 4, 2, 3, 1, 2],
    ])
    mlab = torch.tensor([
        [1, 5, 5, 3, 2, 1, 2, 5, 4],   # path=5 marks some open cells
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
    # corruption flips path<->nonpath
    y, w = build_input_maze(minp, mlab, 1.0, morder, corrupt_rate=1.0)
    ans = minp == MAZE_OPEN
    assert (y[ans] != mlab[ans]).all(), "corrupt=1 must flip every revealed answer cell"
    assert torch.isin(y[ans], torch.tensor([MAZE_OPEN, MAZE_PATH])).all(), "maze answer tokens must be 2 or 5"
    print("maze corrupt=1: all revealed answer cells flipped ✓")
    print("\nALL DIFFUSION-UTILS SELF-TESTS PASSED")
