"""
A1 — STOCHASTIC-RECURSION TRM (future_process.md §4).

Goal: make branching INTRINSIC. Turn TRM's deterministic fixed-point iteration
into a stochastic sampler, so repeated rollouts explore different completions of
the search subtree (the ~31 frozen cells). Paired with the winner-take-all
multiple-choice-learning loss head (`losses.MCLLossHead`), training pushes at
least one of N rollouts to be correct; at test time we draw N rollouts and keep
the one the verifier / the model's own q_halt accepts (no ground truth needed).

This is a faithful copy of `trm.py` (same MLP-mixer / attention block, deep
supervision, ACT halting, EMA-compatible) with TWO additions:
  1. Additive Gaussian noise injected into the latent z_L (and optionally z_H)
     at every L-cycle:  z = net(...) + sigma * eps,  eps ~ N(0, I).  The noise is
     per-element, so N copies of the same puzzle diverge into N distinct rollouts.
  2. The ACT wrapper TILES the batch x N and halts/refills PER PUZZLE (all N
     copies of a puzzle share current_data and a single halt decision), so the N
     rollouts stay coherent across ACT steps under the 1-step-gradient mechanism.

Reduces EXACTLY to the deterministic baseline when n_samples=1 and noise_sigma=0.

Config knobs (all via config/arch/trm_stochastic.yaml -> arch.__pydantic_extra__):
  n_samples (N), noise_sigma, noise_mode {"zL","both"}, eval_noise (bool).
"""
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.common import trunc_normal_init_

IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # --- A1 stochastic-recursion knobs ---
    n_samples: int = 1            # N rollouts per puzzle (batch is tiled x N internally)
    noise_sigma: float = 0.0      # std of additive Gaussian noise on z
    noise_mode: str = "zL"        # "zL" (noise on z_L only) or "both" (z_L and z_H)
    eval_noise: bool = False      # if True, keep noise active in eval (for best-of-N eval)


# ---- block / reasoning module: identical to trm.py ----
class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(hidden_size=self.config.seq_len + self.puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size, head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads, num_key_value_heads=config.num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            # the wrapper tiles each puzzle into N rollouts, so the inner model
            # processes batch_size * n_samples rows per step -> the sparse-embedding
            # scratch buffer (local_weights/local_ids) must be sized accordingly.
            # The sparse optimizer de-duplicates ids (unique + scatter_add), so the
            # N duplicate ids per puzzle correctly sum their gradients.
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size * self.config.n_samples,
                                                    init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _noise(self, z, active):
        if active and self.config.noise_sigma > 0:
            return z + self.config.noise_sigma * torch.randn_like(z)
        return z

    def forward(self, carry, batch):
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # noise active in training, or in eval iff eval_noise (for best-of-N eval)
        noise_on = (self.training or self.config.eval_noise)
        noise_zL = noise_on
        noise_zH = noise_on and (self.config.noise_mode == "both")

        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self._noise(self.L_level(z_L, z_H + input_embeddings, **seq_info), noise_zL)
                z_H = self._noise(self.L_level(z_H, z_L, **seq_info), noise_zH)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self._noise(self.L_level(z_L, z_H + input_embeddings, **seq_info), noise_zL)
        z_H = self._noise(self.L_level(z_H, z_L, **seq_info), noise_zH)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper with N-rollout tiling + per-puzzle synchronized halting."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def _eff_N(self):
        """N rollouts when TRAINING; 1 at eval (noise is off there, so the N copies would
        be identical -> tiling at eval is pure waste; this makes eval ~N x faster)."""
        return self.config.n_samples if self.training else 1

    def _tile(self, t):
        """Repeat each puzzle N times contiguously: puzzle b -> rows [b*N : b*N+N]."""
        N = self._eff_N()
        if N == 1:
            return t
        return t.repeat_interleave(N, dim=0)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        N = self._eff_N()
        bn = batch["inputs"].shape[0] * N
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(bn),  # reset on first pass (all halted)
            steps=torch.zeros((bn,), dtype=torch.int32),
            halted=torch.ones((bn,), dtype=torch.bool),
            current_data={k: torch.empty((bn, *v.shape[1:]), dtype=v.dtype, device=v.device) for k, v in batch.items()},
        )

    def _puzzle_sync(self, flag_bn):
        """Collapse a [B*N] bool to a per-puzzle decision and broadcast back to [B*N].
        A puzzle halts when ANY of its N rollouts would halt (keeps copies coherent)."""
        N = self._eff_N()
        if N == 1:
            return flag_bn
        per_puzzle = flag_bn.view(-1, N).any(dim=1, keepdim=True)
        return per_puzzle.expand(-1, N).reshape(-1)

    def forward(self, carry, batch):
        tiled_batch = {k: self._tile(v) for k, v in batch.items()}

        # reset halted slots; refill their data identically across the N copies
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (tiled_batch[k].ndim - 1)), tiled_batch[k], v)
                            for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halt_signal = q_halt_logits > 0
                else:
                    halt_signal = q_halt_logits > q_continue_logits
                halted = halted | halt_signal
                # exploration: random min number of steps (per puzzle, broadcast to N)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)
                # keep the N copies of each puzzle synchronized
                halted = self._puzzle_sync(halted)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
