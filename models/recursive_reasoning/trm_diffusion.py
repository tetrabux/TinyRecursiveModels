from typing import List, Dict, Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.layers import CastedEmbedding, CastedLinear, RotaryEmbedding
from models.sparse_embedding import CastedSparseEmbedding
from models.common import trunc_normal_init_
from models.recursive_reasoning.trm_stochastic import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)
from models.diffusion_utils import MAZE_UNKNOWN


class TinyRecursiveReasoningModel_DiffusionConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float = 0.0

    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    task: str = "maze"
    mask_schedule_kind: str = "cosine"
    corrupt_rate: float = 0.1
    base_loss_weight: float = 0.1
    n_infer_steps: int = 0


class TinyRecursiveReasoningModel_Diffusion(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_DiffusionConfig(**config_dict)
        c = self.config
        self.forward_dtype = getattr(torch, c.forward_dtype)
        self.embed_scale = math.sqrt(c.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.puzzle_emb_len = -(c.puzzle_emb_ndim // -c.hidden_size) if c.puzzle_emb_len == 0 else c.puzzle_emb_len

        if c.task == "maze":
            assert MAZE_UNKNOWN == c.vocab_size, f"maze UNKNOWN={MAZE_UNKNOWN} must equal data vocab_size={c.vocab_size}"
            self.emb_vocab = c.vocab_size + 1
        else:
            self.emb_vocab = c.vocab_size

        self.embed_tokens = CastedEmbedding(self.emb_vocab, c.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.step_emb = CastedEmbedding(c.halt_max_steps, c.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(c.hidden_size, self.emb_vocab, bias=False)
        self.q_head = CastedLinear(c.hidden_size, 2, bias=True)

        if c.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(c.num_puzzle_identifiers, c.puzzle_emb_ndim,
                                                    batch_size=c.batch_size, init_std=0, cast_to=self.forward_dtype)

        if c.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=c.hidden_size // c.num_heads,
                                              max_position_embeddings=c.seq_len + self.puzzle_emb_len,
                                              base=c.rope_theta)
        elif c.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(c.seq_len + self.puzzle_emb_len, c.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(c.L_layers)])

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(c.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(c.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def embed_puzzle(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        if self.config.puzzle_emb_ndim <= 0:
            return None
        pe = self.puzzle_emb(puzzle_identifiers)
        pad = self.puzzle_emb_len * self.config.hidden_size - pe.shape[-1]
        if pad > 0:
            pe = F.pad(pe, (0, pad))
        return pe.view(-1, self.puzzle_emb_len, self.config.hidden_size)

    def initial_z(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        z_H = self.H_init.to(device).expand(shape).contiguous()
        z_L = self.L_init.to(device).expand(shape).contiguous()
        return z_H, z_L

    def _input_embeddings(self, grid: torch.Tensor, puzzle_embedding, step_idx: int) -> torch.Tensor:
        emb = self.embed_tokens(grid.to(torch.int32))
        if puzzle_embedding is not None:
            emb = torch.cat((puzzle_embedding, emb), dim=-2)
        k = torch.as_tensor(step_idx, device=grid.device, dtype=torch.int32)
        emb = emb + self.step_emb(k).view(1, 1, -1)  # tells the net how revealed the board is
        if self.config.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * emb

    def denoise_step(self, z_H, z_L, grid, puzzle_embedding, step_idx: int):
        # one pass over the board as it currently stands
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_emb = self._input_embeddings(grid, puzzle_embedding, step_idx)

        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_emb, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_halt = self.q_head(z_H[:, 0]).to(torch.float32)[..., 0]
        return (z_H.detach(), z_L.detach()), logits, q_halt
