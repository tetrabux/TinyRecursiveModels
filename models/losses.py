from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, fixed_point_weight: float = 0.0, focal_gamma: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.fixed_point_weight = fixed_point_weight
        self.focal_gamma = focal_gamma
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_cells = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        if self.focal_gamma > 0:
            # weigh the cells the model is unsure of more
            with torch.no_grad():
                p = torch.softmax(outputs["logits"].to(torch.float32), dim=-1)
                idx = torch.where(mask, labels, 0).to(torch.long).unsqueeze(-1)
                p_true = p.gather(-1, idx).squeeze(-1)
                w = ((1.0 - p_true).clamp_min(1e-6) ** self.focal_gamma) * mask
                w = w * (loss_counts.unsqueeze(-1) / w.sum(-1, keepdim=True).clamp_min(1e-6))
            lm_cells = lm_cells * w
        lm_loss = (lm_cells / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        fp_penalty = outputs.get("fp_penalty", None)  # only set by the fixed-point variant
        if fp_penalty is not None and self.fixed_point_weight > 0:
            bsz = outputs["logits"].shape[0]
            total_loss = total_loss + self.fixed_point_weight * fp_penalty * bsz
            metrics["fp_penalty"] = (fp_penalty * bsz).detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class DiffusionLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy"):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        from models.diffusion_utils import (BLANK_TOKEN, FIRST_DIGIT, MAZE_OPEN, MAZE_PATH, MAZE_UNKNOWN)
        self._BLANK, self._FIRST_DIGIT = BLANK_TOKEN, FIRST_DIGIT
        self._MAZE_OPEN, self._MAZE_PATH, self._MAZE_UNKNOWN = MAZE_OPEN, MAZE_PATH, MAZE_UNKNOWN
        c = model.config
        self.n = c.halt_max_steps
        self.task = c.task
        self.kind = c.mask_schedule_kind
        self.corrupt_rate = c.corrupt_rate
        self.base_w = c.base_loss_weight
        self.vocab = c.vocab_size
        self.n_infer = c.n_infer_steps or c.halt_max_steps

    def initial_carry(self, *a, **k):
        return None

    def _answer_mask(self, inputs):
        return (inputs == self._MAZE_OPEN) if self.task == "maze" else (inputs == self._BLANK)

    def _valid_tokens(self):
        return [self._MAZE_OPEN, self._MAZE_PATH] if self.task == "maze" else list(range(self._FIRST_DIGIT, self.vocab))

    def _build(self, inputs, labels, reveal, order):
        from models.diffusion_utils import build_input, build_input_maze
        if self.task == "maze":
            return build_input_maze(inputs, labels, reveal, order, self.corrupt_rate)
        return build_input(inputs, labels, reveal, order, self.corrupt_rate, self.vocab)

    def forward(self, return_keys: Sequence[str], carry=None, batch=None, **kw):
        if batch is None:
            batch = kw["batch"]
        if self.model.training:
            return self._train(batch, return_keys)
        with torch.no_grad():
            return self._sample(batch, return_keys)

    def _train(self, batch, return_keys):
        from models.diffusion_utils import mask_schedule
        inputs, labels = batch["inputs"], batch["labels"]
        pids = batch["puzzle_identifiers"]
        B, L = inputs.shape
        dev = inputs.device
        sched = mask_schedule(self.n, self.kind, dev)
        order = torch.rand(B, L, device=dev)
        pe = self.model.embed_puzzle(pids)
        z = self.model.initial_z(B, dev)
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1).clamp_min(1)

        total = inputs.new_zeros((), dtype=torch.float32)
        last_logits = last_q = None
        for k in range(self.n):
            reveal = 1.0 - sched[k].item()
            y_k, w_k = self._build(inputs, labels, reveal, order)
            z, logits, q = self.model.denoise_step(z[0], z[1], y_k, pe, k)
            ce = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
            weight = (w_k + self.base_w * (1.0 - w_k)) * mask
            lm_k = (ce * weight).sum(-1) / weight.sum(-1).clamp_min(1e-6)
            with torch.no_grad():
                seq_ok = ((logits.argmax(-1) == labels) | ~mask).all(-1).to(q.dtype)
            q_k = F.binary_cross_entropy_with_logits(q, seq_ok, reduction="none")
            total = total + lm_k.sum() + 0.5 * q_k.sum()
            z = (z[0].detach(), z[1].detach())
            last_logits, last_q = logits, q

        with torch.no_grad():
            preds = last_logits.argmax(-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == mask.sum(-1)
            metrics = {
                "count": torch.tensor(B, device=dev),
                "accuracy": (is_correct.float().sum(-1) / loss_counts).sum(),
                "exact_accuracy": seq_is_correct.sum(),
                "q_halt_accuracy": ((last_q >= 0) == seq_is_correct).sum(),
                "steps": torch.tensor(self.n * B, device=dev),
                "lm_loss": total.detach(),
            }
        return None, total, metrics, {}, True

    def _sample(self, batch, return_keys):
        from models.diffusion_utils import mask_schedule
        inputs, labels = batch["inputs"], batch["labels"]
        pids = batch["puzzle_identifiers"]
        B, L = inputs.shape
        dev = inputs.device
        n = self.n_infer
        sched = mask_schedule(n, self.kind, dev)
        pe = self.model.embed_puzzle(pids)
        z = self.model.initial_z(B, dev)
        ans = self._answer_mask(inputs)
        n_ans = ans.sum(-1).to(torch.float32)
        UNK = self._MAZE_UNKNOWN if self.task == "maze" else self._BLANK
        valid = self._valid_tokens()
        grid = torch.where(ans, torch.full_like(inputs, UNK), inputs)
        last_q = None
        last_logits = None
        for k in range(n):
            z, logits, q = self.model.denoise_step(z[0], z[1], grid, pe, k)
            last_q, last_logits = q, logits
            vlog = torch.full_like(logits, float("-inf"), dtype=torch.float32)
            for t in valid:
                vlog[..., t] = logits[..., t].float()
            probs = torch.softmax(vlog, dim=-1)
            conf, pred = probs.max(-1)
            conf = torch.where(ans, conf, torch.full_like(conf, -1.0))
            target = ((1.0 - sched[k]) * n_ans).round().long()
            rank = conf.argsort(-1, descending=True).argsort(-1)
            commit = ans & (rank < target.unsqueeze(-1))
            grid = torch.where(commit, pred, torch.where(ans, torch.full_like(grid, UNK), grid))

        preds = torch.where(ans, grid, inputs)
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1).clamp_min(1)
        is_correct = mask & (preds == labels)
        seq_is_correct = is_correct.sum(-1) == mask.sum(-1)
        metrics = {
            "count": torch.tensor(B, device=dev),
            "accuracy": (is_correct.float().sum(-1) / loss_counts).sum(),
            "exact_accuracy": seq_is_correct.sum(),
            "q_halt_accuracy": ((last_q >= 0) == seq_is_correct).sum(),
            "steps": torch.tensor(n * B, device=dev),
        }
        outputs = {"preds": preds, "logits": last_logits, "q_halt_logits": last_q}
        detached = {k_: outputs[k_] for k_ in return_keys if k_ in outputs}
        return None, inputs.new_zeros((), dtype=torch.float32), metrics, detached, True


class MCLLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, mean_weight: float = 0.1):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.mean_weight = mean_weight

    @property
    def N(self):
        return self.model.config.n_samples if self.model.training else 1

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, return_keys: Sequence[str], **model_kwargs):
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        logits = outputs["logits"]
        q_halt = outputs["q_halt_logits"]
        N = self.N
        BN = labels.shape[0]
        assert BN % N == 0, f"batch {BN} not divisible by n_samples {N}"
        B = BN // N

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            seq_is_correct = (mask & (outputs["preds"] == labels)).sum(-1) == loss_counts

            halted_p = new_carry.halted.view(B, N)[:, 0]
            valid_p = halted_p & (loss_counts.view(B, N)[:, 0] > 0)
            sc_view = seq_is_correct.view(B, N)
            best_of_N = sc_view.any(dim=1)
            cell_frac = (mask & (outputs["preds"] == labels)).sum(-1).float() / loss_counts.clamp_min(1)
            best_cell = cell_frac.view(B, N).max(dim=1).values
            qh_correct = ((q_halt >= 0) == seq_is_correct).view(B, N).float().mean(dim=1)
            metrics = {
                "count": valid_p.sum(),
                "accuracy": torch.where(valid_p, best_cell, 0).sum(),
                "exact_accuracy": (valid_p & best_of_N).sum(),
                "q_halt_accuracy": torch.where(valid_p, qh_correct, 0).sum(),
                "steps": torch.where(valid_p, new_carry.steps.view(B, N)[:, 0], 0).sum(),
            }

        divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        lm_cells = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        per_rollout = (lm_cells / divisor).sum(-1).view(B, N)
        # backprop the best rollout, small nudge from the rest
        lm_loss = per_rollout.min(dim=1).values.sum() + self.mean_weight * per_rollout.mean(dim=1).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt, seq_is_correct.to(q_halt.dtype), reduction="sum") / N

        total_loss = lm_loss + 0.5 * q_halt_loss
        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

