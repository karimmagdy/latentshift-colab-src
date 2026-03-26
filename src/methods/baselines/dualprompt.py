from __future__ import annotations

"""DualPrompt baseline.

Wang et al., "DualPrompt: Complementary Prompting for Rehearsal-Free
Continual Learning" (ECCV 2022).

Maintains two types of prompts:
  - G-Prompt (general): shared across all tasks, attached to early layers
  - E-Prompt (expert): task-specific, selected via key-matching, attached
    to later layers

Key difference from L2P: separates shared vs task-specific knowledge.
Key difference from LatentShift: modifies input space rather than
projecting in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class DualPrompt(ContinualLearningMethod):
    """DualPrompt: Complementary Prompting for Continual Learning."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        e_pool_size: int = 10,
        e_prompt_length: int = 5,
        g_prompt_length: int = 5,
        top_k: int = 5,
        freeze_encoder: bool = True,
        pull_weight: float = 0.5,
        g_layers: int = 2,
    ):
        super().__init__(encoder, decoder, device)
        self.e_pool_size = e_pool_size
        self.e_prompt_length = e_prompt_length
        self.g_prompt_length = g_prompt_length
        self.top_k = top_k
        self.freeze_encoder = freeze_encoder
        self.pull_weight = pull_weight
        self.g_layers = g_layers

        # Infer encoder dimensions
        if hasattr(encoder, "projection"):
            embed_dim = encoder.projection.in_features
        elif hasattr(encoder, "cls_token"):
            embed_dim = encoder.cls_token.shape[-1]
        else:
            raise ValueError("Cannot infer embed_dim from encoder for DualPrompt")

        self.embed_dim = embed_dim
        num_blocks = len(encoder.blocks) if hasattr(encoder, "blocks") else 6

        # G-Prompt: shared prompts for early layers (g_layers count)
        self.g_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, g_prompt_length, embed_dim, device=device) * 0.02)
            for _ in range(min(g_layers, num_blocks))
        ])

        # E-Prompt pool: task-specific prompts for later layers
        e_layer_count = num_blocks - min(g_layers, num_blocks)
        self.e_prompt_pool = nn.ParameterList([
            nn.Parameter(
                torch.randn(e_pool_size, e_prompt_length, embed_dim, device=device) * 0.02
            )
            for _ in range(e_layer_count)
        ])

        # Keys for E-prompt selection
        self.e_prompt_keys = nn.Parameter(
            torch.randn(e_pool_size, embed_dim, device=device) * 0.02
        )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _get_query_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS token features as query for E-prompt selection."""
        B = x.shape[0]
        with torch.no_grad():
            patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.encoder.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)
            tokens = self.encoder.blocks(tokens)
            tokens = self.encoder.norm(tokens)
        return tokens[:, 0]

    def _select_e_prompts(self, query: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Select top-k expert prompts per layer."""
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(self.e_prompt_keys, dim=-1)
        similarity = query_norm @ key_norm.T  # (B, e_pool_size)
        top_k_sim, top_k_idx = similarity.topk(self.top_k, dim=-1)

        B = query.shape[0]
        selected_per_layer = []
        for pool in self.e_prompt_pool:
            # pool: (e_pool_size, e_prompt_length, embed_dim)
            selected = pool[top_k_idx]  # (B, top_k, e_prompt_length, embed_dim)
            selected = selected.view(B, self.top_k * self.e_prompt_length, self.embed_dim)
            selected_per_layer.append(selected)

        return selected_per_layer, top_k_sim

    def _forward_with_dual_prompts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with G-prompts on early layers and E-prompts on later layers."""
        B = x.shape[0]

        # Get query for E-prompt selection
        query = self._get_query_features(x)
        e_prompts_per_layer, similarity = self._select_e_prompts(query)

        # Patch embedding
        patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)

        num_blocks = len(self.encoder.blocks)
        g_count = len(self.g_prompts)
        e_idx = 0

        # Run through blocks with prompts
        for i in range(num_blocks):
            if i < g_count:
                # G-prompt layers: prepend shared prompts
                g = self.g_prompts[i].expand(B, -1, -1)
                prompted = torch.cat([tokens[:, :1], g, tokens[:, 1:]], dim=1)
                prompted = self.encoder.blocks[i](prompted)
                # Remove G-prompt tokens from sequence
                tokens = torch.cat([prompted[:, :1], prompted[:, 1 + self.g_prompt_length:]], dim=1)
            else:
                # E-prompt layers: prepend selected expert prompts
                ep = e_prompts_per_layer[e_idx]
                prompted = torch.cat([tokens[:, :1], ep, tokens[:, 1:]], dim=1)
                prompted = self.encoder.blocks[i](prompted)
                # Remove E-prompt tokens
                prompt_len = self.top_k * self.e_prompt_length
                tokens = torch.cat([prompted[:, :1], prompted[:, 1 + prompt_len:]], dim=1)
                e_idx += 1

        tokens = self.encoder.norm(tokens)
        cls_out = self.encoder.projection(tokens[:, 0])

        pull_loss = (1.0 - similarity).mean()
        return cls_out, pull_loss

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        params = (
            list(self.g_prompts.parameters())
            + list(self.e_prompt_pool.parameters())
            + [self.e_prompt_keys]
            + list(self.decoder.heads[str(task_id)].parameters())
        )
        if not self.freeze_encoder:
            params += list(self.encoder.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                z, pull_loss = self._forward_with_dual_prompts(x)
                logits = self.decoder(z, task_id)
                ce_loss = criterion(logits, y)
                loss = ce_loss + self.pull_weight * pull_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += ce_loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        pass  # No after-task processing

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        self.encoder.eval()
        self.decoder.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z, _ = self._forward_with_dual_prompts(x)
                logits = self.decoder(z, task_id)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def supports_checkpointing(self) -> bool:
        return True

    def _extra_state_dict(self) -> dict:
        return {
            "g_prompts": [p.detach().cpu() for p in self.g_prompts],
            "e_prompt_pool": [p.detach().cpu() for p in self.e_prompt_pool],
            "e_prompt_keys": self.e_prompt_keys.detach().cpu(),
        }

    def _load_extra_state_dict(self, state: dict) -> None:
        if "g_prompts" in state:
            for p, s in zip(self.g_prompts, state["g_prompts"]):
                p.data.copy_(s.to(self.device))
        if "e_prompt_pool" in state:
            for p, s in zip(self.e_prompt_pool, state["e_prompt_pool"]):
                p.data.copy_(s.to(self.device))
        if "e_prompt_keys" in state:
            self.e_prompt_keys.data.copy_(state["e_prompt_keys"].to(self.device))
