from __future__ import annotations

"""Learning to Prompt (L2P) baseline.

Wang et al., "Learning to Prompt for Continual Learning" (CVPR 2022).

Maintains a shared prompt pool. At inference, a lightweight key-matching
mechanism selects top-k prompts to prepend to the ViT input tokens.
Only prompts and the classifier head are trained; the encoder backbone
can optionally be frozen.

Key difference from LatentShift: L2P steers representations via
input-space prompts rather than projecting gradients in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class L2P(ContinualLearningMethod):
    """Learning to Prompt for Continual Learning."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        pool_size: int = 10,
        prompt_length: int = 5,
        top_k: int = 5,
        freeze_encoder: bool = True,
        pull_weight: float = 0.5,
    ):
        super().__init__(encoder, decoder, device)
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.freeze_encoder = freeze_encoder
        self.pull_weight = pull_weight

        # Infer embed_dim from encoder
        if hasattr(encoder, "projection"):
            embed_dim = encoder.projection.in_features
        elif hasattr(encoder, "cls_token"):
            embed_dim = encoder.cls_token.shape[-1]
        else:
            raise ValueError("Cannot infer embed_dim from encoder for L2P")

        self.embed_dim = embed_dim

        # Prompt pool: (pool_size, prompt_length, embed_dim)
        self.prompt_pool = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim, device=device) * 0.02
        )
        # Keys for prompt selection: (pool_size, embed_dim)
        self.prompt_keys = nn.Parameter(
            torch.randn(pool_size, embed_dim, device=device) * 0.02
        )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _get_query_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS token features before projection as query for prompt selection."""
        B = x.shape[0]
        with torch.no_grad():
            patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.encoder.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)
            tokens = self.encoder.blocks(tokens)
            tokens = self.encoder.norm(tokens)
        return tokens[:, 0]  # CLS token: (B, embed_dim)

    def _select_prompts(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k prompts from pool based on cosine similarity with query."""
        # query: (B, embed_dim), prompt_keys: (pool_size, embed_dim)
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(self.prompt_keys, dim=-1)
        similarity = query_norm @ key_norm.T  # (B, pool_size)

        top_k_sim, top_k_idx = similarity.topk(self.top_k, dim=-1)  # (B, top_k)

        # Gather selected prompts: (B, top_k, prompt_length, embed_dim)
        selected = self.prompt_pool[top_k_idx]  # fancy indexing
        # Reshape to (B, top_k * prompt_length, embed_dim)
        B = query.shape[0]
        selected = selected.view(B, self.top_k * self.prompt_length, self.embed_dim)

        return selected, top_k_sim

    def _forward_with_prompts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: select prompts, prepend to tokens, run encoder."""
        B = x.shape[0]

        # Get query for prompt selection
        query = self._get_query_features(x)

        # Select prompts
        prompts, similarity = self._select_prompts(query)  # (B, K*L, D)

        # Build token sequence with prompts prepended
        patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)

        # Prepend prompts after positional encoding
        tokens = torch.cat([tokens[:, :1], prompts, tokens[:, 1:]], dim=1)

        # Run through transformer blocks
        tokens = self.encoder.blocks(tokens)
        tokens = self.encoder.norm(tokens)

        # CLS token output → projection
        cls_out = self.encoder.projection(tokens[:, 0])

        # Pull constraint loss: encourage selected keys to be close to query
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

        # Only train prompts, keys, and task head (+ encoder if not frozen)
        params = [self.prompt_pool, self.prompt_keys] + list(
            self.decoder.heads[str(task_id)].parameters()
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

                z, pull_loss = self._forward_with_prompts(x)
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
        pass  # L2P has no after-task processing; prompt pool is shared

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        self.encoder.eval()
        self.decoder.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z, _ = self._forward_with_prompts(x)
                logits = self.decoder(z, task_id)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def supports_checkpointing(self) -> bool:
        return True

    def _extra_state_dict(self) -> dict:
        return {
            "prompt_pool": self.prompt_pool.detach().cpu(),
            "prompt_keys": self.prompt_keys.detach().cpu(),
        }

    def _load_extra_state_dict(self, state: dict) -> None:
        if "prompt_pool" in state:
            self.prompt_pool.data.copy_(state["prompt_pool"].to(self.device))
        if "prompt_keys" in state:
            self.prompt_keys.data.copy_(state["prompt_keys"].to(self.device))
