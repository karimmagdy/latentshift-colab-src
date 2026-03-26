from __future__ import annotations

"""CODA-Prompt baseline.

Smith et al., "CODA-Prompt: COntinual Decomposed Attention-based
Prompting for Rehearsal-Free Continual Learning" (CVPR 2023).

Uses attention-weighted combination of prompt components rather than
top-k selection. Each prompt is decomposed into orthogonal components
that are combined via learned attention weights, enabling smoother
knowledge sharing across tasks.

Key difference from L2P: uses soft attention over ALL prompts instead
of hard top-k selection.
Key difference from DualPrompt: single prompt type with decomposed
attention rather than G/E split.
Key difference from LatentShift: operates in input/prompt space rather
than projecting gradients in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class CODAPrompt(ContinualLearningMethod):
    """CODA-Prompt: Continual Decomposed Attention-based Prompting."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        pool_size: int = 10,
        prompt_length: int = 5,
        freeze_encoder: bool = True,
        ortho_weight: float = 0.1,
    ):
        super().__init__(encoder, decoder, device)
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.freeze_encoder = freeze_encoder
        self.ortho_weight = ortho_weight

        # Infer embed_dim from encoder
        if hasattr(encoder, "projection"):
            embed_dim = encoder.projection.in_features
        elif hasattr(encoder, "cls_token"):
            embed_dim = encoder.cls_token.shape[-1]
        else:
            raise ValueError("Cannot infer embed_dim from encoder for CODAPrompt")

        self.embed_dim = embed_dim

        # Prompt components: (pool_size, prompt_length, embed_dim)
        self.prompt_components = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim, device=device) * 0.02
        )

        # Keys for attention-based weighting: (pool_size, embed_dim)
        self.prompt_keys = nn.Parameter(
            torch.randn(pool_size, embed_dim, device=device) * 0.02
        )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _get_query_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS token features as query for attention weighting."""
        B = x.shape[0]
        with torch.no_grad():
            patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.encoder.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)
            tokens = self.encoder.blocks(tokens)
            tokens = self.encoder.norm(tokens)
        return tokens[:, 0]

    def _compose_prompts(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose prompts via attention-weighted combination of components."""
        # query: (B, embed_dim)
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(self.prompt_keys, dim=-1)

        # Attention weights over all components (soft, not hard top-k)
        attn_logits = query_norm @ key_norm.T  # (B, pool_size)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, pool_size)

        # Weighted combination: (B, pool_size, 1, 1) * (pool_size, L, D) → sum → (B, L, D)
        weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # (B, pool_size, 1, 1)
        components = self.prompt_components.unsqueeze(0)  # (1, pool_size, L, D)
        composed = (weights * components).sum(dim=1)  # (B, prompt_length, embed_dim)

        # Orthogonality loss: encourage diverse prompt components
        # Gram matrix of component means
        comp_mean = self.prompt_components.mean(dim=1)  # (pool_size, embed_dim)
        comp_norm = F.normalize(comp_mean, dim=-1)
        gram = comp_norm @ comp_norm.T  # (pool_size, pool_size)
        identity = torch.eye(self.pool_size, device=self.device)
        ortho_loss = ((gram - identity) ** 2).mean()

        return composed, ortho_loss

    def _forward_with_prompts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention-composed prompts."""
        B = x.shape[0]

        # Get query and compose prompts
        query = self._get_query_features(x)
        composed_prompts, ortho_loss = self._compose_prompts(query)

        # Patch embedding
        patches = self.encoder.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.encoder.pos_drop(tokens + self.encoder.pos_embed)

        # Prepend composed prompts after CLS
        tokens = torch.cat([tokens[:, :1], composed_prompts, tokens[:, 1:]], dim=1)

        # Run transformer
        tokens = self.encoder.blocks(tokens)
        tokens = self.encoder.norm(tokens)

        # CLS token → projection
        cls_out = self.encoder.projection(tokens[:, 0])

        return cls_out, ortho_loss

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        params = [self.prompt_components, self.prompt_keys] + list(
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

                z, ortho_loss = self._forward_with_prompts(x)
                logits = self.decoder(z, task_id)
                ce_loss = criterion(logits, y)
                loss = ce_loss + self.ortho_weight * ortho_loss

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
                z, _ = self._forward_with_prompts(x)
                logits = self.decoder(z, task_id)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def supports_checkpointing(self) -> bool:
        return True

    def _extra_state_dict(self) -> dict:
        return {
            "prompt_components": self.prompt_components.detach().cpu(),
            "prompt_keys": self.prompt_keys.detach().cpu(),
        }

    def _load_extra_state_dict(self, state: dict) -> None:
        if "prompt_components" in state:
            self.prompt_components.data.copy_(state["prompt_components"].to(self.device))
        if "prompt_keys" in state:
            self.prompt_keys.data.copy_(state["prompt_keys"].to(self.device))
