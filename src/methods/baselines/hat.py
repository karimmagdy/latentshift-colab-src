from __future__ import annotations

"""Hard Attention to the Task (HAT) baseline.

Serra et al., "Overcoming Catastrophic Forgetting with Hard Attention
to the Task" (ICML 2018).

Learns task-specific soft attention masks for each layer. At task boundary,
masks are binarized and accumulated. During training on new tasks, a
regularization term encourages new masks to use capacity not yet claimed
by previous tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ContinualLearningMethod


class HATMaskLayer(nn.Module):
    """A single learnable mask for one encoder layer.

    For a layer with `num_units` output features, this holds a real-valued
    embedding that is passed through sigmoid(s * e) to produce the mask.
    """

    def __init__(self, num_units: int, num_tasks: int = 50):
        super().__init__()
        # Task embeddings: (num_tasks, num_units), one row per task
        self.embeddings = nn.Parameter(torch.empty(num_tasks, num_units))
        nn.init.uniform_(self.embeddings, -1.0, 1.0)

    def forward(self, task_id: int, s: float) -> torch.Tensor:
        """Return gate values in [0, 1] for the given task and scaling factor."""
        return torch.sigmoid(s * self.embeddings[task_id])


class HATEncoder(nn.Module):
    """Wraps an MLP or ResNet encoder to apply HAT-style task-conditioned masks.

    For MLP: masks are applied after each linear layer (before activation).
    For ResNet: BatchNorm is replaced with GroupNorm (no running stats leak),
    and masks are applied at the BasicBlock output level (after residual add)
    to properly handle skip connections.
    """

    def __init__(self, base_encoder: nn.Module, num_tasks: int = 50):
        super().__init__()
        self.base_encoder = base_encoder
        self.latent_dim = base_encoder.latent_dim
        self.mask_layers: nn.ModuleDict = nn.ModuleDict()
        self._layer_order: list[str] = []
        self._is_resnet = hasattr(base_encoder, "backbone")

        if self._is_resnet:
            self._prepare_resnet(num_tasks)
        else:
            self._prepare_mlp(num_tasks)

        # Storage for hooks
        self._hooks: list = []
        self._current_task: int = 0
        self._current_s: float = 1.0

    def _prepare_mlp(self, num_tasks: int) -> None:
        """For MLP encoders: mask each Linear layer."""
        for name, module in self.base_encoder.named_modules():
            if isinstance(module, nn.Linear):
                num_units = module.out_features
                safe_name = name.replace(".", "_")
                self.mask_layers[safe_name] = HATMaskLayer(num_units, num_tasks)
                self._layer_order.append(safe_name)

    def _prepare_resnet(self, num_tasks: int) -> None:
        """For ResNet encoders: replace BatchNorm with GroupNorm and mask at
        block level to handle residual connections correctly."""
        self._replace_batchnorm(self.base_encoder)

        # Mask at block level: layer1-layer4 each contain BasicBlocks.
        # Also mask the initial conv and the final projection.
        backbone = self.base_encoder.backbone

        # Initial conv output (64 channels)
        self.mask_layers["conv1"] = HATMaskLayer(64, num_tasks)
        self._layer_order.append("conv1")
        self._conv1_target = backbone.conv1

        # Each residual block output
        self._block_targets: list[tuple[str, nn.Module]] = []
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(backbone, layer_name)
            for block_idx, block in enumerate(layer):
                safe_name = f"{layer_name}_block{block_idx}"
                out_channels = block.conv2.out_channels
                self.mask_layers[safe_name] = HATMaskLayer(out_channels, num_tasks)
                self._layer_order.append(safe_name)
                self._block_targets.append((safe_name, block))

        # Final projection layer
        proj = self.base_encoder.projection
        self.mask_layers["projection"] = HATMaskLayer(proj.out_features, num_tasks)
        self._layer_order.append("projection")

    @staticmethod
    def _replace_batchnorm(module: nn.Module) -> None:
        """Recursively replace all BatchNorm2d/1d with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                num_groups = min(32, num_channels)
                # Ensure num_channels is divisible by num_groups
                while num_channels % num_groups != 0:
                    num_groups //= 2
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            elif isinstance(child, nn.BatchNorm1d):
                num_features = child.num_features
                setattr(module, name, nn.GroupNorm(1, num_features))
            else:
                HATEncoder._replace_batchnorm(child)

    def set_task(self, task_id: int, s: float) -> None:
        self._current_task = task_id
        self._current_s = s

    def get_masks(self, task_id: int, s: float) -> dict[str, torch.Tensor]:
        """Return current mask values for all layers."""
        return {
            name: self.mask_layers[name](task_id, s)
            for name in self._layer_order
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_resnet:
            return self._forward_resnet(x)
        return self._forward_mlp(x)

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward: hook each Linear layer."""
        self._remove_hooks()
        for name, module in self.base_encoder.named_modules():
            if isinstance(module, nn.Linear):
                safe_name = name.replace(".", "_")
                if safe_name in self.mask_layers:
                    mask_layer = self.mask_layers[safe_name]
                    h = module.register_forward_hook(
                        self._make_mask_hook(mask_layer)
                    )
                    self._hooks.append(h)
        z = self.base_encoder(x)
        self._remove_hooks()
        return z

    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet forward: apply masks at conv1, each BasicBlock, and projection."""
        backbone = self.base_encoder.backbone

        # Initial conv + groupnorm + relu (mask after conv1)
        out = backbone.conv1(x)
        gate = self.mask_layers["conv1"](self._current_task, self._current_s)
        out = out * gate.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = backbone.bn1(out)  # Now GroupNorm
        out = backbone.relu(out)
        out = backbone.maxpool(out)  # Identity for small-input

        # Residual blocks — mask at block output level
        block_idx = 0
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(backbone, layer_name)
            for block in layer:
                safe_name, _ = self._block_targets[block_idx]
                out = block(out)
                gate = self.mask_layers[safe_name](self._current_task, self._current_s)
                out = out * gate.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                block_idx += 1

        # Global average pool + flatten + projection
        out = backbone.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.base_encoder.projection(out)
        gate = self.mask_layers["projection"](self._current_task, self._current_s)
        out = out * gate.unsqueeze(0)
        return out

    def _make_mask_hook(self, mask_layer: HATMaskLayer):
        def hook(module, input, output):
            gate = mask_layer(self._current_task, self._current_s)
            if output.dim() == 2:
                return output * gate.unsqueeze(0)
            elif output.dim() == 4:
                # Conv output: (B, C, H, W) — gate over channels
                return output * gate.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return output * gate
        return hook

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def parameters(self, recurse=True):
        """Yield all parameters (base encoder + mask embeddings)."""
        yield from self.base_encoder.parameters(recurse=recurse)
        yield from self.mask_layers.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        yield from self.base_encoder.named_parameters(
            prefix=prefix + "base_encoder." if prefix else "base_encoder.",
            recurse=recurse,
        )
        yield from self.mask_layers.named_parameters(
            prefix=prefix + "mask_layers." if prefix else "mask_layers.",
            recurse=recurse,
        )

    def train(self, mode=True):
        super().train(mode)
        self.base_encoder.train(mode)
        return self

    def eval(self):
        super().eval()
        self.base_encoder.eval()
        return self

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        yield from self.base_encoder.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )


class HAT(ContinualLearningMethod):
    """Hard Attention to the Task continual learning method.

    The encoder must be wrapped in HATEncoder before passing to this class.
    """

    def __init__(
        self,
        encoder: HATEncoder,
        decoder: nn.Module,
        device: torch.device,
        s_max: float = 400.0,
        mask_reg_coeff: float = 0.01,
    ):
        super().__init__(encoder, decoder, device)
        self.s_max = s_max
        self.mask_reg_coeff = mask_reg_coeff
        self._cumulative_mask: dict[str, torch.Tensor] = {}

    def supports_checkpointing(self) -> bool:
        return True

    def _extra_state_dict(self) -> dict:
        return {
            "cumulative_mask": {
                name: mask.detach().cpu() for name, mask in self._cumulative_mask.items()
            }
        }

    def _load_extra_state_dict(self, state: dict) -> None:
        cumulative_mask = state.get("cumulative_mask", {})
        self._cumulative_mask = {
            name: mask.to(self.device) for name, mask in cumulative_mask.items()
        }

    def prepare_task(self, task_id: int, train_loader: DataLoader) -> None:
        self.current_task = task_id
        self.decoder.add_task_head(task_id)

    def train_task(
        self, task_id: int, train_loader: DataLoader, epochs: int, lr: float
    ) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        # Separate params: base encoder + decoder head vs. mask embeddings
        base_params = list(self.encoder.base_encoder.parameters()) + list(
            self.decoder.heads[str(task_id)].parameters()
        )
        mask_params = list(self.encoder.mask_layers.parameters())
        optimizer = torch.optim.SGD(
            [{"params": base_params}, {"params": mask_params}],
            lr=lr, momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        total_batches = len(train_loader) * epochs
        batch_counter = 0

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                batch_counter += 1

                # Anneal s: from 1/s_max to s_max linearly over training
                progress = batch_counter / max(total_batches, 1)
                s = 1.0 / self.s_max + (self.s_max - 1.0 / self.s_max) * progress

                self.encoder.set_task(task_id, s)
                z = self.encoder(x)
                logits = self.decoder(z, task_id)

                # Task loss + mask regularization
                task_loss = criterion(logits, y)
                reg_loss = self._mask_regularization(task_id, s)
                loss = task_loss + self.mask_reg_coeff * reg_loss

                optimizer.zero_grad()
                loss.backward()

                # Compensate-clip: limit gradient for mask embeddings based
                # on cumulative mask (don't modify gates for old tasks)
                self._compensate_clip(task_id, s)

                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_samples += x.size(0)

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc": total_correct / max(total_samples, 1),
        }

    def after_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Binarize masks and accumulate into cumulative mask."""
        masks = self.encoder.get_masks(task_id, s=self.s_max)
        for name, gate in masks.items():
            binary = (gate > 0.5).float().detach()
            if name in self._cumulative_mask:
                self._cumulative_mask[name] = torch.max(
                    self._cumulative_mask[name], binary
                )
            else:
                self._cumulative_mask[name] = binary.clone()

        total_used = sum(m.sum().item() for m in self._cumulative_mask.values())
        total_units = sum(m.numel() for m in self._cumulative_mask.values())
        print(
            f"  HAT task {task_id}: "
            f"used={int(total_used)}/{total_units} "
            f"({100 * total_used / total_units:.1f}%)"
        )

    def evaluate(self, task_id: int, test_loader: DataLoader) -> float:
        """Evaluate with task-specific masks at s=s_max (hard attention)."""
        self.encoder.eval()
        self.decoder.eval()
        self.encoder.set_task(task_id, self.s_max)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                logits = self.decoder(z, task_id)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def evaluate_class_incremental(
        self, test_loader: DataLoader, classes_per_task: int
    ) -> float:
        """CI evaluation for HAT: run encoder with each task's mask, collect
        that task head's logits, and take the global argmax."""
        self.encoder.eval()
        self.decoder.eval()
        num_heads = self.decoder.num_tasks
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_logits = []
                for t in range(num_heads):
                    self.encoder.set_task(t, self.s_max)
                    z = self.encoder(x)
                    logits = self.decoder(z, t)  # (N, classes_per_task)
                    all_logits.append(logits)
                cat_logits = torch.cat(all_logits, dim=1)
                preds = cat_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_regularization(self, task_id: int, s: float) -> torch.Tensor:
        """Encourage new masks to be sparse AND avoid overlap with old masks.

        reg = sum_l ( mask_current * cumulative_mask_old ).sum()
              + sum_l mask_current.sum() / num_units  [sparsity]
        """
        reg = torch.tensor(0.0, device=self.device)
        masks = self.encoder.get_masks(task_id, s)
        for name, gate in masks.items():
            # Sparsity: encourage gates to be near 0
            reg = reg + gate.sum() / gate.numel()
            # Overlap penalty: discourage using capacity already used
            if name in self._cumulative_mask:
                reg = reg + (gate * self._cumulative_mask[name]).sum() / gate.numel()
        return reg

    def _compensate_clip(self, task_id: int, s: float) -> None:
        """Clip gradients for mask embeddings AND base encoder weights to
        protect channels claimed by cumulative mask from old tasks."""
        # 1. Clip mask embedding gradients
        for name in self.encoder._layer_order:
            mask_layer = self.encoder.mask_layers[name]
            if mask_layer.embeddings.grad is None:
                continue
            if name in self._cumulative_mask:
                # Zero out gradient for units where cumulative mask is 1
                cum = self._cumulative_mask[name]
                mask_layer.embeddings.grad.data[task_id] *= (1.0 - cum)

        # 2. Compensate base encoder weight gradients: zero out gradients
        # for output channels already claimed by previous tasks.
        if not self._cumulative_mask:
            return

        if self.encoder._is_resnet:
            self._compensate_resnet_weights()
        else:
            self._compensate_mlp_weights()

    def _compensate_resnet_weights(self) -> None:
        """Zero out ResNet weight gradients for channels protected by old tasks."""
        backbone = self.encoder.base_encoder.backbone

        # conv1 → output channels protected by cumul_mask["conv1"]
        if "conv1" in self._cumulative_mask and backbone.conv1.weight.grad is not None:
            cum = self._cumulative_mask["conv1"]  # (64,)
            backbone.conv1.weight.grad.data *= (1.0 - cum).view(-1, 1, 1, 1)
            if hasattr(backbone, "bn1") and hasattr(backbone.bn1, "weight"):
                if backbone.bn1.weight is not None and backbone.bn1.weight.grad is not None:
                    backbone.bn1.weight.grad.data *= (1.0 - cum)
                if backbone.bn1.bias is not None and backbone.bn1.bias.grad is not None:
                    backbone.bn1.bias.grad.data *= (1.0 - cum)

        # Each residual block → protect the full residual branch for channels
        # already claimed by previous tasks.
        for safe_name, block in self.encoder._block_targets:
            if safe_name not in self._cumulative_mask:
                continue
            cum = self._cumulative_mask[safe_name]  # (out_channels,)

            if block.conv1.weight.grad is not None:
                block.conv1.weight.grad.data *= (1.0 - cum).view(-1, 1, 1, 1)
            if hasattr(block, 'bn1') and hasattr(block.bn1, 'weight') and block.bn1.weight is not None:
                if block.bn1.weight.grad is not None:
                    block.bn1.weight.grad.data *= (1.0 - cum)
                if block.bn1.bias is not None and block.bn1.bias.grad is not None:
                    block.bn1.bias.grad.data *= (1.0 - cum)

            # conv2 is the last conv in the block — its output channels are masked.
            if block.conv2.weight.grad is not None:
                block.conv2.weight.grad.data *= (1.0 - cum).view(-1, 1, 1, 1)
            if hasattr(block, 'bn2') and hasattr(block.bn2, 'weight') and block.bn2.weight is not None:
                if block.bn2.weight.grad is not None:
                    block.bn2.weight.grad.data *= (1.0 - cum)
                if block.bn2.bias is not None and block.bn2.bias.grad is not None:
                    block.bn2.bias.grad.data *= (1.0 - cum)

            # If there's a downsample, protect the output channels there too.
            if hasattr(block, 'downsample') and block.downsample is not None:
                for mod in block.downsample.modules():
                    if isinstance(mod, nn.Conv2d) and mod.weight.grad is not None:
                        if mod.weight.shape[0] == cum.shape[0]:
                            mod.weight.grad.data *= (1.0 - cum).view(-1, 1, 1, 1)
                    elif isinstance(mod, (nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                        if getattr(mod, 'weight', None) is not None and mod.weight.grad is not None:
                            mod.weight.grad.data *= (1.0 - cum)
                        if getattr(mod, 'bias', None) is not None and mod.bias.grad is not None:
                            mod.bias.grad.data *= (1.0 - cum)

        # Projection layer
        proj = self.encoder.base_encoder.projection
        if "projection" in self._cumulative_mask and proj.weight.grad is not None:
            cum = self._cumulative_mask["projection"]  # (latent_dim,)
            proj.weight.grad.data *= (1.0 - cum).view(-1, 1)
            if proj.bias is not None and proj.bias.grad is not None:
                proj.bias.grad.data *= (1.0 - cum)

    def _compensate_mlp_weights(self) -> None:
        """Zero out MLP weight gradients for units protected by old tasks."""
        for name, module in self.encoder.base_encoder.named_modules():
            if isinstance(module, nn.Linear):
                safe_name = name.replace(".", "_")
                if safe_name in self._cumulative_mask and module.weight.grad is not None:
                    cum = self._cumulative_mask[safe_name]  # (out_features,)
                    module.weight.grad.data *= (1.0 - cum).view(-1, 1)
                    if module.bias is not None and module.bias.grad is not None:
                        module.bias.grad.data *= (1.0 - cum)
