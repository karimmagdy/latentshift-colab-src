from __future__ import annotations

"""Main experiment runner for LatentShift continual learning experiments."""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.models.encoder import MLPEncoder, ResNetEncoder, ViTEncoder
from src.models.decoder import MultiHeadDecoder
from src.data.benchmarks import SplitMNIST, PermutedMNIST, SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet
from src.methods.latent_shift import LatentShiftMethod
from src.methods.baselines.naive import NaiveFineTuning
from src.methods.baselines.ewc import EWC
from src.methods.baselines.gpm import GPM
from src.methods.baselines.packnet import PackNet
from src.methods.baselines.hat import HAT, HATEncoder
from src.methods.baselines.er import ExperienceReplay
from src.methods.baselines.der import DERPlusPlus
from src.methods.baselines.trgp import TRGP
from src.methods.baselines.l2p import L2P
from src.methods.baselines.dualprompt import DualPrompt
from src.methods.baselines.coda_prompt import CODAPrompt
from src.training.trainer import run_continual_learning
from src.utils.metrics import ContinualMetrics


# ---------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------

BENCHMARKS = {
    "split_mnist": SplitMNIST,
    "permuted_mnist": PermutedMNIST,
    "split_cifar10": SplitCIFAR10,
    "split_cifar100": SplitCIFAR100,
    "split_tinyimagenet": SplitTinyImageNet,
    "seq_cifar100": lambda **kw: SplitCIFAR100(classes_per_task=5, **kw),
}


def build_benchmark(cfg: dict):
    name = cfg["benchmark"]
    kwargs = {k: v for k, v in cfg.get("benchmark_args", {}).items()}
    return BENCHMARKS[name](**kwargs)


def build_encoder(cfg: dict) -> torch.nn.Module:
    enc_type = cfg.get("encoder", "mlp")
    latent_dim = cfg.get("latent_dim", 256)
    num_tasks = cfg.get("num_tasks", 50)

    if enc_type == "mlp":
        return MLPEncoder(
            input_dim=cfg.get("input_dim", 784),
            hidden_dim=cfg.get("hidden_dim", 400),
            latent_dim=latent_dim,
        )
    elif enc_type == "resnet18":
        small = cfg.get("small_input", True)
        return ResNetEncoder(latent_dim=latent_dim, pretrained=cfg.get("pretrained", False), small_input=small)
    elif enc_type == "hat_mlp":
        base = MLPEncoder(
            input_dim=cfg.get("input_dim", 784),
            hidden_dim=cfg.get("hidden_dim", 400),
            latent_dim=latent_dim,
        )
        return HATEncoder(base, num_tasks=num_tasks)
    elif enc_type == "hat_resnet18":
        base = ResNetEncoder(latent_dim=latent_dim, pretrained=cfg.get("pretrained", False))
        return HATEncoder(base, num_tasks=num_tasks)
    elif enc_type == "vit_tiny":
        return ViTEncoder(
            latent_dim=latent_dim,
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
            in_channels=cfg.get("in_channels", 3),
            embed_dim=192,
            depth=6,
            num_heads=3,
        )
    else:
        raise ValueError(f"Unknown encoder: {enc_type}")


def build_method(cfg: dict, encoder, decoder, device):
    method_name = cfg["method"]
    latent_dim = cfg.get("latent_dim", 256)

    if method_name in ("latent_shift", "latent_shift_tuned"):
        return LatentShiftMethod(
            encoder, decoder, device,
            latent_dim=latent_dim,
            threshold=cfg.get("threshold", 0.99),
            num_samples=cfg.get("num_samples", 300),
            lossy=cfg.get("lossy", False),
            max_archive_rank=cfg.get("max_archive_rank", None),
        )
    elif method_name == "naive":
        return NaiveFineTuning(encoder, decoder, device)
    elif method_name == "ewc":
        return EWC(
            encoder, decoder, device,
            ewc_lambda=cfg.get("ewc_lambda", 400.0),
            num_fisher_samples=cfg.get("num_fisher_samples", 200),
        )
    elif method_name == "gpm":
        return GPM(
            encoder, decoder, device,
            threshold=cfg.get("threshold", 0.99),
            num_samples=cfg.get("num_samples", 300),
        )
    elif method_name == "gpm_lastlayer":
        return GPM(
            encoder, decoder, device,
            threshold=cfg.get("threshold", 0.99),
            num_samples=cfg.get("num_samples", 300),
            last_layer_only=True,
        )
    elif method_name == "trgp":
        return TRGP(
            encoder, decoder, device,
            threshold=cfg.get("threshold", 0.99),
            num_samples=cfg.get("num_samples", 300),
            trust_alpha=cfg.get("trust_alpha", 0.5),
        )
    elif method_name == "packnet":
        return PackNet(
            encoder, decoder, device,
            prune_ratio=cfg.get("prune_ratio", 0.75),
            retrain_epochs=cfg.get("retrain_epochs", 0),
        )
    elif method_name == "hat":
        return HAT(
            encoder, decoder, device,
            s_max=cfg.get("s_max", 400.0),
            mask_reg_coeff=cfg.get("mask_reg_coeff", 0.01),
        )
    elif method_name == "er":
        return ExperienceReplay(
            encoder, decoder, device,
            buffer_size_per_task=cfg.get("buffer_size_per_task", 200),
            replay_batch_size=cfg.get("replay_batch_size", 64),
        )
    elif method_name == "der":
        return DERPlusPlus(
            encoder, decoder, device,
            buffer_size_per_task=cfg.get("buffer_size_per_task", 200),
            replay_batch_size=cfg.get("replay_batch_size", 64),
            alpha=cfg.get("alpha", 0.5),
            beta=cfg.get("beta", 0.5),
        )
    elif method_name == "l2p":
        return L2P(
            encoder, decoder, device,
            pool_size=cfg.get("pool_size", 10),
            prompt_length=cfg.get("prompt_length", 5),
            top_k=cfg.get("top_k", 5),
            freeze_encoder=cfg.get("freeze_encoder", True),
            pull_weight=cfg.get("pull_weight", 0.5),
        )
    elif method_name == "dualprompt":
        return DualPrompt(
            encoder, decoder, device,
            e_pool_size=cfg.get("e_pool_size", 10),
            e_prompt_length=cfg.get("e_prompt_length", 5),
            g_prompt_length=cfg.get("g_prompt_length", 5),
            top_k=cfg.get("top_k", 5),
            freeze_encoder=cfg.get("freeze_encoder", True),
            pull_weight=cfg.get("pull_weight", 0.5),
            g_layers=cfg.get("g_layers", 2),
        )
    elif method_name == "coda_prompt":
        return CODAPrompt(
            encoder, decoder, device,
            pool_size=cfg.get("pool_size", 10),
            prompt_length=cfg.get("prompt_length", 5),
            freeze_encoder=cfg.get("freeze_encoder", True),
            ortho_weight=cfg.get("ortho_weight", 0.1),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


def checkpoint_stem(cfg: dict, seed: int, class_incremental: bool) -> str:
    suffix = "_ci" if class_incremental else ""
    default_encoders = {"resnet18", "hat_resnet18", "mlp"}
    encoder_name = cfg.get("encoder", "")
    encoder_tag = f"_{encoder_name}" if encoder_name and encoder_name not in default_encoders else ""
    return f"{cfg['method']}{encoder_tag}_{cfg['benchmark']}_seed{seed}{suffix}"


def find_latest_checkpoint(checkpoint_dir: Path, stem: str) -> Path | None:
    candidates = sorted(checkpoint_dir.glob(f"{stem}_task*.pt"))
    return candidates[-1] if candidates else None


def save_checkpoint(
    checkpoint_dir: Path,
    stem: str,
    task_id: int,
    cfg: dict,
    seed: int,
    method,
    metrics: ContinualMetrics,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{stem}_task{task_id:03d}.pt"
    torch.save(
        {
            "config": cfg,
            "seed": seed,
            "task_id": task_id,
            "next_task": task_id + 1,
            "method_state": method.state_dict(),
            "metrics": metrics.to_dict(),
        },
        path,
    )
    return path


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LatentShift CL Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for per-task checkpoints (HAT only)")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the latest checkpoint for this run if available")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a specific checkpoint file")
    parser.add_argument("--class-incremental", action="store_true",
                        help="Also run class-incremental evaluation after standard CL")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)

    # Build components
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    stem = checkpoint_stem(cfg, args.seed, args.class_incremental)

    benchmark = build_benchmark(cfg)
    encoder = build_encoder(cfg).to(device)
    decoder = MultiHeadDecoder(
        latent_dim=cfg.get("latent_dim", 256),
        classes_per_task=benchmark.classes_per_task,
    ).to(device)
    method = build_method(cfg, encoder, decoder, device)

    start_task = 0
    metrics = None
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume_latest:
        resume_path = find_latest_checkpoint(checkpoint_dir, stem)

    if resume_path is not None:
        if not method.supports_checkpointing():
            raise ValueError(f"Method '{cfg['method']}' does not support resume checkpoints")
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        method.load_state_dict(checkpoint["method_state"])
        metrics = ContinualMetrics.from_dict(checkpoint.get("metrics", {}))
        start_task = checkpoint.get("next_task", checkpoint.get("task_id", -1) + 1)
        print(f"Resumed from {resume_path} (next task: {start_task})")

    print(f"Benchmark: {cfg['benchmark']} ({benchmark.num_tasks} tasks)")
    print(f"Method: {cfg['method']}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Run
    t0 = time.time()
    metrics = run_continual_learning(
        method, benchmark,
        epochs_per_task=cfg.get("epochs", 10),
        lr=cfg.get("lr", 0.01),
        start_task=start_task,
        metrics=metrics,
        checkpoint_callback=(
            (lambda task_id, current_metrics: save_checkpoint(
                checkpoint_dir, stem, task_id, cfg, args.seed, method, current_metrics
            ))
            if method.supports_checkpointing() else None
        ),
    )
    elapsed = time.time() - t0

    # Save results
    results = {
        "config": cfg,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "metrics": metrics.summary(),
        "accuracy_matrix": metrics.get_accuracy_matrix().tolist(),
        "per_task_extras": {str(k): v for k, v in metrics.extras.items()},
    }
    # Class-incremental evaluation (optional)
    if args.class_incremental:
        from src.data.benchmarks import ClassIncrementalWrapper
        ci_wrapper = ClassIncrementalWrapper(benchmark)
        cumul_loader = ci_wrapper.get_cumulative_test_loader(benchmark.num_tasks - 1)
        ci_acc = method.evaluate_class_incremental(cumul_loader, benchmark.classes_per_task)
        results["class_incremental_accuracy"] = ci_acc
        print(f"Class-Incremental Accuracy: {ci_acc:.4f}")

    suffix = "_ci" if args.class_incremental else ""
    # Embed encoder name for non-default architectures to avoid filename collisions
    default_encoders = {"resnet18", "hat_resnet18", "mlp"}
    encoder_file_tag = {"vit_tiny": "vit"}  # keep backward-compat filenames
    encoder_name = cfg.get("encoder", "")
    if encoder_name in encoder_file_tag:
        encoder_tag = f"_{encoder_file_tag[encoder_name]}"
    elif encoder_name and encoder_name not in default_encoders:
        encoder_tag = f"_{encoder_name}"
    else:
        encoder_tag = ""
    out_path = output_dir / f"{cfg['method']}{encoder_tag}_{cfg['benchmark']}_seed{args.seed}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
