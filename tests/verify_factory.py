"""Quick verification that new method configs work."""
import sys
sys.path.insert(0, ".")
from experiments.run_experiment import build_method, build_encoder, build_benchmark
from src.models.decoder import MultiHeadDecoder
import torch, yaml

device = torch.device("cpu")

cfg = yaml.safe_load(open("configs/latent_shift_tuned_split_cifar10.yaml"))
benchmark = build_benchmark(cfg)
encoder = build_encoder(cfg).to(device)
decoder = MultiHeadDecoder(cfg.get("latent_dim", 256), benchmark.classes_per_task).to(device)
method = build_method(cfg, encoder, decoder, device)
print(f"[OK] latent_shift_tuned: {type(method).__name__}, latent_dim={cfg['latent_dim']}")

cfg2 = yaml.safe_load(open("configs/gpm_lastlayer_split_cifar10.yaml"))
encoder2 = build_encoder(cfg2).to(device)
decoder2 = MultiHeadDecoder(cfg2.get("latent_dim", 256), benchmark.classes_per_task).to(device)
method2 = build_method(cfg2, encoder2, decoder2, device)
print(f"[OK] gpm_lastlayer: {type(method2).__name__}, last_layer_only={method2.last_layer_only}")
print("All factory checks passed!")
