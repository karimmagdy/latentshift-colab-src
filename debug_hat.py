from src.models.encoder import MLPEncoder
from src.methods.baselines.hat import HATEncoder
import torch.nn as nn

mlp = MLPEncoder(784, 400, 256)
hat = HATEncoder(mlp, num_tasks=10)

print("Mask layer names:", list(hat.mask_layers.keys()))
print("Layer order:", hat._layer_order)

for name, mod in hat.base_encoder.named_modules():
    if isinstance(mod, nn.Linear):
        safe = name.replace(".", "_")
        print(f"  Linear: name={name} safe_name={safe} shape={mod.weight.shape}")
