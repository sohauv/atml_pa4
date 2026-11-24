import json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from task1.quantization import fold_mlp_up_down

revision = "8faed761d45a263340a0528343f099c05c9a4323"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", revision=revision,
                                            dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
torch.manual_seed(6304)
torch.backends.cuda.matmul.allow_tf32 = False
checks = []
with torch.inference_mode():
    for i, block in enumerate(model.model.layers):
        gate, up, down = (getattr(block.mlp, name + "_proj") for name in ("gate", "up", "down"))
        x = torch.randn(64, up.in_features, device="cuda")
        g = F.silu(F.linear(x, gate.weight.float()))
        intermediate = g * F.linear(x, up.weight.float())
        scale = intermediate.abs().mean(0).clamp_min(1e-6).sqrt()
        scale = (scale / (scale.max() * scale.min()).sqrt()).half()
        folded_up, scaled_down, _ = fold_mlp_up_down(up.weight, down.weight, scale)
        reference = F.linear(intermediate, down.weight.float())
        actual = F.linear(g * F.linear(x, folded_up), scaled_down)
        fp16_reference = down(F.silu(gate(x.half())) * up(x.half())).float()
        fp16_folded = F.linear(F.silu(gate(x.half())) * F.linear(x.half(), folded_up.half()),
                               scaled_down.half()).float()
        def error(a, b):
            return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
        e32, e16 = error(reference, actual), error(fp16_reference, fp16_folded)
        if not (e32 < 1e-4 and e16 < .01):
            raise AssertionError((i, e32, e16))
        checks.append(dict(layer=i, fp32_relative_l2=e32, fp16_relative_l2=e16))
out = Path("runs/pa4-awq-fold-check")
out.mkdir(parents=True, exist_ok=True)
(out / "fold-check.json").write_text(json.dumps(dict(revision=revision, checks=checks), indent=2))
