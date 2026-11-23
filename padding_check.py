import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from speculative import speculative

def load(name, revision):
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-" + name + "-Instruct", revision=revision,
        dtype=torch.float16, attn_implementation="sdpa"
    ).cuda().eval()

target = load("1.5B", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
draft = load("0.5B", "7ae557604adf67be50417f59c2c2f167def9a775")
prompt = torch.tensor([[970, 1000, 3000, 5000]], device="cuda")
checks = []
for gamma in (1, 2, 4):
    expected = speculative(target, draft, prompt, gamma, 12, {151645}, 6304)
    for side in ("left", "right"):
        pads = torch.full((1, 3), 151645, device="cuda")
        mask = torch.ones_like(prompt)
        pieces = (pads, prompt) if side == "left" else (prompt, pads)
        masks = (torch.zeros_like(pads), mask) if side == "left" else (mask, torch.zeros_like(pads))
        actual = speculative(target, draft, torch.cat(pieces, 1), gamma, 12, {151645}, 6304,
                             attention_mask=torch.cat(masks, 1))
        assert actual == expected, (gamma, side)
        checks.append(dict(gamma=gamma, side=side, matched=True))
out = Path("runs/pa4-sampler-padding-results")
out.mkdir(parents=True, exist_ok=True)
(out / "padding.json").write_text(json.dumps(checks, indent=2))
