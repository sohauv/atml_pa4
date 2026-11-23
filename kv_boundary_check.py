
import json
from pathlib import Path
import torch
from kv_cache import quantize_appended_groups

rng = torch.Generator(device="cuda").manual_seed(6304)
key, value = [torch.randn(1, 2, 194, 128, device="cuda", dtype=torch.float16,
                          generator=rng) for _ in range(2)]
start = 67
base = ((key[..., :start, :], value[..., :start, :]),)
checks = []
for bits, channel in ((4, False), (4, True), (2, True)):
    for appended in (63, 64, 65, 127):
        stop = start + appended
        cache = ((key[..., :stop, :], value[..., :stop, :]),)
        updated = quantize_appended_groups(cache, base, bits, channel)
        complete = appended // 64 * 64
        changed = []
        for old, new in zip(cache[0], updated[0], strict=True):
            assert old.shape == new.shape and torch.isfinite(new).all()
            assert torch.equal(old[..., :start, :], new[..., :start, :])
            assert torch.equal(old[..., start+complete:, :], new[..., start+complete:, :])
            changed.append((old != new).count_nonzero().item())
        assert all((count > 0) == (complete > 0) for count in changed)
        checks.append(dict(bits=bits, channelwise_key=channel, appended=appended,
                           quantized_tokens=complete, changed_elements=changed))
out = Path("runs/pa4-kv-boundary-results")
out.mkdir(parents=True, exist_ok=True)
(out / "boundary.json").write_text(json.dumps(checks, indent=2))
