"""Audit folding opportunities in the saved PA4 AWQ scale vectors."""
import json
from pathlib import Path
from zipfile import ZipFile


def inspect(record):
    projections = record["projections"]
    cases = []
    for index in sorted({int(name.split(".")[0]) for name in projections}):
        def scale(name):
            return projections[f"{index}.{name}_proj"]["scale_fp16"]

        q, k, v = (scale(name) for name in ("q", "k", "v"))
        gate, up, down, out = (scale(name) for name in ("gate", "up", "down", "o"))
        if record['revision'] != '8faed761d45a263340a0528343f099c05c9a4323':
            raise ValueError('This audit uses the pinned Qwen2.5-1.5B head configuration')
        groups = 12 // 2
        # Qwen GQA repeats each KV head contiguously across its query-head group.
        head_dim = 128
        blocks = [out[i:i + head_dim] for i in range(0, len(out), head_dim)]
        repeats_match = all(blocks[i] == blocks[i // groups * groups] for i in range(len(blocks)))
        cases.append(dict(
            layer=index,
            shared_attention_norm_foldable=q == k == v,
            shared_mlp_norm_foldable=gate == up,
            output_projection_foldable_into_value=repeats_match,
            down_projection_foldable_into_up=True,
            down_scale_channels=len(down),
        ))
    flags = ("shared_attention_norm_foldable", "shared_mlp_norm_foldable",
             "output_projection_foldable_into_value", "down_projection_foldable_into_up")
    return dict(layers=len(cases), counts={flag: sum(row[flag] for row in cases) for flag in flags},
                by_layer=cases)


def main():
    result = {}
    for filename in ("pa4-q2-results.zip", "pa4-q3-results.zip"):
        with ZipFile(filename) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".json"):
                    continue
                data = json.loads(archive.read(name))
                if "condition" in data:
                    result[data["condition"]] = inspect(data)
    assert len(result) == 5 and all(row["layers"] == 28 for row in result.values())
    Path("awq-folding-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, row in result.items():
        print(name, row["counts"])


if __name__ == "__main__":
    main()
