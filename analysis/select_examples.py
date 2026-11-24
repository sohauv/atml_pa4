"""Select fixed matched examples from the PA4 evidence archives."""
import json
from pathlib import Path
from zipfile import ZipFile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def archived(path):
    with ZipFile(path) as archive:
        return {name: json.loads(archive.read(name)) for name in sorted(archive.namelist())
                if name.endswith(".json")}


def main():
    baseline = read("results/task1/baseline-20261121T134102Z.json")["hellaswag"]["examples"]
    wanda = read("results/task1/wanda_50.json")["hellaswag"]["examples"]
    chosen = []
    for ref, test in zip(baseline, wanda, strict=True):
        assert ref["id"] == test["id"]
        if ref["correct_norm"] and not test["correct_norm"]:
            chosen.append(dict(id=ref["id"], fp16=ref, wanda50=test))
        if len(chosen) == 3:
            break

    kv = archived("results/task2/pa4-kv-residual-results.zip")
    retrieval = {}
    for policy in ("sink_recent", "int4_token", "hybrid"):
        failures = [row for name, row in kv.items() if Path(name).name.startswith("KV-")
                    and row["policies"]["full"]["exact_match"]
                    and not row["policies"][policy]["exact_match"]]
        retrieval[policy] = failures[:2]

    evaluation = archived("results/task3/pa4-speculative-results.zip")
    official = archived("results/task3/pa4-course-bank-results.zip")
    course = {(name.split("/")[-2], row["id"]): row["prefix_metrics"]
              for name, row in official.items() if "/course_alignment/" in name}
    alignment, continuations = [], []
    for name, reference in evaluation.items():
        if "/per_prompt/original/" not in name:
            continue
        candidate = evaluation[name.replace("/original/", "/off_policy-6304/")]
        if not any(x["domain"] == reference["domain"] for x in continuations):
            continuations.append(dict(domain=reference["domain"], id=reference["id"],
                                      original=reference["standalone"], off_policy=candidate["standalone"]))
        left, right = (course[draft, reference["id"]]["teacher"]
                       for draft in ("original", "off_policy-6304"))
        for a, b in zip(left, right, strict=True):
            assert a["position"] == b["position"]
            if a["greedy_agreement"] != b["greedy_agreement"]:
                alignment.append(dict(id=reference["id"], original=a, off_policy=b))
    result = dict(
        selection="First matching examples in saved manifest order for HellaSwag and sorted archive order for KV and drafts.",
        hellaswag=chosen, kv=retrieval, speculative_continuations=continuations,
        teacher_prefix_disagreements=alignment[:6],
    )
    assert len(chosen) == 3 and len(continuations) == 3 and len(alignment) >= 6
    assert all(retrieval.values())
    Path("results/shared/matched-examples.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
