"""PA4 Task 2C retrieval, packed KV cache, and decode benchmarks."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from task1.baseline import read_jsonl
from task1.pruning import write_json


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICIES = ("full", "sink_recent", "question", "int4_token", "int4_channel", "hybrid")


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    start = perf_counter()
    value = fn()
    sync()
    return value, (perf_counter() - start) * 1000


def encode_groups(x, bits):
    groups = x.float().reshape(-1, 64)
    offset = groups.amin(-1, keepdim=True).half()
    vmax = (1 << bits) - 1
    scale = ((groups.amax(-1, keepdim=True) - offset.float()) / vmax)
    scale = scale.clamp_min(2 ** -24).half()
    codes = ((groups - offset.float()) / scale.float()).round().clamp_(0, vmax).to(torch.uint8)
    stride = 8 // bits
    chunks = codes.reshape(-1, 64 // stride, stride)
    packed = chunks[..., 0].clone()
    for i in range(1, stride):
        packed.bitwise_or_(chunks[..., i] << (i * bits))
    return packed, scale, offset, x.shape


def nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def decode_groups(encoded, bits):
    packed, scale, offset, shape = encoded
    codes = torch.stack([(packed >> i) & ((1 << bits) - 1)
                         for i in range(0, 8, bits)], dim=-1)
    return (codes.flatten(-2).float() * scale.float() + offset.float()).half().reshape(shape)


def pack_layer(key, value, bits, channel, high=None):
    if high is None:
        complete = key.size(-2) // 64 * 64
        low_key, low_value = key[..., :complete, :], value[..., :complete, :]
        key16, value16 = key[..., complete:, :].clone(), value[..., complete:, :].clone()
        indices = None
    else:
        high = high.to(torch.int32)
        mask = torch.ones(key.size(-2), device=key.device, dtype=torch.bool)
        mask[high.long()] = False
        low = mask.nonzero().flatten()
        complete = low.numel() // 64 * 64
        indices = torch.cat((high, low[complete:].to(torch.int32)))
        low = low[:complete]
        low_key, low_value = key.index_select(-2, low), value.index_select(-2, low)
        key16 = key.index_select(-2, indices.long())
        value16 = value.index_select(-2, indices.long())
    kdata = encode_groups(low_key.transpose(-1, -2) if channel else low_key, bits)
    vdata = encode_groups(low_value, bits)
    return kdata, vdata, key16, value16, indices


def unpack_layer(packed, bits, channel):
    key, value, key16, value16, indices = packed
    k, v = decode_groups(key, bits), decode_groups(value, bits)
    if channel:
        k = k.transpose(-1, -2)
    if indices is None:
        return torch.cat((k, key16), -2), torch.cat((v, value16), -2)
    length = k.size(-2) + key16.size(-2)
    index = indices.long()
    mask = torch.ones(length, dtype=torch.bool, device=k.device)
    mask[index] = False
    low = mask.nonzero().flatten()
    shape = (*k.shape[:-2], length, k.size(-1))
    out_k, out_v = k.new_empty(shape), v.new_empty(shape)
    out_k.index_copy_(-2, low, k)
    out_k.index_copy_(-2, index, key16)
    out_v.index_copy_(-2, low, v)
    out_v.index_copy_(-2, index, value16)
    return out_k, out_v


def packed_size(layers):
    return sum(sum(nbytes(t) for t in (*k[:3], *v[:3], ku, vu)) +
               (nbytes(idx) if idx is not None else 0) for k, v, ku, vu, idx in layers)


def cache_bytes(layers):
    return sum(nbytes(key) + nbytes(value) for key, value in layers)


def budget_bytes(channels, length, bits):
    groups, residual = divmod(length, 64)
    return channels * (groups * (8 * bits + 4) + 2 * residual)


def choose_positions(length, count, device, importance=None):
    mandatory = torch.cat((torch.arange(8, device=device),
                           torch.arange(length - 64, length, device=device)))
    if count < mandatory.numel():
        raise ValueError("Budget cannot cover mandatory positions")
    if importance is None:
        additional = torch.arange(length - count + 8, length - 64, device=device)
    else:
        scores = importance.clone()
        scores[mandatory] = -torch.inf
        additional = scores.argsort(descending=True, stable=True)[:count - mandatory.numel()]
    return torch.cat((mandatory, additional)).sort().values


def choose_hybrid(length, channels, layer_count, budget):
    for count in range(length, 71, -1):
        groups, residual = divmod(length - count, 64)
        size = channels * (groups * 20 + 2 * (count + residual)) + 4 * layer_count * (count + residual)
        if size <= budget:
            return count
    raise ValueError("Budget cannot cover a hybrid cache")


def legacy(output):
    return output.past_key_values.to_legacy_cache()


def forward(model, tokens, start, cache=None, last=0):
    physical = cache[0][0].size(-2) if cache is not None else 0
    cache_position = torch.arange(physical, physical + tokens.size(1), device=tokens.device)
    absolute = torch.arange(start, start + tokens.size(1), device=tokens.device)[None]
    return model(tokens, past_key_values=DynamicCache.from_legacy_cache(cache) if cache else None,
                 position_ids=absolute, cache_position=cache_position, use_cache=True,
                 logits_to_keep=last)


def question_score(model, tokens, start, cache):
    captured, handles = {}, []
    for i, layer in enumerate(model.model.layers):
        def hook(_, __, output, index=i):
            captured[index] = output
        handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
    try:
        suffix = forward(model, tokens, start, cache, last=1)
    finally:
        for handle in handles:
            handle.remove()
    positions = torch.arange(start, start + tokens.size(1), device=tokens.device)[None]
    heads = model.config.num_attention_heads
    score = torch.zeros(start, device=tokens.device)
    for i, (key, _) in enumerate(legacy(suffix)):
        q = captured[i].view(1, tokens.size(1), heads, key.size(-1)).transpose(1, 2)
        cos, sin = model.model.rotary_emb(q, positions)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        key = key.repeat_interleave(heads // key.size(1), dim=1)
        logits = (q.float() @ key.float().transpose(-1, -2)) / math.sqrt(key.size(-1))
        mask = torch.arange(key.size(-2), device=key.device) <= positions.T
        score += logits.masked_fill(~mask, -torch.inf).softmax(-1)[..., :start].mean((0, 1, 2))
    return score / len(cache), suffix


def tokenize_case(tokenizer, row, device):
    text = tokenizer.apply_chat_template([{"role": "user", "content": row["user_text"]}],
                                         tokenize=False, add_generation_prompt=True)
    source = text.index(row["user_text"])
    question = source + row["user_text"].rindex("QUESTION_BEGIN")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    boundary = next(i for i, (_, hi) in enumerate(offsets) if hi > question)
    lo, hi = (source + row[key] for key in ("evidence_char_start", "evidence_char_end"))
    evidence = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    if not evidence or max(evidence) >= boundary:
        raise ValueError(f"Invalid evidence offsets for {row['id']}")
    return torch.tensor([encoded["input_ids"]], device=device), boundary, evidence


def quantize_appended_groups(cache, base, bits, channel):
    # Keep the question and short generated suffix in FP16. If its
    # tail reaches a group boundary, quantize only the new group.
    previous = base[0][0].size(-2)
    extra = cache[0][0].size(-2) - previous
    complete = extra // 64 * 64
    if not complete or not bits:
        return cache
    return tuple((torch.cat((bk, nk, k[..., previous+complete:, :]), -2),
                  torch.cat((bv, nv, v[..., previous+complete:, :]), -2))
                 for (bk,bv),(k,v) in zip(base,cache)
                 for nk,nv in [unpack_layer(pack_layer(k[..., previous:previous+complete, :],
                     v[..., previous:previous+complete, :], bits, channel), bits, channel)])


def residual_group_updater(base, bits, channel, high):
    """Consume complete groups from the context residual and appended tokens."""
    length = base[0][0].size(-2)
    device = base[0][0].device
    if high is None:
        low = torch.arange(length, device=device)
    else:
        mask = torch.ones(length, dtype=torch.bool, device=device)
        mask[high] = False
        low = mask.nonzero().flatten()
    pending = low[low.numel() // 64 * 64:]
    observed = length

    def update(cache):
        nonlocal pending, observed
        end = cache[0][0].size(-2)
        pending = torch.cat((pending, torch.arange(observed, end, device=device)))
        observed = end
        complete = pending.numel() // 64 * 64
        if not complete:
            return cache
        indices, pending = pending[:complete], pending[complete:]
        result = []
        for key, value in cache:
            packed = pack_layer(key.index_select(-2, indices), value.index_select(-2, indices),
                                bits, channel)
            k, v = unpack_layer(packed, bits, channel)
            result.append((key.index_copy(-2, indices, k), value.index_copy(-2, indices, v)))
        return tuple(result)

    return update


def decode(model, tokenizer, initial, absolute, eos, packed_update=None):
    out = initial
    generated = []
    for _ in range(16):
        token = out.logits[:, -1:].argmax(-1)
        token_id = token.item()
        if token_id in eos:
            break
        generated.append(token_id)
        cache = legacy(out)
        if packed_update is not None:
            cache = packed_update(cache)
        out = forward(model, token, absolute, cache, last=1)
        absolute += 1
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def compressed_cache(policy, cache, importance, budget, channels):
    length = cache[0][0].size(-2)
    device = importance.device
    high = None
    if policy == "full":
        return cache, cache_bytes(cache), high, None, None
    if policy in ("sink_recent", "question"):
        count = min(length, budget // (2 * channels + 4))
        high = choose_positions(length, count, device, importance if policy == "question" else None)
        result = tuple((k.index_select(-2, high), v.index_select(-2, high)) for k, v in cache)
        return result, cache_bytes(result) + 4 * high.numel(), high, None, None
    bits = 2 if policy == "hybrid" else 4
    channel = policy != "int4_token"
    if policy == "hybrid":
        count = choose_hybrid(length, channels, len(cache), budget)
        high = choose_positions(length, count, device, importance)
    packed = [pack_layer(k, v, bits, channel, high) for k, v in cache]
    result = tuple(unpack_layer(layer, bits, channel) for layer in packed)
    return result, packed_size(packed), high, bits, channel


def handle_case(model, tokenizer, row, check_reference):
    tokens, start, evidence = tokenize_case(tokenizer, row, model.device)
    torch.cuda.reset_peak_memory_stats()
    initial, prefill_ms = timed(lambda: forward(model, tokens[:, :start], 0, last=1))
    prefix = legacy(initial)
    prefix_peak = torch.cuda.max_memory_allocated()
    suffix_tokens = tokens[:, start:]
    channels = cache_bytes(prefix) // (2 * start)
    budget = budget_bytes(channels, start, 4)
    (importance, full), question_ms = timed(lambda: question_score(model, suffix_tokens, start, prefix))
    reference = full.logits[0, -1].float().log_softmax(-1)
    if not torch.isfinite(reference).all():
        raise FloatingPointError(f"Invalid reference logits on {row['id']}")
    reference_check = None
    if check_reference:
        direct = forward(model, tokens, 0, last=1)
        full_n = torch.arange(start, device=tokens.device)
        retained = tuple((k.index_select(-2, full_n), v.index_select(-2, full_n)) for k, v in prefix)
        repeated = forward(model, suffix_tokens, start, retained, last=1)
        if not torch.isfinite(direct.logits).all():
            raise FloatingPointError("Non-finite direct cache reference")
        split_delta = (direct.logits.float() - full.logits.float()).abs().max().item()
        retained_delta = (repeated.logits.float() - full.logits.float()).abs().max().item()
        if split_delta > .25 or retained_delta > .02:
            raise ValueError(f"Full cache reference disagrees: {split_delta}, {retained_delta}")
        reference_check = dict(split_logit_max_abs=split_delta, retain_all_logit_max_abs=retained_delta)
        del direct, repeated, retained
    answer = tokenizer.encode(row["code"], add_special_tokens=False)
    cfg_eos = model.generation_config.eos_token_id
    eos = set(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])
    scores = {}
    for policy in POLICIES:
        torch.cuda.reset_peak_memory_stats()
        (base, size, high, bits, channel), preprocess_ms = timed(
            lambda: compressed_cache(policy, prefix, importance, budget, channels)
        )
        if policy != "full" and size > budget:
            raise ValueError(f"{policy} exceeds INT4 budget")
        out, suffix_ms = timed(lambda: forward(model, suffix_tokens, start, base, last=1))
        logp = out.logits[0, -1].float().log_softmax(-1)
        logkl = (reference.exp() * (reference - logp)).sum().item()
        teacher_tokens = torch.tensor([answer[:-1]], device=tokens.device)
        logits = out.logits
        if teacher_tokens.numel():
            forced = forward(model, teacher_tokens, tokens.size(1), legacy(out))
            logits = torch.cat((logits, forced.logits), dim=1)
            del forced
        nll = -logits[0].float().log_softmax(-1)[torch.arange(len(answer), device=tokens.device), answer].sum().item()
        if not math.isfinite(nll) or not math.isfinite(logkl):
            raise FloatingPointError(f"Non-finite scores on {row['id']} {policy}")
        update = residual_group_updater(base, bits, channel, high) if bits else None

        (text, count), decode_ms = timed(lambda: decode(model, tokenizer, out, tokens.size(1), eos, update))
        extracted = re.findall(r"(?<!\d)\d{5}(?!\d)", text)
        high_list = high.tolist() if high is not None else None
        errors = None
        if bits:
            errors = []
            for component in (0, 1):
                numerator = sum((x[component].float() - y[component].float()).square().sum().item()
                                for x, y in zip(prefix, base))
                denominator = sum(x[component].float().square().sum().item() for x in prefix)
                errors.append(math.sqrt(numerator / denominator))
        scores[policy] = dict(
            generated=text, generated_tokens=count, exact_match=len(extracted) == 1 and extracted[0] == row["code"],
            answer_nll_sum=nll, answer_tokens=len(answer), reference_to_policy_kl=logkl,
            evidence_fp16_fraction=(len(set(high_list) & set(evidence)) / len(evidence) if high_list is not None else None),
            selected_positions=high_list, relative_kv_l2=errors, packed_context_bytes=size,
            fp16_context_byte_fraction=size / cache_bytes(prefix), preprocessing_ms=preprocess_ms,
            question_scoring_ms=question_ms if policy in ("question","hybrid") else 0,
            suffix_prefill_ms=suffix_ms, decode_ms=decode_ms,
            shared_fp16_suffix_bytes_at_start=2 * channels * suffix_tokens.size(1),
            shared_fp16_suffix_bytes_upper_bound=2 * channels * (suffix_tokens.size(1) + count),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        del base, out, logits
    return dict(
        **{key: row[key] for key in ("id", "target_tokens", "position", "distractors", "code",
                                    "evidence_char_start", "evidence_char_end")},
        encoded_tokens=tokens.size(1), context_tokens=start, evidence_token_positions=evidence,
        context_prefill_ms=prefill_ms, context_prefill_peak_allocated_bytes=prefix_peak,
        int4_budget_bytes=budget, reference_check=reference_check, policies=scores,
    )


def prepare_sweep_cache(model, tokens, chunk_size=128):
    cache = None
    for start in range(0, tokens.size(1), chunk_size):
        cache = legacy(forward(model, tokens[:, start:start + chunk_size], start, cache, last=1))
    return cache


def packing_sweep(model, destination):
    runs = json.loads(destination.read_text()) if destination.exists() else []
    completed = {(row["batch_size"], row["length"], row["policy"]) for row in runs}
    for batch in (1, 8):
        for length in (2048, 4096):
            tokens = torch.randint(100, 10000, (batch, length), device=model.device)
            policies = [name for name in ("full", "int4_token", "int4_channel", "hybrid")
                        if (batch, length, name) not in completed]
            if not policies:
                continue
            torch.cuda.empty_cache()
            raw = prepare_sweep_cache(model, tokens)
            token = tokens[:, :1]
            for policy in policies:
                print(f"Packing sweep: batch={batch} length={length} {policy}", flush=True)
                bits = 2 if policy == "hybrid" else 4
                channel = policy != "int4_token"
                high = None
                if policy == "hybrid":
                    channels = cache_bytes(raw) // (2 * length)
                    budget = budget_bytes(channels, length, 4)
                    count = choose_hybrid(length, channels, len(raw), budget)
                    high = choose_positions(length, count, model.device)
                records = []
                for repetition in range(13):
                    if policy == "full":
                        unpacked, byte_count, enc_ms, dec_ms = raw, cache_bytes(raw), 0., 0.
                    else:
                        packed, enc_ms = timed(lambda: [pack_layer(k, v, bits, channel, high) for k, v in raw])
                        byte_count = packed_size(packed)
                        unpacked, dec_ms = timed(lambda: tuple(unpack_layer(x, bits, channel) for x in packed))
                    _, step_ms = timed(lambda: forward(model, token, length, unpacked, last=1).logits)
                    if repetition >= 3:
                        records.append(dict(encode_ms=enc_ms, unpack_ms=dec_ms, decode_step_ms=step_ms,
                                            decode_with_unpack_ms=dec_ms + step_ms))
                    del unpacked
                    if policy != "full":
                        del packed
                runs.append(dict(batch_size=batch, length=length, policy=policy,
                                 serialized_bytes=byte_count, prefill_chunk_size=128, samples=records))
                write_json(destination, runs)
            del raw
    return runs


def verify_cache_generation(model, tokenizer, cases):
    records = []
    stop = model.generation_config.eos_token_id
    eos = set(stop if isinstance(stop, list) else [stop])
    for row in cases[::10]:
        tokens, boundary, _ = tokenize_case(tokenizer, row, model.device)
        direct = forward(model, tokens, 0, last=1)
        prefix = legacy(forward(model, tokens[:, :boundary], 0, last=1))
        positions = torch.arange(boundary, device=model.device)
        retained = tuple((k.index_select(-2, positions), v.index_select(-2, positions))
                         for k, v in prefix)
        split = forward(model, tokens[:, boundary:], boundary, retained, last=1)
        first, second = direct.logits.float(), split.logits.float()
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise FloatingPointError(row["id"])
        delta = (first - second).abs().max().item()
        a = decode(model, tokenizer, direct, tokens.size(1), eos)
        b = decode(model, tokenizer, split, tokens.size(1), eos)
        records.append(dict(id=row["id"], max_logit_difference=delta,
                            greedy_generation_matches=a == b, direct=a, retain_all=b))
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(6304)
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    path = args.course_data / "data/kv_retrieval/kv_retrieval_cases.jsonl.gz"
    cases = read_jsonl(path)
    assert len(cases) == len({x["id"] for x in cases}) == 180
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, dtype=torch.float16, attn_implementation="sdpa"
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
    manifest = dict(model=MODEL, revision=args.revision, gpu=torch.cuda.get_device_name(),
                    dtype="float16", backend="sdpa", seed=6304, torch_version=torch.__version__,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    residual_grouping="complete groups span context residual and appended positions")
    if (out / "metadata.json").exists() and json.loads((out / "metadata.json").read_text()) != manifest:
        raise ValueError("Existing results have different provenance")
    write_json(out / "metadata.json", manifest)
    if args.reference_only:
        write_json(out / "reference_generation.json", verify_cache_generation(model, tokenizer, cases))
        return
    if not args.sweep_only:
        for i, row in enumerate(cases):
            result = out / f"{row['id']}.json"
            if not result.exists():
                write_json(result, handle_case(model, tokenizer, row, check_reference=i % 30 == 0))
            if (i + 1) % 10 == 0:
                print(f"Retrieval: {i + 1}/{len(cases)}", flush=True)
    if not args.retrieval_only:
        packing_sweep(model, out / "packing_sweep.json")


if __name__ == "__main__":
    main()

