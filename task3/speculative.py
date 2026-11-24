"""Exact speculative decoding and a reproducible joint-distribution test."""
import argparse
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from time import perf_counter


def accepted_prefix(proposals, target, draft, uniform):
    """Return the number of accepted proposals before the first rejection."""
    for i, token in enumerate(proposals):
        p, q = float(target[i][token]), float(draft[i][token])
        if not 0 < q <= 1 or not 0 <= p <= 1:
            raise ValueError("Invalid proposal probability")
        if uniform() >= min(1.0, p / q):
            return i
    return len(proposals)


def resolve(proposals, target, draft, uniform, sample, residual, eos, remaining):
    """Resolve one round, reserving room for the extra token only when available."""
    count = accepted_prefix(proposals, target, draft, uniform)
    emitted = list(proposals[:count])
    terminal = len(emitted) >= remaining or (emitted and emitted[-1] in eos)
    if not terminal:
        distribution = residual(target[count], draft[count]) if count < len(proposals) else target[count]
        emitted.append(sample(distribution))
    return emitted, count


class Toy:
    def __init__(self, seed=6304):
        self.rng = random.Random(seed)
        self.p = ((.73, .15, .12), (.13, .69, .18), (.27, .17, .56))
        self.q = ((.15, .65, .20), (.61, .16, .23), (.14, .64, .22))

    def sample(self, weights):
        return self.rng.choices(range(len(weights)), weights=weights)[0]

    def simulate(self, length=3, gamma=2, incorrect=False, eos=()):
        output = []
        while len(output) < length:
            state = output[-1] if output else 0
            proposals, draft, target = [], [], []
            remaining = length - len(output)
            for _ in range(min(gamma, remaining)):
                draft.append(self.q[state])
                target.append(self.p[state])
                state = self.sample(draft[-1])
                proposals.append(state)
                if state in eos:
                    break
            target.append(self.p[state])
            def residual(p, q):
                return p if incorrect else [max(0., x - y) for x, y in zip(p, q)]
            emitted, _ = resolve(proposals, target, draft, self.rng.random,
                                 self.sample, residual, eos, remaining)
            output.extend(emitted)
            if output[-1] in eos:
                break
        return tuple(output)


def markov_check(trials=100_000):
    toy = Toy()
    categories = list(itertools.product(range(3), repeat=3))
    expected = {s: toy.p[0][s[0]] * toy.p[s[0]][s[1]] * toy.p[s[1]][s[2]] for s in categories}
    assert min(expected.values()) * trials >= 10
    runs = {}
    for incorrect in (False, True):
        toy = Toy()
        draws = Counter(toy.simulate(incorrect=incorrect) for _ in range(trials))
        chi2 = sum((draws[s] - trials * p) ** 2 / (trials * p) for s, p in expected.items())
        # Survival function for chi-square with 26 degrees of freedom.
        x = chi2 / 2
        terms = [-x + k * math.log(x) - math.lgamma(k + 1) for k in range(13)] if x else [0.]
        log_p = max(terms) + math.log(sum(math.exp(y - max(terms)) for y in terms))
        if (log_p < math.log(.01)) != incorrect:
            raise AssertionError(f"Unexpected goodness of fit: incorrect={incorrect}, chi2={chi2}")
        runs["negative_control" if incorrect else "exact"] = dict(
            chi_squared=chi2, log_p=log_p, p_value=math.exp(log_p),
            sequences=[dict(sequence=s, expected_probability=p, empirical_probability=draws[s]/trials)
                       for s, p in expected.items()],
        )

    # Deterministic paths for bonus, residual, EOS, and budget truncation.
    pick = lambda p: max(range(len(p)), key=p.__getitem__)
    residual = lambda p, q: [max(0, x-y) for x, y in zip(p, q)]
    target, draft = [(1., 0.), (0., 1.)], [(1., 0.)]
    assert resolve([0], target, draft, lambda: .5, pick, residual, (), 2) == ([0, 1], 1)
    assert resolve([0], target, draft, lambda: .5, pick, residual, (), 1) == ([0], 1)
    assert resolve([0], target, draft, lambda: .5, pick, residual, {0}, 2) == ([0], 1)
    target = [(0., 1.), (1., 0.)]
    assert resolve([0], target, draft, lambda: .5, pick, residual, (), 2) == ([1], 0)
    return dict(seed=6304, trials=trials, sequence_length=3, degrees_of_freedom=26, **runs)


@dataclass
class ModelState:
    model: object
    cache: object
    probabilities: object

    @classmethod
    def prefill(cls, model, prompt):
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
            raise ValueError("Pass one nonempty, unpadded prompt")
        output = model(prompt, use_cache=True, logits_to_keep=1)
        return cls(model, output.past_key_values, output.logits[0, -1].float().softmax(-1))

    @property
    def length(self):
        return self.cache.get_seq_length()

    def advance(self, tokens):
        import torch
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.model.device).reshape(1, -1)
        result = self.model(ids, past_key_values=self.cache, use_cache=True, logits_to_keep=0)
        self.cache = result.past_key_values
        distributions = result.logits[0].float().softmax(dim=-1)
        self.probabilities = distributions[-1]
        return distributions


def speculative(target, draft, prompt, gamma, cap, eos, seed, measure=False, attention_mask=None):
    """Sample one completion at temperature 1 with no top-p truncation."""
    import torch

    if attention_mask is not None:
        if prompt.ndim != 2 or prompt.shape[0] != 1 or attention_mask.shape != prompt.shape:
            raise ValueError("The attention mask must match one prompt")
        mask = attention_mask.to(prompt.device)
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("The attention mask must be binary")
        prompt = prompt[:, mask[0].bool()]
        if prompt.shape[1] == 0:
            raise ValueError("The prompt must contain an unmasked token")

    if gamma < 1 or cap < 1:
        raise ValueError("Gamma and generation cap must be positive")
    rng = torch.Generator(device=prompt.device).manual_seed(seed)
    uniforms = random.Random(seed)
    def sample(p):
        return torch.multinomial(p, 1, generator=rng).item()

    def tick():
        if measure:
            torch.cuda.synchronize(prompt.device)
        return perf_counter()

    with torch.inference_mode():
        start = tick()
        t = ModelState.prefill(target, prompt)
        t_prefilled = tick()
        d = ModelState.prefill(draft, prompt)
        d_prefilled = tick()
        tokens, rounds = [], []
        while len(tokens) < cap:
            remaining = cap - len(tokens)
            start_length = t.length
            proposals, p_draft = [], []
            round_started = tick()
            for _ in range(min(gamma, remaining)):
                p_draft.append(d.probabilities)
                x = sample(d.probabilities)
                proposals.append(x)
                d.advance([x])
                if x in eos:
                    break
            proposed_at = tick()
            first = t.probabilities
            verified = t.advance(proposals)
            p_target = [first, *verified.unbind(0)]
            verified_at = tick()
            emitted, accepted = resolve(
                proposals, p_target, p_draft, uniforms.random, sample,
                lambda p, q: (p - q).clamp_min(0), eos, remaining,
            )
            resolved_at = tick()
            tokens.extend(emitted)
            terminal = tokens[-1] in eos or len(tokens) >= cap
            advanced = False
            if not terminal:
                if accepted < len(proposals):
                    t.cache.crop(start_length + accepted)
                    d.cache.crop(start_length + accepted)
                if len(emitted) > accepted:
                    t.advance([emitted[-1]])
                    d.advance([emitted[-1]])
                    advanced = True
                if t.length != d.length or t.length != prompt.shape[1] + len(tokens):
                    raise ValueError("Speculative round left the caches on different prefixes")
            advanced_at = tick()
            rounds.append(dict(proposed=len(proposals), accepted=accepted,
                               examined=accepted + int(accepted < len(proposals)),
                               verified=1, target_advance=int(advanced),
                               draft_passes=len(proposals) + int(advanced)))
            if measure:
                rounds[-1].update(
                    draft_ms=1000 * (proposed_at - round_started),
                    verification_ms=1000 * (verified_at - proposed_at),
                    sampling_ms=1000 * (resolved_at - verified_at),
                    cache_ms=1000 * (advanced_at - resolved_at),
                )
            if terminal:
                break
        result = dict(token_ids=tokens, rounds=rounds, eos=tokens[-1] in eos,
                      truncated=len(tokens) == cap and tokens[-1] not in eos)
        if measure:
            end = tick()
            result.update(target_prefill_ms=1000 * (t_prefilled - start),
                          draft_prefill_ms=1000 * (d_prefilled - t_prefilled),
                          decode_ms=1000 * (end - d_prefilled), total_ms=1000 * (end - start))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markov-output", required=True)
    args = parser.parse_args()
    result = markov_check()
    with open(args.markov_output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for label in ("exact", "negative_control"):
        test = result[label]
        print(label, "chi-squared", round(test["chi_squared"], 3), "log p", round(test["log_p"], 3))


if __name__ == "__main__":
    main()

