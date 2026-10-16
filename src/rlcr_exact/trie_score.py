"""Exact log-probabilities of every candidate confidence string after a prefix, in one forward pass.

The candidates (e.g. "0.8</confidence>" + end of turn) are merged into a token trie and appended after the
prefix. Each trie node attends to the whole prefix, its ancestors and itself, and sits at RoPE position
len(prefix) + depth, so its logits equal those of the candidate's own full sequence. A candidate's
log-probability is the sum, over its tokens, of the log-softmax at the parent position.
"""
from dataclasses import dataclass

import torch


@dataclass
class Trie:
    tokens: list[int]  # token of each node, in insertion order (parents before children)
    parents: list[int]  # parent node index, -1 for children of the prefix
    depths: list[int]  # 0 for children of the prefix
    leaf_paths: list[list[int]]  # node indices of each candidate, in candidate order


def build_trie(candidates: list[list[int]]) -> Trie:
    tokens, parents, depths, children, paths = [], [], [], {}, []
    for cand in candidates:
        node, path = -1, []
        for t in cand:
            key = (node, t)
            if key not in children:
                children[key] = len(tokens)
                tokens.append(t); parents.append(node); depths.append(0 if node < 0 else depths[node] + 1)
            node = children[key]
            path.append(node)
        paths.append(path)
    return Trie(tokens, parents, depths, paths)


def trie_mask(trie: Trie, prefix_len: int, dtype, device) -> torch.Tensor:
    """[1, 1, L + N, L + N] additive mask: causal over the prefix, ancestors-only within the trie."""
    n, total = len(trie.tokens), prefix_len + len(trie.tokens)
    allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
    allowed[:prefix_len, :prefix_len] = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
    allowed[prefix_len:, :prefix_len] = True
    for i in range(n):
        j = i
        while j >= 0:
            allowed[prefix_len + i, prefix_len + j] = True
            j = trie.parents[j]
    mask = torch.zeros(total, total, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def candidate_logps(model, prefix_ids: list[int], trie: Trie, greedy: bool = False, hidden: bool = False):
    """[num_candidates] summed log-probabilities of each candidate after `prefix_ids` (differentiable).

    With greedy=True also returns the candidate that token-level greedy decoding reaches from the prefix,
    or None when the greedy token at some step is not in the trie. With hidden=True also returns (last) the
    forward's hidden states [L + nodes, d]; the first L rows are the prefix's own (trie nodes come after).
    """
    model = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap PEFT
    device = next(model.parameters()).device
    L = len(prefix_ids)
    ids = torch.tensor([prefix_ids + trie.tokens], device=device)
    positions = torch.tensor([list(range(L)) + [L + d for d in trie.depths]], device=device)
    dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
    mask = trie_mask(trie, L, dtype, device)
    states = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).last_hidden_state[0]
    # Logits are needed only where a trie token is predicted: the last prefix position and each parent node.
    read_rows = sorted({L - 1 if p < 0 else L + p for p in trie.parents})
    row_of = {r: k for k, r in enumerate(read_rows)}
    logp = model.lm_head(states[read_rows]).float().log_softmax(-1)
    node_logp = torch.stack([logp[row_of[L - 1 if p < 0 else L + p], t] for t, p in zip(trie.tokens, trie.parents)])
    cand = torch.stack([node_logp[path].sum() for path in trie.leaf_paths])
    extra = (states,) if hidden else ()
    if not greedy:
        return (cand, *extra) if extra else cand
    argmax = logp.argmax(-1).tolist()
    child = {(p, t): i for i, (t, p) in enumerate(zip(trie.tokens, trie.parents))}
    leaf_of = {path[-1]: k for k, path in enumerate(trie.leaf_paths)}
    node = -1
    while True:
        row = L - 1 if node < 0 else L + node
        if row not in row_of:  # a leaf: the candidate is complete
            return (cand, leaf_of.get(node), *extra)
        node = child.get((node, argmax[row_of[row]]))
        if node is None:
            return (cand, None, *extra)


def _ancestor_matrix(trie: Trie) -> torch.Tensor:
    """[N, N] bool, cached on the trie: node i may attend node j when j is i or one of its ancestors."""
    if getattr(trie, "_anc", None) is None:
        n = len(trie.tokens)
        anc = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            j = i
            while j >= 0:
                anc[i, j] = True
                j = trie.parents[j]
        trie._anc = anc
    return trie._anc


def packed_candidate_logps(model, prefixes: list[list[int]], trie: Trie, greedy: bool = False, hidden: bool = False):
    """candidate_logps for several prefixes at once, running their common start a single time.

    Layout of the one sequence: [shared][suffix_1][trie_1][suffix_2][trie_2]...  where shared is the longest
    common prefix (e.g. the prompt that all rollouts of a question share). suffix_b attends the shared part
    and itself causally; trie_b's nodes attend shared, suffix_b and their ancestors. RoPE positions are each
    row's own, so every row's logits equal its candidate_logps. Gradients reach the shared part once, summed
    over rows. Returns [rows, num_candidates] logps, then the greedy picks (list) with greedy=True, then
    per-row hidden states [len(prefix_b), d] with hidden=True.
    """
    model = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap PEFT
    device = next(model.parameters()).device
    dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
    shared = 0
    limit = min(len(p) for p in prefixes) - 1  # every row keeps at least one token of its own
    while shared < limit and all(p[shared] == prefixes[0][shared] for p in prefixes):
        shared += 1
    n = len(trie.tokens)
    anc = _ancestor_matrix(trie)
    tokens, positions, spans = list(prefixes[0][:shared]), list(range(shared)), []
    for p in prefixes:
        s0 = len(tokens)
        tokens += p[shared:]; positions += range(shared, len(p))
        t0 = len(tokens)
        tokens += trie.tokens; positions += [len(p) + d for d in trie.depths]
        spans.append((s0, t0, len(p)))
    total = len(tokens)
    allowed = torch.zeros(total, total, dtype=torch.bool)
    allowed[:shared, :shared] = torch.tril(torch.ones(shared, shared, dtype=torch.bool))
    for s0, t0, _ in spans:
        ls = t0 - s0
        allowed[s0:t0, :shared] = True
        allowed[s0:t0, s0:t0] = torch.tril(torch.ones(ls, ls, dtype=torch.bool))
        allowed[t0:t0 + n, :shared] = True
        allowed[t0:t0 + n, s0:t0] = True
        allowed[t0:t0 + n, t0:t0 + n] = anc
    mask = torch.zeros(total, total, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    states = model.model(input_ids=torch.tensor([tokens], device=device), attention_mask=mask[None, None].to(device),
                         position_ids=torch.tensor([positions], device=device), use_cache=False).last_hidden_state[0]
    # Logits where a trie token is predicted: each row's last prefix position and each of its parent nodes.
    parent_rows = sorted({p for p in trie.parents if p >= 0})
    reads = [[t0 - 1] + [t0 + p for p in parent_rows] for s0, t0, _ in spans]
    logp = model.lm_head(states[torch.tensor(sum(reads, []), device=device)]).float().log_softmax(-1)
    logp = logp.view(len(prefixes), 1 + len(parent_rows), -1)
    slot = {-1: 0} | {p: k + 1 for k, p in enumerate(parent_rows)}
    tok_idx = torch.tensor(trie.tokens, device=device)
    par_idx = torch.tensor([slot[p] for p in trie.parents], device=device)
    node_logp = logp[:, par_idx, tok_idx]  # [rows, nodes]: log p(node token | its parent's context)
    cand = torch.stack([node_logp[:, path].sum(-1) for path in trie.leaf_paths], dim=-1)  # [rows, candidates]
    out = [cand]
    if greedy:
        child = {(p, t): i for i, (t, p) in enumerate(zip(trie.tokens, trie.parents))}
        leaf_of = {path[-1]: k for k, path in enumerate(trie.leaf_paths)}
        argmax = logp.argmax(-1).tolist()
        picks = []
        for b in range(len(prefixes)):
            node = -1
            while True:
                if node >= 0 and node not in slot:  # a leaf: the candidate is complete
                    picks.append(leaf_of.get(node)); break
                node = child.get((node, argmax[b][slot[node]]))
                if node is None:
                    picks.append(None); break
        out.append(picks)
    if hidden:
        out.append([torch.cat([states[:shared], states[s0:t0]]) for s0, t0, _ in spans])
    return out[0] if len(out) == 1 else tuple(out)
