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
