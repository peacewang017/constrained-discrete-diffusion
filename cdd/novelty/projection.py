"""Best-first-search novelty projection operator for CDD."""

import heapq
from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple

import torch


def _flip_cost(top_prob: float, cand_prob: float) -> float:
    return max(top_prob - cand_prob, 0.0)


@dataclass(order=True)
class _SearchNode:
    cost: float
    position: int = field(compare=False)
    tokens: Tuple[int, ...] = field(compare=False)


def best_first_novel_sequence(
    probs: torch.Tensor,
    seen: Set[str],
    decode_fn: Callable[[List[int]], str],
    top_k: int = 5,
    max_nodes: int = 20000,
) -> Tuple[List[int], float, bool]:
    """Find the minimal-flip-cost token sequence whose decode isn't in `seen`.

    top_k bounds the branching factor per position (an unbounded search over
    vocab_size^seq_len is intractable); max_nodes is a fallback safety cap.
    Returns (token_ids, total_cost, found), where found=False means the
    search budget ran out and token_ids is just the plain argmax sequence.
    """
    seq_len, vocab_size = probs.shape
    probs = probs.detach().to('cpu')
    top_probs, top_idx = probs.topk(min(top_k, vocab_size), dim=-1)
    argmax_prob = top_probs[:, 0].tolist()
    argmax_tokens = top_idx[:, 0].tolist()

    heap = [_SearchNode(cost=0.0, position=0, tokens=())]
    nodes_popped = 0

    while heap:
        node = heapq.heappop(heap)
        nodes_popped += 1
        if nodes_popped > max_nodes:
            break

        if node.position == seq_len:
            if decode_fn(list(node.tokens)) not in seen:
                return list(node.tokens), node.cost, True
            continue

        pos = node.position
        for rank in range(top_idx.shape[1]):
            tok = top_idx[pos, rank].item()
            cost = _flip_cost(argmax_prob[pos], top_probs[pos, rank].item())
            heapq.heappush(
                heap,
                _SearchNode(cost=node.cost + cost, position=pos + 1, tokens=node.tokens + (tok,)),
            )

    return argmax_tokens, 0.0, False


class NoveltyProjector:
    """Projects per-step token distributions onto the set of novel sequences."""

    def __init__(
        self,
        seen_smiles: Set[str],
        tokenizer,
        top_k: int = 5,
        max_nodes: int = 20000,
    ):
        self.seen = set(seen_smiles)
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.max_nodes = max_nodes

    def _decode(self, token_ids: List[int]) -> str:
        smi = self.tokenizer.decode(token_ids)
        return smi.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').strip()

    def project(self, q: torch.Tensor) -> torch.Tensor:
        """Project a [batch, seq_len, vocab] distribution onto novel sequences.

        Elements already decoding to a novel sequence are left untouched.
        For the rest, swaps the probability mass between the old argmax
        token and the searched replacement at each flipped position, so the
        replacement becomes argmax with a minimal, targeted perturbation.
        """
        batch_size, seq_len, vocab_size = q.shape
        q_proj = q.clone()

        for b in range(batch_size):
            probs = q[b]
            argmax_tokens = probs.argmax(dim=-1).tolist()
            if self._decode(argmax_tokens) not in self.seen:
                continue

            new_tokens, _, found = best_first_novel_sequence(
                probs, self.seen, self._decode, top_k=self.top_k, max_nodes=self.max_nodes,
            )
            if not found:
                continue

            for pos, (old_tok, new_tok) in enumerate(zip(argmax_tokens, new_tokens)):
                if new_tok == old_tok:
                    continue
                old_p = q_proj[b, pos, old_tok].clone()
                new_p = q_proj[b, pos, new_tok].clone()
                q_proj[b, pos, old_tok] = new_p
                q_proj[b, pos, new_tok] = old_p

        return q_proj

    def mark_seen(self, smiles: str) -> None:
        self.seen.add(smiles)
