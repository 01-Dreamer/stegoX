from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Optional


@dataclass(order=True)
class HeapNode:
    freq: float
    order: int
    token: Optional[int] = field(compare=False, default=None)
    left: Optional["HeapNode"] = field(compare=False, default=None)
    right: Optional["HeapNode"] = field(compare=False, default=None)


class HuffmanCoding:
    def __init__(self):
        self.heap: list[HeapNode] = []
        self.codes: dict[int, str] = {}
        self.reverse_mapping: dict[str, int] = {}
        self._counter = count()

    def make_heap_from_array(self, freqs) -> None:
        self.heap.clear()
        for index, freq in enumerate(freqs):
            heapq.heappush(self.heap, HeapNode(float(freq), next(self._counter), index))

    def merge_nodes(self) -> None:
        if not self.heap:
            return
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = HeapNode(
                node1.freq + node2.freq,
                next(self._counter),
                None,
                left=node1,
                right=node2,
            )
            heapq.heappush(self.heap, merged)

    def make_codes(self) -> Optional[HeapNode]:
        if not self.heap:
            return None
        root = heapq.heappop(self.heap)
        self._make_codes_helper(root, "")
        return root

    def _make_codes_helper(self, root: Optional[HeapNode], current_code: str) -> None:
        if root is None:
            return
        if root.token is not None:
            self.codes[root.token] = current_code or "0"
            self.reverse_mapping[current_code or "0"] = root.token
            return
        self._make_codes_helper(root.left, current_code + "0")
        self._make_codes_helper(root.right, current_code + "1")
